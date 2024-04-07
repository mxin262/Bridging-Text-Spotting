# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from DiG.models.transformer_layer import (
    PositionalEncoding, TransformerDecoderLayer, get_pad_mask,
    get_subsequent_mask, DecoupledTransformerDecoderLayer)

def create_decoder(args):
    decoder = TFDecoder(
            num_classes=97,
            max_seq_len=25,
            text_cond_vis=False,)
    return decoder

class BaseDecoder(nn.Module):
    """Base decoder class for text recognition."""

    def __init__(self, init_cfg=None, **kwargs):
        super().__init__()

    def forward_train(self, feat, out_enc, targets, tgt_lens, img_metas, cls_query_attn_maps):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas, cls_query_attn_maps):
        raise NotImplementedError

    def forward(self,
                feat,
                out_enc,
                targets=None,
                tgt_lens=None,
                img_metas=None,
                train_mode=True,
                cls_query_attn_maps=None,
                trg_word_emb=None,
                beam_width=0,):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets, tgt_lens, img_metas, cls_query_attn_maps, trg_word_emb)
        
        if beam_width > 0:
            return self.beam_search(feat, out_enc, img_metas, cls_query_attn_maps, trg_word_emb, beam_width)

        return self.forward_test(feat, out_enc, img_metas, cls_query_attn_maps, trg_word_emb)


class TFDecoder(BaseDecoder):
    """Transformer Decoder block with self attention mechanism.
    Args:
        n_layers (int): Number of attention layers.
        d_embedding (int): Language embedding dimension.
        n_head (int): Number of parallel attention heads.
        d_k (int): Dimension of the key vector.
        d_v (int): Dimension of the value vector.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
        d_inner (int): Hidden dimension of feedforward layers.
        n_position (int): Length of the positional encoding vector. Must be
            greater than ``max_seq_len``.
        dropout (float): Dropout rate.
        num_classes (int): Number of output classes :math:`C`.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        start_idx (int): The index of `<SOS>`.
        padding_idx (int): The index of `<PAD>`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss as specified in
        :obj:`mmocr.models.textrecog.recognizer.EncodeDecodeRecognizer`.
    """

    def __init__(self,
                 n_layers=6,
                 d_embedding=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=256,
                 n_position=200,
                 dropout=0.1,
                 num_classes=97,
                 max_seq_len=40,
                 padding_idx=95,
                 init_cfg=None,
                 text_cond_vis=False,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.padding_idx = padding_idx
        self.max_seq_len = max_seq_len
        self.start_idx = num_classes # the last one is used as the <BOS>.
        self.d_embedding = d_embedding

        self.trg_word_emb = nn.Embedding(
            num_classes + 1, d_embedding)

        self.position_enc = PositionalEncoding(
            d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout, text_cond_vis=text_cond_vis)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        pred_num_class = num_classes  # ignore start_idx
        self.classifier = nn.Linear(d_model, pred_num_class)
        self.num_classes = num_classes
        

    def _attention(self, trg_seq, tgt_lens, src, src_mask=None, cls_query_attn_maps=None, trg_word_emb=None):
        if trg_word_emb is not None:
            trg_embedding = trg_word_emb(trg_seq)
        else:
            trg_embedding = self.trg_word_emb(trg_seq)
        # TODO: debug
        trg_pos_encoded = self.position_enc(trg_embedding)
        tgt = self.dropout(trg_pos_encoded)

        trg_mask = get_pad_mask(
            trg_seq, seq_len=tgt_lens) & get_subsequent_mask(trg_seq)
        output = tgt
        idx = 0
        for dec_layer in self.layer_stack:
            output, attn_maps = dec_layer(
                output,
                src,
                self_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask,
                cls_query_attn_maps=cls_query_attn_maps)
            output = self.adapter[idx](output)
            idx = idx + 1
        output = self.layer_norm(output)

        return output, attn_maps

    def forward_train(self, feat, out_enc, targets, tgt_lens, img_metas, cls_query_attn_maps, trg_word_emb):
        r"""
        Args:
            feat (None): Unused.
            out_enc (Tensor): Encoder output of shape :math:`(N, D_m, H, W)`
                where :math:`D_m` is ``d_model``.
            targets_dict (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.
        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, T, C)`.
        """
        # out_enc: [N, len, C]
        src_mask = None
        targets = targets.to(out_enc.device)
        # Insert the start_idx to the first step
        init_query = torch.full((out_enc.size(0), 1), self.start_idx).type_as(targets)
        query_targets = torch.cat([init_query, targets], dim=-1)[:, :-1]

        attn_output, attn_maps = self._attention(query_targets, tgt_lens, out_enc,
                                                 src_mask=src_mask,
                                                 cls_query_attn_maps=cls_query_attn_maps,
                                                 trg_word_emb=trg_word_emb)
        outputs = self.classifier(attn_output)
        return outputs, attn_maps

    def forward_test(self, feat, out_enc, img_metas, cls_query_attn_maps, trg_word_emb):
        src_mask = None
        init_target_seq = torch.zeros((out_enc.size(0), self.max_seq_len + 1),
                                       device=out_enc.device,
                                       dtype=torch.long)
        # bsz * seq_len
        init_target_seq[:, 0] = self.start_idx
        outputs, attn_maps_list = [], []
        for step in range(0, self.max_seq_len):
            cur_tgt_lens = torch.full((out_enc.size(0),), step+1, device=out_enc.device, dtype=torch.long)
            decoder_output, attn_maps = self._attention(
                init_target_seq, cur_tgt_lens, out_enc,
                src_mask=src_mask,
                cls_query_attn_maps=cls_query_attn_maps,
                trg_word_emb=trg_word_emb)
            # bsz * seq_len * 512
            step_result = F.softmax(
                self.classifier(decoder_output[:, step, :]), dim=-1)
            # bsz * num_classes
            outputs.append(step_result)
            attn_maps_list.append(attn_maps[:, step, :])
            _, step_max_index = torch.max(step_result, dim=-1)
            init_target_seq[:, step + 1] = step_max_index

        outputs = torch.stack(outputs, dim=1)
        attn_maps = torch.stack(attn_maps_list, dim=1)

        return outputs, attn_maps

    def beam_search(self, feat, out_enc, img_metas, cls_query_attn_maps, trg_word_emb, beam_width, eos=94):

        def _inflate(tensor, times, dim):
            repeat_dims = [1] * tensor.dim()
            repeat_dims[dim] = times
            return tensor.repeat(*repeat_dims)

        B, N, C = out_enc.size()
        inflated_out_enc = out_enc.unsqueeze(1).permute((1,0,2,3)).repeat((beam_width,1,1,1)).permute((1,0,2,3)).reshape(-1, N, C)

        # Initialize the input vector
        src_mask = None
        init_target_seq = torch.zeros((B * beam_width, self.max_seq_len + 1),
                                       device=out_enc.device,
                                       dtype=torch.long)
        init_target_seq[:, 0] = self.start_idx
        pos_index = (torch.Tensor(range(B)) * beam_width).long().to(out_enc.device).view(-1, 1)

        # Initialize the scores
        seq_scores = torch.zeros((B * beam_width, 1), device=out_enc.device)
        seq_scores.fill_(-float('Inf'))
        seq_scores.index_fill_(0, torch.Tensor([i * beam_width for i in range(0, B)]).long().to(out_enc.device), 0.0)

        # Store decisions for backtracking
        stored_scores          = list()
        stored_predecessors    = list()
        stored_emitted_symbols = list()

        for step in range(0, self.max_seq_len):
            cur_tgt_lens = torch.full((B * beam_width,), step+1, device=out_enc.device, dtype=torch.long)
            decoder_output, attn_maps = self._attention(
                init_target_seq, cur_tgt_lens, inflated_out_enc,
                src_mask=src_mask,
                cls_query_attn_maps=cls_query_attn_maps,
                trg_word_emb=trg_word_emb)
            log_softmax_output = F.log_softmax(
                self.classifier(decoder_output[:, step, :]), dim=-1)
            seq_scores = _inflate(seq_scores, self.num_classes, 1)
            seq_scores += log_softmax_output
            scores, candidates = seq_scores.view(B, -1).topk(beam_width, dim=1)

            step_max_index = (candidates % self.num_classes).view(B * beam_width)
            seq_scores = scores.view(B * beam_width, 1)

            predecessors = (candidates // self.num_classes + pos_index.expand_as(candidates)).view(B * beam_width, 1)

            stored_scores.append(seq_scores.clone())
            eos_indices = step_max_index.view(-1, 1).eq(eos)
            if eos_indices.nonzero().dim() > 0:
                seq_scores.masked_fill_(eos_indices, -float('inf'))
            
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(step_max_index)

            init_target_seq[:, step + 1] = step_max_index
        
        # Do backtracking to return the optimal values
        #====== backtrak ======#
        # Initialize return variables given different types
        p = list()
        l = [[self.max_seq_len] * beam_width for _ in range(B)]  # Placeholder for lengths of top-k sequences

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = stored_scores[-1].view(B, beam_width).topk(beam_width)
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * B  # the number of EOS found
                                            # in the backward loop below for each batch
        t = self.max_seq_len - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + pos_index.expand_as(sorted_idx)).view(B * beam_width)
        while t >= 0:
            # Re-order the variables with the back pointer
            current_symbol = stored_emitted_symbols[t].index_select(0, t_predecessors)
            t_predecessors = stored_predecessors[t].index_select(0, t_predecessors).squeeze()
            eos_indices = stored_emitted_symbols[t].eq(eos).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / beam_width)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = beam_width - (batch_eos_found[b_idx] % beam_width) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * beam_width + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = stored_predecessors[t][idx[0]]
                    current_symbol[res_idx] = stored_emitted_symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = stored_scores[t][idx[0], [0]]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(beam_width)
        for b_idx in range(B):
            l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

        re_sorted_idx = (re_sorted_idx + pos_index.expand_as(re_sorted_idx)).view(B*beam_width)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        p = [step.index_select(0, re_sorted_idx).view(B, beam_width, -1) for step in reversed(p)]
        p = torch.cat(p, -1)[:,0,:]
        return p, torch.ones_like(p)        


class DecoupledTFDecoder(TFDecoder):
    def __init__(self,
                 n_layers=6,
                 d_embedding=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=256,
                 n_position=200,
                 dropout=0.1,
                 num_classes=97,
                 max_seq_len=40,
                 padding_idx=95,
                 init_cfg=None,
                 text_cond_vis=False,
                 **kwargs):
        super().__init__(
                 n_layers=n_layers,
                 d_embedding=d_embedding,
                 n_head=n_head,
                 d_k=d_k,
                 d_v=d_v,
                 d_model=d_model,
                 d_inner=d_inner,
                 n_position=n_position,
                 dropout=dropout,
                 num_classes=num_classes,
                 max_seq_len=max_seq_len,
                 padding_idx=padding_idx,
                 init_cfg=init_cfg,
                 text_cond_vis=text_cond_vis,)

        self.order_enc = nn.Embedding(max_seq_len+1, d_embedding) # for evaluation, an extra position is inserted.
        self.order_dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            DecoupledTransformerDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def _attention(self, trg_seq, tgt_lens, src, src_mask=None, cls_query_attn_maps=None):
        trg_embedding = self.trg_word_emb(trg_seq)
        trg_pos_encoded = self.position_enc(trg_embedding)
        tgt = self.dropout(trg_pos_encoded)

        orders = torch.arange(trg_seq.size(1)).unsqueeze(0).expand(trg_seq.size(0), -1).to(trg_seq.device)
        order_embedding = self.order_enc(orders)
        order_embedding = self.order_dropout(order_embedding)

        trg_mask = get_pad_mask(
            trg_seq, seq_len=tgt_lens) & get_subsequent_mask(trg_seq)
        output = tgt
        for dec_layer in self.layer_stack:
            output, attn_maps = dec_layer(
                output,
                src,
                self_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask,
                cls_query_attn_maps=cls_query_attn_maps,
                order_embed=order_embedding)
        output = self.layer_norm(output)

        return output, attn_maps