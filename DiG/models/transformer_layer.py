# Copyright (c) OpenMMLab. All rights reserved.
"""This code is from https://github.com/jadore801120/attention-is-all-you-need-
pytorch."""
import numpy as np
import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    """"""

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 mask_value=0,
                 act_layer=nn.GELU,):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            qkv_bias=qkv_bias,
            dropout=dropout,
            mask_value=mask_value)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_layer=act_layer)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x, x, x, mask)
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)

        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 mask_value=0,
                 act_layer=nn.GELU,
                 text_cond_vis=False,):
        super().__init__()
        self.text_cond_vis = text_cond_vis
        self.self_attn = MultiHeadAttention()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            qkv_bias=qkv_bias,
            mask_value=mask_value)
        if text_cond_vis:
            self.enc_attn = TextConditionalMultiHeadAttention(
                n_head,
                d_model,
                d_k,
                d_v,
                dropout=dropout,
                qkv_bias=qkv_bias,
                mask_value=mask_value)
        else:
            self.enc_attn = MultiHeadAttention(
                n_head,
                d_model,
                d_k,
                d_v,
                dropout=dropout,
                qkv_bias=qkv_bias,
                mask_value=mask_value)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_layer=act_layer)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                dec_enc_attn_mask=None,
                cls_query_attn_maps=None,):
        self_attn_in = self.norm1(dec_input)
        self_attn_out = self.self_attn(self_attn_in, self_attn_in,
                                       self_attn_in, self_attn_mask)
        enc_attn_in = dec_input + self_attn_out

        enc_attn_q = self.norm2(enc_attn_in)
        enc_attn_out, attn_maps = self.enc_attn(enc_attn_q, enc_output,
                                                enc_output, dec_enc_attn_mask,
                                                return_attn_map=True,
                                                cls_query_attn_maps=cls_query_attn_maps)

        mlp_in = enc_attn_in + enc_attn_out
        mlp_out = self.mlp(self.norm3(mlp_in))
        out = mlp_in + mlp_out

        return out, attn_maps


class DecoupledTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 mask_value=0,
                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            qkv_bias=qkv_bias,
            mask_value=mask_value)
        self.enc_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            qkv_bias=qkv_bias,
            mask_value=mask_value)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_layer=act_layer)

        self.mlp_order2cls_attn = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, 97),
            nn.Dropout(dropout),
        )
        dim_v = n_head * d_v
        self.new_linear_v = nn.Linear(dim_v, dim_v, bias=qkv_bias)

    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                dec_enc_attn_mask=None,
                cls_query_attn_maps=None,
                order_embed=None):
        # semantics
        self_attn_in = self.norm1(dec_input)
        self_attn_out = self.self_attn(self_attn_in, self_attn_in,
                                       self_attn_in, self_attn_mask)
        enc_attn_in = dec_input + self_attn_out
        
        # vision
        order_embed_q = self.norm2(order_embed)
        order_attn_out, attn_maps = self.enc_attn(order_embed_q, enc_output,
                                                  enc_output, dec_enc_attn_mask,
                                                  return_attn_map=True,)
        order_attn_out = order_attn_out + order_embed
        order2cls_attn = torch.softmax(self.mlp_order2cls_attn(order_attn_out), dim=-1) # [b, len_order, num_class]

        b, nc = cls_query_attn_maps.size()[:2]
        cls_query_attn_maps = cls_query_attn_maps.view(b, nc, -1) # [b, num_class, len_k]

        order_attn = torch.matmul(order2cls_attn, cls_query_attn_maps) # [b, len_order, len_k]
        order_attn += attn_maps
        enc_output = self.new_linear_v(enc_output)
        enc_attn_out = torch.matmul(order_attn, enc_output)

        # semantics and vision fusion
        mlp_in = enc_attn_in + enc_attn_out
        mlp_out = self.mlp(self.norm3(mlp_in))
        out = mlp_in + mlp_out

        return out, attn_maps


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self,
                 n_head=8,
                 d_model=512,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 mask_value=0):
        super().__init__()

        self.mask_value = mask_value

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.scale = d_k**-0.5

        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v

        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)

        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)

        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)

        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, return_attn_map=False, cls_query_attn_maps=None):
        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()

        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        logits = torch.matmul(q, k) * self.scale

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            logits = logits.masked_fill(mask == self.mask_value, float('-inf'))
        weights = logits.softmax(dim=-1)

        if cls_query_attn_maps is not None:
            # [b, num_classes, *path_shape]
            b, nc = cls_query_attn_maps.size()[:2]
            # cls_query_attn_maps = cls_query_attn_maps.view(b, 1, 1, nc, -1) # [b, 1, 1, num_class, len_k]
            # weights = (weights.unsqueeze(3) * cls_query_attn_maps).sum(3) # [b, n_head, len_q, len_k]
            cls_query_attn_maps = cls_query_attn_maps.view(b, 1, nc, -1) # [b, 1, num_class, len_k]
            squeeze_weights = weights.mean(1, keepdim=True).transpose(1, 2) # [b, len_q, 1, len_k]
            weights = (squeeze_weights * cls_query_attn_maps).sum(2).unsqueeze(1).expand_as(weights) # [b, n_head, len_q, len_k]

        vis_attn_maps = weights.mean(1) # [b, len_q, len_k]
        weights = self.attn_drop(weights)

        attn_out = torch.matmul(weights, v).transpose(1, 2)
        attn_out = attn_out.reshape(batch_size, len_q, self.dim_v)
        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)

        if return_attn_map:
            return attn_out, vis_attn_maps
        else:
            return attn_out


class TextConditionalMultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self,
                 n_head=8,
                 d_model=512,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 mask_value=0):
        super().__init__()

        self.mask_value = mask_value

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.scale = d_k**-0.5

        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v

        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)

        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)

        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)

        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # query-conditional visual feature enhancement
        self.gamma_decode = nn.Linear(self.dim_k, 2 * self.dim_k)
        self.vis_proj = nn.Linear(self.dim_k, self.dim_k)
        self.vis_norm = nn.LayerNorm(self.dim_k)
        self.vis_cond_norm = nn.LayerNorm(self.dim_k)

        # self.concat_reduce = nn.Linear(2 * self.dim_k, self.dim_k)

    def forward(self, q, k, v, mask=None, return_attn_map=False, cls_query_attn_maps=None):
        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()

        # query insert to visual feature
        film_param = self.gamma_decode(q)
        film_param = film_param.view(q.size(0), q.size(1), 1, 2*self.dim_k).repeat(1, 1, len_k, 1)
        gammas, betas = torch.split(film_param, self.dim_k, dim=3) # [B, len_q, len_k, dim_k]
        gammas, betas = torch.tanh(gammas), torch.tanh(betas)

        cond_k = self.vis_norm(self.vis_proj(k)).unsqueeze(1) # [B, 1, len_k, dim_k]
        cond_k = (gammas * cond_k) + betas # [B, len_q, len_k, dim_k]
        cond_k = k.unsqueeze(1) + self.vis_cond_norm(cond_k) # v1
        # cond_k = self.vis_cond_norm(k.unsqueeze(1) + cond_k) # v2 [B, len_q, len_k, dim_k]
        # cond_k = self.vis_cond_norm(self.concat_reduce(torch.cat((k.unsqueeze(1).repeat(1, len_q, 1, 1), cond_k), dim=-1)))

        # TODO: query-wised cross-attention

        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.linear_k(cond_k).view(batch_size, len_q, len_k, self.n_head, self.d_k)
        v = self.linear_v(cond_k).view(batch_size, len_q, len_k, self.n_head, self.d_v)

        q = q.permute(0, 2, 1, 3).unsqueeze(3) # [B, n_head, len_q, 1, d_k]
        k = k.permute(0, 3, 1, 4, 2) # [B, n_head, len_q, d_k, len_k]
        v = v.permute(0, 3, 1, 2, 4) # [B, n_head, len_q, len_k, d_k]

        logits = torch.matmul(q, k) * self.scale # [B, n_head, len_q, 1, len_k]

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            logits = logits.masked_fill(mask == self.mask_value, float('-inf'))
        weights = logits.softmax(dim=-1)

        if cls_query_attn_maps is not None:
            # [b, num_classes, *path_shape]
            b, nc = cls_query_attn_maps.size()[:2]
            # cls_query_attn_maps = cls_query_attn_maps.view(b, 1, 1, nc, -1) # [b, 1, 1, num_class, len_k]
            # weights = (weights.unsqueeze(3) * cls_query_attn_maps).sum(3) # [b, n_head, len_q, len_k]
            cls_query_attn_maps = cls_query_attn_maps.view(b, 1, nc, -1) # [b, 1, num_class, len_k]
            squeeze_weights = weights.mean(1, keepdim=True).transpose(1, 2) # [b, len_q, 1, len_k]
            weights = (squeeze_weights * cls_query_attn_maps).sum(2).unsqueeze(1).expand_as(weights) # [b, n_head, len_q, len_k]

        vis_attn_maps = weights.squeeze(3).mean(1) # [b, len_q, len_k]
        weights = self.attn_drop(weights)

        attn_out = torch.matmul(weights, v).squeeze(3).transpose(1, 2)
        attn_out = attn_out.reshape(batch_size, len_q, self.dim_v)
        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)

        if return_attn_map:
            return attn_out, vis_attn_maps
        else:
            return attn_out


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module."""

    def __init__(self, d_in, d_hid, dropout=0.1, act_layer=nn.GELU):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid=512, n_position=200):
        super().__init__()

        # Not a parameter
        self.register_buffer(
            'position_table',
            self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = torch.Tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        self.device = x.device
        return x + self.position_table[:, :x.size(1)].clone().detach()


# def get_pad_mask(seq, pad_idx):
#     return (seq != pad_idx).unsqueeze(-2)

def get_pad_mask(seq, seq_len):
    batch_size, max_len = seq.size()[:2]
    tmp1 = seq.new_zeros(max_len)
    tmp1[:max_len] = torch.arange(0, max_len, dtype=seq.dtype).unsqueeze(0)

    tmp1 = tmp1.expand(batch_size, max_len)
    tmp2 = seq_len.type(tmp1.type())
    tmp2 = tmp2.unsqueeze(1).expand(batch_size, max_len)
    # [N, max_len]
    mask = torch.lt(tmp1, tmp2)
    return mask.unsqueeze(-2)


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    len_s = seq.size(1)
    subsequent_mask = 1 - torch.triu(
        torch.ones((len_s, len_s), device=seq.device), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).bool()
    return subsequent_mask