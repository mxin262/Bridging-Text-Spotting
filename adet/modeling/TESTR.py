from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess as d2_postprocesss
from detectron2.structures import ImageList, Instances

from adet.layers.pos_encoding import PositionalEncoding2D
from adet.modeling.testr.losses import SetCriterion
from adet.modeling.testr.matcher import build_matcher
from adet.modeling.testr.models import TESTR
from adet.utils.misc import NestedTensor, box_xyxy_to_cxcywh
from DiG.models.model_builder import RecModel
from DiG.loss import SeqCrossEntropyLoss
from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes
from functools import partial
from .adapter import Adapter
from .bridge import Bridge

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    In addition to the post processing of detectron2, we add scalign for 
    bezier control points.
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    # results = d2_postprocesss(results, output_height, output_width, mask_threshold)

    # scale bezier points
    if results.has("beziers"):
        beziers = results.beziers
        # scale and clip in place
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y

    if results.has("polygons"):
        polygons = results.polygons
        polygons[:, 0::2] *= scale_x
        polygons[:, 1::2] *= scale_y

    return results

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

@META_ARCH_REGISTRY.register()
class TransformerDetector(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        d2_backbone = MaskedBackbone(cfg)
        N_steps = cfg.MODEL.TRANSFORMER.HIDDEN_DIM // 2
        self.test_score_threshold = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON
        backbone = Joiner(d2_backbone, PositionalEncoding2D(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels
        self.testr = TESTR(cfg, backbone)
        
        self.recognizer = RecModel(cfg)
        checkpoint = torch.load("DiG/checkpoint-9.pth", map_location='cpu')
        self.recognizer.load_state_dict(checkpoint["model"], False)

        for name, p in self.testr.named_parameters():
            p.requires_grad = False
            if "transformer.encoder" in name:
                if "norm" in name:
                    p.requires_grad = True

        self.testr.transformer.encoder.adapter = nn.ModuleList([Adapter(256) for i in range(6)])
        
        for name, p in self.recognizer.named_parameters():
            p.requires_grad = False
            if "norm" in name:
                p.requires_grad = True

        self.recognizer.decoder.adapter = nn.ModuleList([Adapter(512) for i in range(6)])

        self.box_pooler = ROIPooler(
            output_size=(32, 128),
            scales=(1.0,),
            sampling_ratio=2,
            pooler_type="ROIAlignV2",
        )

        self.box_pooler2 = ROIPooler(
            output_size=(32, 128),
            scales=(0.125, 0.0625, 0.03125),
            sampling_ratio=2,
            pooler_type="ROIAlignV2",
        )

        self.bridge = Bridge(img_size=(32, 128), patch_size=4, embed_dim=384, depth=1, num_heads=8, mlp_ratio=4, qkv_bias=True,
                                                norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.bridge.patch_embed.proj = zero_module(nn.Conv2d(256, 384, kernel_size=4, stride=4))

        self.rec_criterion = SeqCrossEntropyLoss()

        box_matcher, point_matcher = build_matcher(cfg)
        
        loss_cfg = cfg.MODEL.TRANSFORMER.LOSS
        weight_dict = {'loss_ce': loss_cfg.POINT_CLASS_WEIGHT, 'loss_ctrl_points': loss_cfg.POINT_COORD_WEIGHT, 'loss_texts': loss_cfg.POINT_TEXT_WEIGHT}
        enc_weight_dict = {'loss_bbox': loss_cfg.BOX_COORD_WEIGHT, 'loss_giou': loss_cfg.BOX_GIOU_WEIGHT, 'loss_ce': loss_cfg.BOX_CLASS_WEIGHT}
        if loss_cfg.AUX_LOSS:
            aux_weight_dict = {}
            # decoder aux loss
            for i in range(cfg.MODEL.TRANSFORMER.DEC_LAYERS - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()})
            # encoder aux loss
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in enc_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        enc_losses = ['labels', 'boxes']
        dec_losses = ['labels', 'ctrl_points',]

        self.criterion = SetCriterion(self.testr.num_classes, box_matcher, point_matcher,
                                      weight_dict, enc_losses, dec_losses, self.testr.num_ctrl_points, 
                                      focal_alpha=loss_cfg.FOCAL_ALPHA, focal_gamma=loss_cfg.FOCAL_GAMMA)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        rec_mean = rec_std = torch.tensor(0.5)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.normalizer2 = lambda x: (x/255 - rec_mean) / rec_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        tmp_images = [self.normalizer2(x["image"].to(self.device)) for x in batched_inputs]
        tmp_images = ImageList.from_tensors(tmp_images)
        return images, tmp_images

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        images, tmp_images = self.preprocess_image(batched_inputs)
        output, encoder_feat = self.testr(images)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            target_boxes = [Boxes(t['rec_boxes']) for t in targets]
            target_text = torch.cat([t['texts'] for t in targets], dim=0)
            target_len = (target_text!=95).sum(1)
            rec_img = self.box_pooler([tmp_images.tensor], target_boxes)
            rec_feat = self.box_pooler2(encoder_feat[:-1], target_boxes)

            rec_imgs = (rec_img, target_text, target_len, rec_feat)
            rec_outputs = self.recognizer(rec_imgs, self.bridge)
            rec_outputs = rec_outputs[0]

            loss_rec = self.rec_criterion(rec_outputs, target_text, target_len)
            # compute the loss
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            loss_dict["loss_texts"] = loss_rec
            return loss_dict
        else:
            ctrl_point_cls = output["pred_logits"]
            ctrl_point_coord = output["pred_ctrl_points"]
            results = self.inference(ctrl_point_cls, ctrl_point_coord, images.image_sizes)
            boxes_results = []
            batch_len = []
            for i in results:
                boxes_results.append(Boxes(i.boxes))
                batch_len.append(i.boxes.shape[0])
            batch_len = torch.as_tensor(batch_len)
            rec_img = self.box_pooler([tmp_images.tensor], boxes_results)
            rec_feat = self.box_pooler2(encoder_feat[:-1], boxes_results)
            if rec_img.shape[0]>0:
                rec_imgs = (rec_img, None, None, rec_feat)
                output = self.recognizer(rec_imgs, self.bridge)
                if isinstance(output, tuple):
                    if len(output) == 3:
                        output, ctc_rec_score, _ = output
                        cls_logit = None
                    elif len(output) == 2:
                        output, _ = output
                        cls_logit = None
                    else:
                        output, cls_logit, _, _ = output
                else:
                    cls_logit = None
                _, pred_ids = output.max(-1)
                pred_ids = pred_ids.split(batch_len)
                rec_scores = output.split(batch_len)
            else:
                pred_ids = torch.tensor([])
                rec_scores =  torch.tensor([])
            processed_results = []
            for results_per_image, input_per_image, image_size, pred_id, rec_score in zip(results, batched_inputs, images.image_sizes, pred_ids, rec_scores):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                r.recs = pred_id
                r.rec_scores = rec_score
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            raw_ctrl_points = targets_per_image.polygons if self.use_polygon else targets_per_image.beziers
            gt_ctrl_points = raw_ctrl_points.reshape(-1, self.testr.num_ctrl_points, 2) / torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            gt_text = targets_per_image.text
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "ctrl_points": gt_ctrl_points, "texts": gt_text, "rec_boxes": targets_per_image.gt_boxes.tensor})
        return new_targets

    def inference(self, ctrl_point_cls, ctrl_point_coord, image_sizes):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []

        # text_pred = torch.softmax(text_pred, dim=-1)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        for scores_per_image, labels_per_image, ctrl_point_per_image, image_size in zip(
            scores, labels, ctrl_point_coord, image_sizes
        ):
            selector = scores_per_image >= self.test_score_threshold
            scores_per_image = scores_per_image[selector]
            labels_per_image = labels_per_image[selector]
            ctrl_point_per_image = ctrl_point_per_image[selector]
            # text_per_image = text_per_image[selector]
            result = Instances(image_size)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            # result.rec_scores = text_per_image
            ctrl_point_per_image[..., 0] *= image_size[1]
            ctrl_point_per_image[..., 1] *= image_size[0]
            if self.use_polygon:
                result.polygons = ctrl_point_per_image.flatten(1)
                if len(result.polygons):
                    maxx = result.polygons[:,::2].max(1)[0]
                    minx = result.polygons[:,::2].min(1)[0]
                    maxy = result.polygons[:,1::2].max(1)[0]
                    miny = result.polygons[:,1::2].min(1)[0]
                    boxes = torch.stack((minx,miny,maxx,maxy),1)
                    result.boxes = boxes
                else:
                    result.boxes = result.polygons.reshape(-1,4) 
            else:
                result.beziers = ctrl_point_per_image.flatten(1)
            # _, topi = text_per_image.topk(1)
            # result.recs = topi.squeeze(-1)
            results.append(result)
        return results