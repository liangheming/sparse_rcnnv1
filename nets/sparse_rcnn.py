import torch
import math
from torch import nn
from nets.pooling import MultiScaleRoIAlign
from losses.sparse_rcnn_loss import BoxCoder, SparseRCNNLoss
from nets import resnet
from nets.common import FrozenBatchNorm2d


class FPN(nn.Module):
    def __init__(self, c2, c3, c4, c5, inner_channel=256, bias=False):
        super(FPN, self).__init__()
        self.c2_to_f2 = nn.Conv2d(c2, inner_channel, 1, 1, 0, bias=bias)
        self.c3_to_f3 = nn.Conv2d(c3, inner_channel, 1, 1, 0, bias=bias)
        self.c4_to_f4 = nn.Conv2d(c4, inner_channel, 1, 1, 0, bias=bias)
        self.c5_to_f5 = nn.Conv2d(c5, inner_channel, 1, 1, 0, bias=bias)

        self.p2_out = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1, bias=bias)
        self.p3_out = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1, bias=bias)
        self.p4_out = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1, bias=bias)
        self.p5_out = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1, bias=bias)

    def forward(self, c2, c3, c4, c5):
        latent_2 = self.c2_to_f2(c2)
        latent_3 = self.c3_to_f3(c3)
        latent_4 = self.c4_to_f4(c4)
        latent_5 = self.c5_to_f5(c5)

        f4 = latent_4 + nn.UpsamplingBilinear2d(size=(latent_4.shape[2:]))(latent_5)
        f3 = latent_3 + nn.UpsamplingBilinear2d(size=(latent_3.shape[2:]))(f4)
        f2 = latent_2 + nn.UpsamplingBilinear2d(size=(latent_2.shape[2:]))(f3)
        p2 = self.p2_out(f2)
        p3 = self.p3_out(f3)
        p4 = self.p4_out(f4)
        p5 = self.p5_out(latent_5)
        return p2, p3, p4, p5


class DynamicConv(nn.Module):
    def __init__(self, in_channel, inner_channel, pooling_resolution, activation=nn.ReLU, **kwargs):
        super(DynamicConv, self).__init__()
        self.in_channel = in_channel
        self.inner_channel = inner_channel
        self.param_num = in_channel * inner_channel
        self.dynamic_layer = nn.Linear(in_channel, 2 * in_channel * inner_channel)

        self.norm1 = nn.LayerNorm(inner_channel)
        self.norm2 = nn.LayerNorm(in_channel)
        self.activation = activation(**kwargs)
        flatten_dim = in_channel * pooling_resolution ** 2
        self.out_layer = nn.Linear(flatten_dim, in_channel)
        self.norm3 = nn.LayerNorm(in_channel)

    def forward(self, x: torch.Tensor, param_x: torch.Tensor):
        """
        :param x:  [pooling_resolution**2,N * nr_boxes,in_channel]
        :param param_x: [N * nr_boxes, in_channel]
        :return:
        """
        # [N*nr_boxes,49,in_channel]
        x = x.permute(1, 0, 2)
        # [N*nr_boxes, 2*in_channel * inner_channel]
        params = self.dynamic_layer(param_x)
        # [N*nr_boxes,in_channel,inner_channel]
        param1 = params[:, :self.param_num].view(-1, self.in_channel, self.inner_channel)

        x = torch.bmm(x, param1)
        x = self.norm1(x)
        x = self.activation(x)

        # [N*nr_boxes,inner_channel,in_channel]
        param2 = params[:, self.param_num:].view(-1, self.inner_channel, self.in_channel)

        x = torch.bmm(x, param2)
        x = self.norm2(x)
        x = self.activation(x)

        x = x.flatten(1)
        x = self.out_layer(x)
        x = self.norm3(x)
        x = self.activation(x)
        return x


class RCNNHead(nn.Module):
    def __init__(self,
                 in_channel,
                 inner_channel,
                 num_cls,
                 dim_feedforward=2048,
                 nhead=8,
                 dropout=0.1,
                 pooling_resolution=7,
                 activation=nn.ReLU,
                 cls_tower_num=1,
                 reg_tower_num=3, **kwargs):
        super(RCNNHead, self).__init__()
        self.self_attn = nn.MultiheadAttention(in_channel, nhead, dropout=dropout)

        self.inst_interact = DynamicConv(in_channel,
                                         inner_channel,
                                         pooling_resolution=pooling_resolution,
                                         activation=activation,
                                         **kwargs)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_channel, dim_feedforward),
            activation(**kwargs),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, in_channel)
        )

        self.norm1 = nn.LayerNorm(in_channel)
        self.norm2 = nn.LayerNorm(in_channel)
        self.norm3 = nn.LayerNorm(in_channel)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.cls_tower = list()
        for _ in range(cls_tower_num):
            self.cls_tower.append(nn.Linear(in_channel, in_channel, False))
            self.cls_tower.append(nn.LayerNorm(in_channel))
            self.cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*self.cls_tower)

        self.reg_tower = list()
        for _ in range(reg_tower_num):
            self.reg_tower.append(nn.Linear(in_channel, in_channel, False))
            self.reg_tower.append(nn.LayerNorm(in_channel))
            self.reg_tower.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*self.reg_tower)

        self.class_logits = nn.Linear(in_channel, num_cls)
        self.bboxes_delta = nn.Linear(in_channel, 4)
        self.box_coder = BoxCoder()

    def forward(self, x: torch.Tensor, params_x: torch.Tensor, boxes: torch.Tensor):
        """
        :param x: [N * nr_boxes,in_channel,pooling_resolution,pooling_resolution]
        :param params_x: [N, nr_boxes, in_channel]
        :param boxes:[N * nr_boxes,4]
        :return:
        """
        nxp, c, _, _ = x.shape
        n, p, _ = params_x.shape
        # [res**2,N * nr_boxes,in_channel]
        x = x.view(nxp, c, -1).permute(2, 0, 1)
        # [nr_boxes, N, in_channel]
        params_x = params_x.permute(1, 0, 2)
        params_attn = self.self_attn(params_x, params_x, value=params_x)[0]
        params_x = self.norm1(params_x + self.dropout1(params_attn))

        params_x = params_x.permute(1, 0, 2).contiguous().view(-1, params_x.size(2))
        # [N*nr_boxes,in_channel]
        param_intersect = self.inst_interact(x, params_x)
        params_x = self.norm2(params_x + self.dropout2(param_intersect))

        param_feedforward = self.feed_forward(params_x)
        # [N*nr_boxes,in_channel]
        out = self.norm3(params_x + self.dropout3(param_feedforward))
        cls_tower = self.cls_tower(out)
        reg_tower = self.reg_tower(out)
        cls_out = self.class_logits(cls_tower)
        reg_delta = self.bboxes_delta(reg_tower)
        pred_bboxes = self.box_coder.decoder(reg_delta, boxes)
        return cls_out.view(n, p, -1), pred_bboxes.view(n, p, -1), out.view(n, p, -1)


class DynamicHead(nn.Module):
    def __init__(self,
                 in_channel,
                 inner_channel,
                 num_cls,
                 dim_feedforward=2048,
                 nhead=8,
                 dropout=0.1,
                 pooling_resolution=7,
                 activation=nn.ReLU,
                 cls_tower_num=1,
                 reg_tower_num=3,
                 num_heads=6,
                 return_intermediate=True,
                 **kwargs):
        super(DynamicHead, self).__init__()
        self.pooling_layer = MultiScaleRoIAlign(
            ['p2', 'p3', 'p4', 'p5'],
            output_size=pooling_resolution,
            sampling_ratio=2
        )
        self.heads = list()
        for _ in range(num_heads):
            self.heads.append(RCNNHead(in_channel,
                                       inner_channel,
                                       num_cls,
                                       dim_feedforward,
                                       nhead,
                                       dropout,
                                       pooling_resolution,
                                       activation,
                                       cls_tower_num,
                                       reg_tower_num,
                                       **kwargs))
        self.heads = nn.ModuleList(self.heads)
        self.return_intermediate = return_intermediate

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            if p.shape[-1] == num_cls:
                nn.init.constant_(p, -math.log((1 - 0.01) / 0.01))

    def forward(self, x, init_boxes, init_params, shapes):
        """
        :param x: dict("p2":(bs,in_channel,h,w),...)
        :param init_boxes:[num_proposal,4]
        :param init_params:[num_proposal,in_channel]
        :param shapes: [(640,640),...]
        :return:
        """
        inter_class_logits = list()
        inter_pred_bboxes = list()
        bs = x['p2'].shape[0]
        bboxes = init_boxes[None, :, :].repeat(bs, 1, 1)
        proposal_features = init_params[None, :, :].repeat(bs, 1, 1)
        class_logits = None
        pred_bboxes = None
        for rcnn_head in self.heads:
            roi_features = self.pooling_layer(x, [box for box in bboxes], shapes)
            class_logits, pred_bboxes, proposal_features = rcnn_head(roi_features, proposal_features,
                                                                     bboxes.view(-1, 4))
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()
        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)
        return class_logits[None], pred_bboxes[None]


default_cfg = {
    "in_channel": 256,
    "inner_channel": 64,
    "num_cls": 80,
    "dim_feedforward": 2048,
    "nhead": 8,
    "dropout": 0,
    "pooling_resolution": 7,
    "activation": nn.ReLU,
    "cls_tower_num": 1,
    "reg_tower_num": 3,
    "num_heads": 6,
    "return_intermediate": True,
    "num_proposals": 100,
    "backbone": "resnet18",
    "pretrained": True,
    "norm_layer": FrozenBatchNorm2d,
    # loss cfg
    "iou_type": "giou",
    "iou_weights": 2.0,
    "iou_cost": 1.0,
    "cls_weights": 2.0,
    "cls_cost": 1.0,
    "l1_weights": 5.0,
    "l1_cost": 1.0
}


class SparseRCNN(nn.Module):
    def __init__(self, **cfg):
        super(SparseRCNN, self).__init__()
        self.cfg = {**default_cfg, **cfg}
        self.backbones = getattr(resnet, self.cfg['backbone'])(pretrained=self.cfg['pretrained'],
                                                               norm_layer=self.cfg['norm_layer'])
        c2, c3, c4, c5 = self.backbones.inner_channels
        self.fpn = FPN(c2, c3, c4, c5, self.cfg['in_channel'])

        self.init_proposal_features = nn.Embedding(self.cfg['num_proposals'], self.cfg['in_channel'])
        self.init_proposal_boxes = nn.Embedding(self.cfg['num_proposals'], 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        self.head = DynamicHead(
            in_channel=self.cfg['in_channel'],
            inner_channel=self.cfg['inner_channel'],
            num_cls=self.cfg['num_cls'],
            dim_feedforward=self.cfg['dim_feedforward'],
            nhead=self.cfg['nhead'],
            dropout=self.cfg['dropout'],
            pooling_resolution=self.cfg['pooling_resolution'],
            activation=self.cfg['activation'],
            cls_tower_num=self.cfg['cls_tower_num'],
            reg_tower_num=self.cfg['reg_tower_num'],
            num_heads=self.cfg['num_heads'],
            return_intermediate=self.cfg['return_intermediate'],
            inplace=True
        )
        self.shape_weights = None
        self.loss = SparseRCNNLoss(iou_type=self.cfg['iou_type'],
                                   iou_weights=self.cfg['iou_weights'],
                                   iou_cost=self.cfg['iou_cost'],
                                   cls_weights=self.cfg['cls_weights'],
                                   cls_cost=self.cfg['cls_cost'],
                                   l1_weights=self.cfg['l1_weights'],
                                   l1_cost=self.cfg['l1_cost'])

    def forward(self, x, targets=None):
        c2, c3, c4, c5 = self.backbones(x)
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)
        h, w = x.shape[2:]
        shapes = [(h, w)] * x.size(0)
        if self.shape_weights is None:
            self.shape_weights = torch.tensor([w, h, w, h], device=x.device)[None, :]

        init_boxes = self.init_proposal_boxes.weight * self.shape_weights
        init_boxes_x1y1 = init_boxes[:, :2] - init_boxes[:, 2:] / 2.0
        init_boxes_x2y2 = init_boxes_x1y1 + init_boxes[:, 2:]
        init_boxes_xyxy = torch.cat([init_boxes_x1y1, init_boxes_x2y2], dim=-1)
        cls_predicts, box_predicts = self.head({"p2": p2, "p3": p3, "p4": p4, "p5": p5},
                                               init_boxes_xyxy,
                                               self.init_proposal_features.weight,
                                               shapes)
        ret = dict()
        if self.training:
            assert targets is not None
            cls_losses, iou_losses, l1_losses, pos_num = self.loss(cls_predicts, box_predicts, targets, shapes[0])
            ret['cls_loss'] = cls_losses
            ret['iou_loss'] = iou_losses
            ret['l1_loss'] = l1_losses
            ret['match_num'] = pos_num
        else:
            cls_predict = cls_predicts[-1]
            box_predict = box_predicts[-1]
            predicts = self.post_process(cls_predict, box_predict, shapes)
            ret['predicts'] = predicts
        return ret

    def post_process(self, cls_predict, box_predict, shapes):
        assert len(cls_predict) == len(box_predict)
        scores = cls_predict.sigmoid()
        result = list()
        labels = torch.arange(self.cfg['num_cls'], device=cls_predict.device). \
            unsqueeze(0).repeat(self.cfg['num_proposals'], 1).flatten(0, 1)
        for score, box, shape in zip(scores, box_predict, shapes):
            scores_per_image, topk_indices = score.flatten(0, 1).topk(self.cfg['num_proposals'], sorted=False)
            labels_per_image = labels[topk_indices]
            box = box.view(-1, 1, 4).repeat(1, self.cfg['num_cls'], 1).view(-1, 4)
            x1y1x2y2 = box[topk_indices]
            x1y1x2y2[:, [0, 2]] = x1y1x2y2[:, [0, 2]].clamp(min=0, max=shape[1])
            x1y1x2y2[:, [1, 3]] = x1y1x2y2[:, [1, 3]].clamp(min=0, max=shape[0])
            result.append(torch.cat([x1y1x2y2, scores_per_image[:, None], labels_per_image[:, None]], dim=-1))
        return result


def roi_test():
    m = MultiScaleRoIAlign(['feat1', 'feat3'], 7, 2)
    i = dict()
    i['feat1'] = torch.rand(2, 5, 64, 64)
    i['feat2'] = torch.rand(2, 5, 32, 32)
    i['feat3'] = torch.rand(2, 5, 16, 16)
    boxes1 = torch.rand(6, 4) * 256
    boxes1[..., 2:] += boxes1[..., :2]
    boxes2 = boxes1[:2, ...]
    input_boxes = [boxes1, boxes2]
    image_sizes = [(512, 512), (512, 512)]
    output = m(i, input_boxes, image_sizes)
    print(output.shape)


if __name__ == '__main__':
    input_x = torch.rand((4, 3, 640, 640))
    net = SparseRCNN()
    net(input_x)
    # net(input_x)
