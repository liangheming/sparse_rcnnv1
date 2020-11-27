import torch
from losses.commons import IOULoss, BoxSimilarity, focal_loss
from scipy.optimize import linear_sum_assignment


class BoxCoder(object):
    def __init__(self, weights=None):
        super(BoxCoder, self).__init__()
        if weights is None:
            weights = [0.1, 0.1, 0.2, 0.2]
        self.weights = torch.tensor(data=weights, requires_grad=False)

    def encoder(self, anchors, gt_boxes):
        """
        :param gt_boxes:[box_num, 4]
        :param anchors: [box_num, 4]
        :return:
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[..., [2, 3]] - anchors[..., [0, 1]]
        anchors_xy = anchors[..., [0, 1]] + 0.5 * anchors_wh
        gt_wh = (gt_boxes[..., [2, 3]] - gt_boxes[..., [0, 1]]).clamp(min=1.0)
        gt_xy = gt_boxes[..., [0, 1]] + 0.5 * gt_wh
        delta_xy = (gt_xy - anchors_xy) / anchors_wh
        delta_wh = (gt_wh / anchors_wh).log()

        delta_targets = torch.cat([delta_xy, delta_wh], dim=-1) / self.weights

        return delta_targets

    def decoder(self, predicts, anchors):
        """
        :param predicts: [anchor_num, 4] or [bs, anchor_num, 4]
        :param anchors: [anchor_num, 4]
        :return: [anchor_num, 4] (x1,y1,x2,y2)
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[:, [2, 3]] - anchors[:, [0, 1]]
        anchors_xy = anchors[:, [0, 1]] + 0.5 * anchors_wh
        scale_reg = predicts * self.weights
        scale_wh = scale_reg[..., 2:].exp() * anchors_wh
        scale_x1y1 = (anchors_xy + scale_reg[..., :2] * anchors_wh) - 0.5 * scale_wh
        scale_x2y2 = scale_x1y1 + scale_wh
        return torch.cat([scale_x1y1, scale_x2y2], dim=-1)
        # scale_reg[..., :2] = anchors_xy + scale_reg[..., :2] * anchors_wh
        # scale_reg[..., 2:] = scale_reg[..., 2:].exp() * anchors_wh
        # scale_reg[..., :2] -= (0.5 * scale_reg[..., 2:])
        # scale_reg[..., 2:] = scale_reg[..., :2] + scale_reg[..., 2:]
        #
        # return scale_reg


class HungarianMatcher(object):
    def __init__(self, alpha=0.25, gamma=2.0, cls_cost=1, iou_cost=1, l1_cost=1):
        super(HungarianMatcher, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cls_cost = cls_cost
        self.iou_cost = iou_cost
        self.l1_cost = l1_cost
        self.similarity = BoxSimilarity(iou_type="giou")

    @torch.no_grad()
    def __call__(self, predicts_cls, predicts_box, gt_boxes, shape_norm):
        """
        :param predicts_box: [bs,proposal,80]
        :param predicts_cls: [bs,proposal,4]
        :param gt_boxes: [(num,5)](label_idx,x1,y1,x2,y2)
        :return:
        """
        bs, num_queries = predicts_cls.shape[:2]
        predicts_cls = predicts_cls.view(-1, predicts_cls.size(-1)).sigmoid()
        predicts_box = predicts_box.view(-1, predicts_box.size(-1))
        combine_gts = torch.cat(gt_boxes)
        gt_num = [len(i) for i in gt_boxes]
        positive_loss = -self.alpha * ((1 - predicts_cls) ** self.gamma) * predicts_cls.log()
        negative_loss = - (1 - self.alpha) * (predicts_cls ** self.gamma) * ((1 - predicts_cls).log())
        cls_cost = positive_loss[:, combine_gts[:, 0].long()] - negative_loss[:, combine_gts[:, 0].long()]

        pred_norm = predicts_box / shape_norm[None, :]
        target_norm = combine_gts[:, 1:] / shape_norm[None, :]
        l1_cost = torch.cdist(pred_norm, target_norm, p=1)

        pred_expand = predicts_box[:, None, :].repeat(1, len(combine_gts), 1).view(-1, 4)
        target_expand = combine_gts[:, 1:][None, :, :].repeat(len(predicts_box), 1, 1).view(-1, 4)
        iou_cost = -self.similarity(pred_expand, target_expand).view(len(predicts_cls), -1)
        cost = self.iou_cost * iou_cost + self.cls_cost * cls_cost + self.l1_cost * l1_cost
        cost = cost.view(bs, num_queries, -1).cpu()
        ret = list()
        for i, item in enumerate(cost.split(gt_num, -1)):
            if item.shape[-1] == 0:
                continue
            indices = linear_sum_assignment(item[i])
            ret.append((i, indices[0].tolist(), indices[1].tolist()))
        return ret


class SparseRCNNLoss(object):
    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 iou_weights=2.0,
                 cls_weights=2.0,
                 l1_weights=5.0,
                 iou_type="giou",
                 iou_cost=1.0,
                 cls_cost=1.0,
                 l1_cost=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weights = iou_weights
        self.cls_weights = cls_weights
        self.l1_weights = l1_weights
        self.iou_loss = IOULoss(iou_type=iou_type)
        self.matcher = HungarianMatcher(
            iou_cost=iou_cost,
            cls_cost=cls_cost,
            l1_cost=l1_cost
        )

    def __call__(self, cls_predicts, reg_predicts, targets, shape):
        h, w = shape
        shape_norm = torch.tensor([w, h, w, h], device=cls_predicts.device)
        gt_boxes = targets['target'].split(targets['batch_len'])
        cls_losses = 0.
        iou_losses = 0.
        l1_losses = 0.
        pos_num = 0
        if cls_predicts.dtype == torch.float16:
            cls_predicts = cls_predicts.float()
        for batch_cls_predict, batch_reg_predict in zip(cls_predicts, reg_predicts):
            matches = self.matcher(batch_cls_predict.detach(), batch_reg_predict.detach(), gt_boxes, shape_norm)
            match_cls_bidx = sum([[i] * len(j) for i, j, _ in matches], [])
            match_proposal_idx = sum([j for _, j, _ in matches], [])
            match_cls_label_idx = torch.cat([gt_boxes[i][:, 0][k].long() for i, _, k in matches])
            match_box = torch.cat([gt_boxes[i][:, 1:][k] for i, _, k in matches])
            pos_num = len(match_cls_bidx)

            cls_targets = torch.zeros_like(batch_cls_predict)
            cls_targets[match_cls_bidx, match_proposal_idx, match_cls_label_idx] = 1.0
            cls_loss = self.cls_weights * focal_loss(batch_cls_predict.sigmoid(), cls_targets).sum() / pos_num

            box_pred = batch_reg_predict[match_cls_bidx, match_proposal_idx]
            box_loss = self.iou_weights * self.iou_loss(box_pred, match_box).sum() / pos_num

            l1_loss = self.l1_weights * torch.nn.functional.l1_loss(box_pred / shape_norm[None, :],
                                                                    match_box / shape_norm[None, :],
                                                                    reduction="none").sum() / pos_num
            cls_losses += cls_loss
            iou_losses += box_loss
            l1_losses += l1_loss
        return cls_losses, iou_losses, l1_losses, pos_num
