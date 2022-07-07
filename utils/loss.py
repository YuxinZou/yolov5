# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(
        eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(
            reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# class ComputeLoss:
#     # Compute losses
#     def __init__(self, model, autobalance=False):
#         device = next(model.parameters()).device  # get model device
#         h = model.hyp  # hyperparameters
#
#         # Define criteria
#         BCEcls = nn.BCEWithLogitsLoss(
#             pos_weight=torch.tensor([h['cls_pw']], device=device))
#         BCEobj = nn.BCEWithLogitsLoss(
#             pos_weight=torch.tensor([h['obj_pw']], device=device))
#
#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(
#             eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
#         self.sort_obj_iou = h.get('sorted_iou', False)
#
#         # Focal loss
#         g = h['fl_gamma']  # focal loss gamma
#         alpha = h.get('fl_alpha', 0.25)
#         use_qfl = h.get('qfl', False)
#         self.iou_type = h.get('iou_type', 'CIoU').lower()
#         if g > 0:
#             if use_qfl:
#                 BCEcls, BCEobj = QFocalLoss(BCEcls, gamma=g, alpha=alpha), QFocalLoss(BCEobj, gamma=g, alpha=alpha)
#             else:
#                 BCEcls, BCEobj = FocalLoss(BCEcls, gamma=g, alpha=alpha), FocalLoss(BCEobj, gamma=g, alpha=alpha)
#         m = de_parallel(model).model[-1]  # Detect() module
#         # samll object with high loss weight
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06,
#                                                        0.02])  # P3-P7
#         # 自动
#         self.ssi = list(m.stride).index(
#             16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
#         self.na = m.na  # number of anchors
#         self.nc = m.nc  # number of classes
#         self.nl = m.nl  # number of layers
#         self.anchors = m.anchors
#         self.device = device
#
#     def __call__(self, p, targets):  # predictions, targets
#         lcls = torch.zeros(1, device=self.device)  # class loss
#         lbox = torch.zeros(1, device=self.device)  # box loss
#         lobj = torch.zeros(1, device=self.device)  # object loss
#         tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
#         # 每一个都是append的 有feature map个 每个都是当前这个feature map中3个anchor筛选出的所有的target(3个grid_cell进行预测)
#         # tcls: 表示这个target所属的class index, shape为(3, 808)
#         # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量, (3, ([808, 4]))
#         # indices: b: 表示这个target属于的image index
#         #          a: 表示这个target使用的anchor index
#         #          gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
#         #          gi: 表示这个网格的左上角x坐标 (3, ([808], [808], [808], [808]))
#         # anchors: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算,(3, ([808, 2]))
#
#         # Losses
#         for i, pi in enumerate(p):  # layer index, layer predictions
#             # pi shape [4, 3, h, w, 85]
#             b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#
#             # shape Batch, anchor_num(3), feat_h, feat_w, 存的是iou的值
#             tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype,
#                                device=self.device)  # target obj
#
#             n = b.shape[0]  # number of targets
#             if n:
#                 # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
#                 # (808, 2), (808, 2), (808, 80)
#                 pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc),
#                                                            1)  # target-subset of predictions
#
#                 # Regression  只计算所有正样本的回归损失
#                 pxy = pxy.sigmoid() * 2 - 0.5
#                 # 表示把预测框的宽和高限制在4倍的anchors内，这个4和默认的超参数anchor_t是相等的
#                 pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1)  # predicted box
#                 # CIOU loss
#                 # 计算808个预测框与gt box的giou了， 得到的iou的shape是(808)
#                 if self.iou_type == 'siou':
#                     iou = bbox_iou(pbox, tbox[i], SIoU=True).squeeze()  # iou(prediction, target)
#                 else:
#                     iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
#                 lbox += (1.0 - iou).mean()  # iou loss
#
#                 # Objectness loss stpe1
#                 iou = iou.detach().clamp(0).type(tobj.dtype)
#                 if self.sort_obj_iou:
#
#                     # https://github.com/ultralytics/yolov5/issues/3605
#                     # There maybe several GTs match the same anchor when calculate ComputeLoss in the scene with dense targets
#                     # 排序之后 如果同一个grid出现两个gt 那么我们经过排序之后每个grid中的score_iou都能保证是最大的
#                     # (小的会被覆盖 因为同一个grid坐标肯定相同)那么从时间顺序的话, 最后1个总是和最大的IOU去计算LOSS, 梯度传播
#                     j = iou.argsort()
#                     # 小iou
#                     b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
#                 if self.gr < 1:
#                     iou = (1.0 - self.gr) + self.gr * iou
#                 tobj[b, a, gj, gi] = iou  # iou ratio
#
#                 # Classification 只计算所有正样本的分类损失
#                 if self.nc > 1:  # cls loss (only if multiple classes)
#                     t = torch.full_like(pcls, self.cn,
#                                         device=self.device)  # targets
#                     # 没有label smoth的话就是self.cp=1
#                     t[range(n), tcls[i]] = self.cp
#                     lcls += self.BCEcls(pcls, t)  # BCE
#
#                 # Append targets to text file
#                 # with open('targets.txt', 'a') as file:
#                 #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
#
#             # Objectness loss stpe2 置信度损失是用所有样本(正样本 + 负样本)一起计算损失的
#             obji = self.BCEobj(pi[..., 4], tobj)
#             lobj += obji * self.balance[i]  # obj loss
#             if self.autobalance:
#                 self.balance[i] = self.balance[
#                                       i] * 0.9999 + 0.0001 / obji.detach().item()
#
#         if self.autobalance:
#             self.balance = [x / self.balance[self.ssi] for x in self.balance]
#         lbox *= self.hyp['box']
#         lobj *= self.hyp['obj']
#         lcls *= self.hyp['cls']
#         bs = tobj.shape[0]  # batch size
#
#         return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
#
#     def build_targets(self, p, targets):
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch = [], [], [], []
#         # gain是为了后面将targets=[na,nt,7]中的归一化了的xywh映射到相对feature map尺度上
#         gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#         # 先假设所有的target都由这层的三个anchor进行检测(复制三份)  再进行筛选  并将ai加进去标记当前是哪个anchor的target, shape为(3, 190, 7)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]),
#                             2)  # append anchor indices
#
#         g = 0.5  # bias  中心偏移  用来衡量target中心点离哪个格子更近
#         off = torch.tensor(
#             [
#                 [0, 0],
#                 [1, 0],
#                 [0, 1],
#                 [-1, 0],
#                 [0, -1],  # j,k,l,m
#                 # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#             ],
#             device=self.device).float() * g  # offsets
#
#         # 遍历每个head
#         for i in range(self.nl):
#             anchors = self.anchors[i]
#             gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
#
#             # Match targets to anchors
#             t = targets * gain  # shape(3,n,7)
#             # nt: number target
#             if nt:
#                 # Matches
#                 # 当gt box的w和h与anchor的w和h的比值比设置的超参数anchor_t大时，则此gt box去除
#                 # r的shape为[3, 190, 2]， 2分别表示gt box的w和h与anchor的w和h的比值。
#                 r = t[..., 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1 / r).max(2)[0] < self.hyp[
#                     'anchor_t']  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#
#                 # t: [3, 63(gt的个数), 7] -> [126, 7]  [num_Positive_sample, image_index+class+xywh+anchor_index]
#                 # j的shape为(3, 190), 里面的值均为true或false， 表示每一个gt box是否将要过滤掉
#                 # t shape (271, 7), 表示过滤后还剩271个target
#                 t = t[j]  # filter
#
#                 # Offsets
#                 # t之前的shape为(n_gt_rest, 7)， 这里将t复制5个，然后使用j来过滤，
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = gain[[2, 3]] - gxy  # inverse
#                 j, k = ((gxy % 1 < g) & (gxy > 1)).T
#                 l, m = ((gxi % 1 < g) & (gxi > 1)).T
#                 j = torch.stack((torch.ones_like(j), j, k, l, m))
#                 t = t.repeat((5, 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#             else:
#                 t = targets[0]
#                 offsets = 0
#
#             # Define
#             bc, gxy, gwh, a = t.chunk(4,
#                                       1)  # (image, class), grid xy, grid wh, anchors
#             a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid indices
#
#             # Append
#             indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[
#                 2] - 1)))  # image, anchor, grid indices
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class
#
#         return tcls, tbox, indices, anch

class IouAwareLoss:
    """
    iou aware loss, see https://arxiv.org/abs/1912.05992
    Args:
        loss_weight (float): iou aware loss weight, default is 1.0
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
    """

    def __init__(self,
                 loss_weight=1.0,
                 giou=False,
                 diou=False,
                 ciou=False, ):
        self.loss_weight = loss_weight
        self.giou = giou
        self.diou = diou
        self.ciou = ciou

    def __call__(self, ioup, pbox, gbox):
        iou = bbox_iou(
            pbox, gbox, GIoU=self.giou, DIoU=self.diou, CIoU=self.ciou)
        loss_iou_aware = F.binary_cross_entropy_with_logits(
            ioup, iou, reduction='none')
        loss_iou_aware = loss_iou_aware * self.loss_weight
        return loss_iou_aware


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, iou_aware=False):
        self.iou_aware = iou_aware
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['obj_pw']], device=device))
        if self.iou_aware:
            self.BCEiouaware = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([h['obj_pw']], device=device))
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(
            eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
        self.sort_obj_iou = h.get('sorted_iou', False)

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        alpha = h.get('fl_alpha', 0.25)
        use_qfl = h.get('qfl', False)
        self.iou_type = h.get('iou_type', 'CIoU').lower()
        if g > 0:
            if use_qfl:
                BCEcls, BCEobj = QFocalLoss(BCEcls, gamma=g, alpha=alpha), QFocalLoss(BCEobj, gamma=g, alpha=alpha)
            else:
                BCEcls, BCEobj = FocalLoss(BCEcls, gamma=g, alpha=alpha), FocalLoss(BCEobj, gamma=g, alpha=alpha)
        m = de_parallel(model).model[-1]  # Detect() module
        # samll object with high loss weight
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06,
                                                       0.02])  # P3-P7
        # 自动
        self.ssi = list(m.stride).index(
            16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        if self.iou_aware:
            liouaware = torch.zeros(1, device=self.device)  # iou aware loss

        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        # 每一个都是append的 有feature map个 每个都是当前这个feature map中3个anchor筛选出的所有的target(3个grid_cell进行预测)
        # tcls: 表示这个target所属的class index, shape为(3, 808)
        # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量, (3, ([808, 4]))
        # indices: b: 表示这个target属于的image index
        #          a: 表示这个target使用的anchor index
        #          gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
        #          gi: 表示这个网格的左上角x坐标 (3, ([808], [808], [808], [808]))
        # anchors: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算,(3, ([808, 2]))

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # pi shape [4, 3, h, w, 85]
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx

            # shape Batch, anchor_num(3), feat_h, feat_w, 存的是iou的值
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype,
                               device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # (808, 2), (808, 2), (808, 80)
                if self.iou_aware:
                    pxy, pwh, _, piouaware, pcls = pi[b, a, gj, gi].split((2, 2, 1, 1, self.nc),
                                                                          1)  # target-subset of predictions
                else:
                    pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc),
                                                               1)  # target-subset of predictions

                # Regression  只计算所有正样本的回归损失
                pxy = pxy.sigmoid() * 2 - 0.5
                # 表示把预测框的宽和高限制在4倍的anchors内，这个4和默认的超参数anchor_t是相等的
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # CIOU loss
                # 计算808个预测框与gt box的giou了， 得到的iou的shape是(808)
                if self.iou_type == 'siou':
                    iou = bbox_iou(pbox, tbox[i], SIoU=True).squeeze()  # iou(prediction, target)
                else:
                    iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness loss stpe1
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.iou_aware:
                    target_iou = bbox_iou(pbox, tbox[i]).squeeze()
                    liouaware += self.BCEiouaware(piouaware.squeeze(), target_iou) * self.balance[i]
                if self.sort_obj_iou:
                    # https://github.com/ultralytics/yolov5/issues/3605
                    # There maybe several GTs match the same anchor when calculate ComputeLoss in the scene with dense targets
                    # 排序之后 如果同一个grid出现两个gt 那么我们经过排序之后每个grid中的score_iou都能保证是最大的
                    # (小的会被覆盖 因为同一个grid坐标肯定相同)那么从时间顺序的话, 最后1个总是和最大的IOU去计算LOSS, 梯度传播
                    j = iou.argsort()
                    # 小iou
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                if self.iou_aware:
                   iou = 1.0
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification 只计算所有正样本的分类损失
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn,
                                        device=self.device)  # targets
                    # 没有label smoth的话就是self.cp=1
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            # Objectness loss stpe2 置信度损失是用所有样本(正样本 + 负样本)一起计算损失的
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[
                                      i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        if self.iou_aware:
            liouaware *= 0  # self.hyp['cls']
            lobj += liouaware
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # gain是为了后面将targets=[na,nt,7]中的归一化了的xywh映射到相对feature map尺度上
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # 先假设所有的target都由这层的三个anchor进行检测(复制三份)  再进行筛选  并将ai加进去标记当前是哪个anchor的target, shape为(3, 190, 7)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]),
                            2)  # append anchor indices

        g = 0.5  # bias  中心偏移  用来衡量target中心点离哪个格子更近
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        # 遍历每个head
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            # nt: number target
            if nt:
                # Matches
                # 当gt box的w和h与anchor的w和h的比值比设置的超参数anchor_t大时，则此gt box去除
                # r的shape为[3, 190, 2]， 2分别表示gt box的w和h与anchor的w和h的比值。
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))

                # t: [3, 63(gt的个数), 7] -> [126, 7]  [num_Positive_sample, image_index+class+xywh+anchor_index]
                # j的shape为(3, 190), 里面的值均为true或false， 表示每一个gt box是否将要过滤掉
                # t shape (271, 7), 表示过滤后还剩271个target
                t = t[j]  # filter

                # Offsets
                # t之前的shape为(n_gt_rest, 7)， 这里将t复制5个，然后使用j来过滤，
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4,
                                      1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[
                2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
