"""Normalized Wasserstein Distance (NWD) for tiny objects (Wang et al. 2021, arXiv 2110.13389), patched into Ultralytics.

A box is modelled as a 2-D Gaussian N((cx, cy), diag(w^2/4, h^2/4)); then
    W2^2 = (cx1-cx2)^2 + (cy1-cy2)^2 + ((w1-w2)/2)^2 + ((h1-h2)/2)^2,   NWD = exp(-sqrt(W2^2) / C).
Unlike IoU, NWD is smooth and non-zero for non-overlapping boxes, so a 1-2 px shift of a 15 px truck does not
collapse the matching score (the tiny-object argument).

Declared design (10 Oct, before any run):
  - assignment: the TaskAlignedAssigner overlap term uses NWD instead of CIoU (paper: assignment carries most of the gain);
  - box loss:   0.5 * (1 - CIoU) + 0.5 * (1 - NWD) (paper's combined loss);
  - C = 44 px = median sqrt(w*h) of the train boxes (22 px) at the 2x training scale (fixed, not tuned).
Importing this module applies the patch. NWDTrainer is passed to model.train(trainer=...) so the class lives in this
module; Ultralytics DDP pickles the trainer class by reference, so each worker re-imports this module and is patched too.
"""
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import loss as uloss
from ultralytics.utils import tal
from ultralytics.utils.metrics import bbox_iou

C = 44.0


def nwd(a, b, c=C, eps=1e-7):
    """NWD between matching xyxy boxes a, b (..., 4) in pixels -> (...)."""
    acx, acy = (a[..., 0] + a[..., 2]) / 2, (a[..., 1] + a[..., 3]) / 2
    bcx, bcy = (b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2
    aw, ah = a[..., 2] - a[..., 0], a[..., 3] - a[..., 1]
    bw, bh = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    w2 = (acx - bcx) ** 2 + (acy - bcy) ** 2 + ((aw - bw) / 2) ** 2 + ((ah - bh) / 2) ** 2
    return torch.exp(-torch.sqrt(w2 + eps) / c)


def _iou_calculation(self, gt_bboxes, pd_bboxes):
    # assigner works in input pixels (pred boxes * stride, gt in pixels)
    return nwd(gt_bboxes, pd_bboxes).clamp_(0)


_orig_forward = uloss.BboxLoss.forward


def _bbox_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                  fg_mask, imgsz, stride):
    loss_iou, loss_dfl = _orig_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                       target_scores_sum, fg_mask, imgsz, stride)
    idx = fg_mask.nonzero(as_tuple=True)
    if idx[0].numel() == 0:
        return loss_iou, loss_dfl
    weight = target_scores[idx].sum(-1)
    s = stride[idx[1]]  # loss boxes are in grid units; NWD's C is in pixels
    n = nwd(pred_bboxes[idx] * s, target_bboxes[idx] * s)
    loss_nwd = ((1.0 - n) * weight).sum() / target_scores_sum
    return 0.5 * loss_iou + 0.5 * loss_nwd, loss_dfl


tal.TaskAlignedAssigner.iou_calculation = _iou_calculation
uloss.BboxLoss.forward = _bbox_forward


class NWDTrainer(DetectionTrainer):
    """DetectionTrainer whose module import patches the assigner and box loss with NWD."""


if __name__ == "__main__":  # sanity: identical boxes -> 1; 2 px shift of a 15 px box: IoU vs NWD
    a = torch.tensor([[0., 0., 15., 15.]])
    b = torch.tensor([[2., 2., 17., 17.]])
    far = torch.tensor([[20., 20., 35., 35.]])
    print("NWD same", nwd(a, a).item(), "| shift2: IoU", bbox_iou(a, b, xywh=False).item(), "NWD", nwd(a, b).item(),
          "| no overlap: IoU", bbox_iou(a, far, xywh=False).item(), "NWD", nwd(a, far).item())
