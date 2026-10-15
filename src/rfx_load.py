"""Build an RF-DETR matching the xView checkpoint (old LW-DETR-style args) and load it (weights_only)."""
import argparse, dataclasses, torch
from rfdetr.config import RFDETRBaseConfig
from rfdetr.models.lwdetr import build_model_from_config
from rfdetr.models._defaults import MODEL_DEFAULTS
torch.serialization.add_safe_globals([argparse.Namespace])

import os
CKPT = os.environ.get('RFX_CKPT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights', 'rfdetr_xview_ckpt1008.pth'))

def load_ckpt():
    return torch.load(CKPT, map_location='cpu', weights_only=True)

def build(num_windows=4, use_ema=True, device='cpu', num_classes=62):
    c = load_ckpt(); a = vars(c['args'])
    mc = RFDETRBaseConfig(hidden_dim=512, dec_layers=6, sa_nheads=4, ca_nheads=4, dec_n_points=4,
                          projector_scale=['P4', 'P5'], out_feature_indexes=[10, 11], resolution=1008,
                          positional_encoding_size=37, patch_size=14, num_windows=num_windows,
                          num_classes=num_classes, group_detr=1, num_queries=300, num_select=100,
                          pretrain_weights=None, device=device,
                          two_stage=False, bbox_reparam=False, lite_refpoint_refine=False, layer_norm=False)
    # architectural constants from the checkpoint override package defaults where both exist
    dd = {f.name: a[f.name] for f in dataclasses.fields(MODEL_DEFAULTS) if f.name in a and a[f.name] is not None}
    defaults = dataclasses.replace(MODEL_DEFAULTS, **dd)
    model = build_model_from_config(mc, defaults=defaults)
    sd = c['ema_model'] if use_ema and c.get('ema_model') else c['model']
    r = model.load_state_dict(sd, strict=False)
    return model.eval(), r, dd

if __name__ == '__main__':
    m, r, dd = build()
    print('defaults from ckpt:', dd)
    print('missing', len(r.missing_keys), r.missing_keys[:8]); print('unexpected', len(r.unexpected_keys), r.unexpected_keys[:8])


def criterion(device='cpu'):
    """Loss matching the checkpoint's own training setup (focal loss, cls coef 2, aux loss)."""
    from rfdetr.config import TrainConfig
    from rfdetr.models.lwdetr import build_criterion_from_config
    c = load_ckpt(); a = vars(c['args'])
    mc = RFDETRBaseConfig(hidden_dim=512, dec_layers=6, sa_nheads=4, ca_nheads=4, dec_n_points=4,
                          projector_scale=['P4', 'P5'], out_feature_indexes=[10, 11], resolution=1008,
                          positional_encoding_size=37, patch_size=14, num_windows=4, num_classes=62,
                          group_detr=1, num_queries=300, num_select=100, pretrain_weights=None, device=device,
                          two_stage=False, bbox_reparam=False, lite_refpoint_refine=False, layer_norm=False,
                          ia_bce_loss=False)
    tc = TrainConfig(dataset_dir='.', output_dir='.', cls_loss_coef=a['cls_loss_coef'])
    dd = {f.name: a[f.name] for f in dataclasses.fields(MODEL_DEFAULTS) if f.name in a and a[f.name] is not None}
    crit, _ = build_criterion_from_config(mc, tc, dataclasses.replace(MODEL_DEFAULTS, **dd))
    return crit
