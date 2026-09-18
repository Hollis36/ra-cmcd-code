# C3 label-level gate on DroneVehicle U (2026-09-11 21:38)

Teachers trained on L only; WBF tuple selected on the official validation split and applied to U unchanged.
Tuple: {'iou': 0.55, 'mode': 'weighted', 'min_models': 2, 'post_conf': 0.03}. U: 14190 images, 343329 fused boxes.

| scoring | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| plain | 0.6060 | 0.4071 |
| conf_only | 0.5884 | 0.4016 |
| eq3 | 0.5921 | 0.4041 |

- Eq. 3 minus plain WBF: -0.29 pp mAP@0.5:0.95 (gate needs >= 0.30 pp)
- Eq. 3 minus confidence-only: +0.25 pp mAP@0.5:0.95 (this is the structural-terms contribution)

**Gate FAILED**
