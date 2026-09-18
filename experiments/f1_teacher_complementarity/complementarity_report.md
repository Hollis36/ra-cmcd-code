# Teacher-error complementarity (F1), computed 2026-09-07 12:26

Best IoU per human box per teacher (detections with confidence >= 0.1); oracle = best teacher chosen per box; consensus = WBF training-config fused boxes where cached.

## val (579 human boxes)

| quantity | YOLO11-RGBT | C2Former | CFT | ICAFusion | oracle | WBF consensus |
|---|---|---|---|---|---|---|
| mean best IoU | 0.755 | 0.764 | 0.760 | 0.750 | 0.808 | 0.779 |
| boxes with IoU >= 0.75 | 68.6% | 69.1% | 71.0% | 68.0% | 82.4% | 73.2% |
| boxes with IoU >= 0.90 | 8.6% | 6.6% | 8.3% | 6.0% | 17.3% | 9.8% |

Best single teacher: C2Former. Boxes where another teacher is tighter: 75.5%. Share of boxes each teacher is tightest on: YOLO11-RGBT 29.4%, C2Former 24.5%, CFT 23.3%, ICAFusion 22.8%.

Pairwise correlation of per-box best IoU: YOLO11-RGBT|C2Former 0.69, YOLO11-RGBT|CFT 0.66, YOLO11-RGBT|ICAFusion 0.72, C2Former|CFT 0.78, C2Former|ICAFusion 0.79, CFT|ICAFusion 0.76.

| size stratum | n | YOLO11-RGBT | C2Former | CFT | ICAFusion | oracle | best teacher changes | WBF consensus |
|---|---|---|---|---|---|---|---|---|
| small (<28 px) | 94 | 0.685 | 0.709 | 0.705 | 0.706 | 0.778 | 71.3% | 0.741 |
| medium (28-39 px) | 139 | 0.704 | 0.721 | 0.723 | 0.703 | 0.773 | 82.7% | 0.737 |
| large (>=39 px) | 346 | 0.795 | 0.797 | 0.790 | 0.781 | 0.831 | 73.7% | 0.805 |

## night160 (158 human boxes)

| quantity | C2Former | CFT | ICAFusion | YOLO11-RGBT | oracle | WBF consensus |
|---|---|---|---|---|---|---|
| mean best IoU | 0.817 | 0.815 | 0.792 | 0.811 | 0.840 | - |
| boxes with IoU >= 0.75 | 85.4% | 86.7% | 76.6% | 83.5% | 89.2% | - |
| boxes with IoU >= 0.90 | 12.7% | 12.0% | 7.6% | 15.2% | 25.3% | - |

Best single teacher: C2Former. Boxes where another teacher is tighter: 71.5%. Share of boxes each teacher is tightest on: C2Former 28.5%, CFT 26.6%, ICAFusion 17.7%, YOLO11-RGBT 27.2%.

Pairwise correlation of per-box best IoU: C2Former|CFT 0.95, C2Former|ICAFusion 0.89, C2Former|YOLO11-RGBT 0.94, CFT|ICAFusion 0.89, CFT|YOLO11-RGBT 0.94, ICAFusion|YOLO11-RGBT 0.85.

| size stratum | n | C2Former | CFT | ICAFusion | YOLO11-RGBT | oracle | best teacher changes | WBF consensus |
|---|---|---|---|---|---|---|---|---|
| small (<28 px) | 14 | 0.707 | 0.725 | 0.714 | 0.747 | 0.752 | 85.7% | - |
| medium (28-39 px) | 19 | 0.818 | 0.820 | 0.803 | 0.819 | 0.842 | 89.5% | - |
| large (>=39 px) | 125 | 0.829 | 0.824 | 0.799 | 0.817 | 0.849 | 67.2% | - |
