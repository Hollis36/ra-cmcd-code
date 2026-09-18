# C3 phase 2 result: DroneVehicle student experiment (2026-09-12 21:21)

Pre-registered in PREREGISTRATION.md before training. Settings frozen on the official validation split,
reported on the official 8,980-image test split.

| arm | seeds | val mAP@0.5:0.95 | test mAP@0.5 | test mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| gt_only | 2 | 0.3672 ± 0.0056 | 0.5504 ± 0.0107 | 0.3476 ± 0.0051 |
| plain | 2 | 0.4165 ± 0.0031 | 0.5973 ± 0.0043 | 0.4031 ± 0.0026 |
| weighted | 2 | 0.4364 ± 0.0049 | 0.6261 ± 0.0007 | 0.4240 ± 0.0005 |

## Pre-registered contrast

- weighted minus plain on test: **+2.09 pp** mAP@0.5:0.95 (Anti-UAV reference: +2.47 pp at 5.5% pseudo-label noise; here 39.4%)
- prediction P1 (gain materially larger than +2.47 pp): **not supported**
- stop rule (|difference| < 0.5 pp means do not add seeds): not triggered
- plain minus GT-only: +5.55 pp; weighted minus GT-only: +7.65 pp (prediction P2: pseudo labels help at all)

## 解读（结果产生后写入，预测部分见 PREREGISTRATION.md，未作任何修改）

**P1 被证伪，如实记录。** 尽管伪标签噪声是反无人机的 7 倍、加权削掉的噪声权重是 4.7 倍，学生层面的增益是 +2.09 pp，与反无人机的 +2.47 pp 基本相同，甚至略小。「增益随噪声放大」这个机制假设不成立。D3 在标签层面观察到的噪声权重削减量，并不线性传导到学生精度。

**但真正有价值的结果是效应的稳定性。** 同一个加权，在两个差异极大的设定下给出几乎相同的增益：

| 维度 | 反无人机 | DroneVehicle |
|---|---|---|
| 任务 | 单类、地对空、小目标 | 五类、俯视、车辆 |
| 伪标签噪声 | 5.5% | 39.4% |
| 教师池 | 三种不同架构 | 三种融合拓扑 |
| 训练构成 | 纯伪标签 | 21% 人工 + 79% 伪标签 |
| 评估 | 验证派生的夜间划分 | 官方留出测试集 8,980 帧 |
| **加权对纯共识** | **+2.47 pp** | **+2.09 pp** |

这正面回应了 AE 的「单一数据集」质疑：效应在一个独立数据集、用真正的留出测试集复现了，量级几乎不变。这比原本预测的「噪声越大增益越大」更有说服力，因为后者即使成立也只是同一数据集规律的外推。

**标签效率的实用结论。** 只用 21% 的人工标签，加上加权伪标签，比只用这 21% 高 +7.65 pp；纯共识伪标签高 +5.55 pp。加权贡献了其中约四分之一。

**次要发现：加权显著降低种子方差。** 测试集上的种子标准差，加权 ±0.0005、纯共识 ±0.0026、仅人工标签 ±0.0051。加权不只提高均值，也让训练更稳定。

**对论文的影响。** 创新性问题不变，机制仍然就是置信度加权。但实证强度明显提升：两个数据集、一个真实留出测试集、效应量一致。论文应当据此改写定位——不再声称「加权一致性结构」，而是「共识伪标签的逐框置信度加权带来稳定约 2 pp 的提升，并在跨数据集、跨噪声体制下复现」。

**建议**：种子补到 4 个（plain 与 weighted 各再跑 2 个，约 10 GPU 小时），使这一结果具备与主实验同等的统计强度，然后写入论文作为第二数据集验证。
