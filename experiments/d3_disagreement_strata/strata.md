# Where the reliability score acts: disagreement strata

Matched = IoU >= 0.5 with a human training box, unmatched = IoU < 0.2; ambiguous boxes excluded.
AUROC separates matched from unmatched. `eq3_struct` is the bracketed agreement factor with confidence removed,
so it isolates what the structural terms contribute on their own.

## anti_uav

| stratum | boxes | matched | unmatched | AUROC conf-only | AUROC Eq.3 | AUROC structure alone | noise weight: uniform -> Eq.3 |
|---|---:|---:|---:|---:|---:|---:|---|
| all | 1500 | 1418 | 82 | 0.9893 | 0.9912 | 0.8892 | 0.055 -> 0.011 |
| full support (3 sources) | 1461 | 1417 | 44 | 0.9878 | 0.9871 | 0.7940 | 0.030 -> 0.007 |
| partial support (2 sources) | 39 | 1 | 38 | 0.5789 | 0.6316 | 1.0000 | 0.974 -> 0.971 |
| lowest-quartile localization agreement | 374 | 318 | 56 | 0.9801 | 0.9847 | 0.8789 | 0.150 -> 0.035 |

## dronevehicle_U

| stratum | boxes | matched | unmatched | AUROC conf-only | AUROC Eq.3 | AUROC structure alone | noise weight: uniform -> Eq.3 |
|---|---:|---:|---:|---:|---:|---:|---|
| all | 329364 | 199653 | 129711 | 0.8703 | 0.8707 | 0.7537 | 0.394 -> 0.192 |
| full support (3 sources) | 292088 | 193284 | 98804 | 0.8563 | 0.8552 | 0.7035 | 0.338 -> 0.178 |
| partial support (2 sources) | 37276 | 6369 | 30907 | 0.6697 | 0.6696 | 0.5533 | 0.829 -> 0.740 |
| lowest-quartile localization agreement | 77322 | 29587 | 47735 | 0.8606 | 0.8614 | 0.6442 | 0.617 -> 0.347 |
| upper three quartiles | 252042 | 170066 | 81976 | 0.8630 | 0.8634 | 0.7257 | 0.325 -> 0.177 |

