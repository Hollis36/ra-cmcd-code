# E2 student recall by stratum (validation split, frozen tuples)

| arm | stratum | n seeds | recall | mAP@0.5:0.95 |
|---|---|---|---|---|
| RA-CMCD all-consensus | small | 4 | 0.8808 ± 0.0108 | 0.5455 |
| RA-CMCD all-consensus | medium | 4 | 0.9558 ± 0.0098 | 0.6132 |
| RA-CMCD all-consensus | large | 4 | 0.9807 ± 0.0030 | 0.6754 |
| Plain consensus | small | 4 | 0.8901 ± 0.0151 | 0.5380 |
| Plain consensus | medium | 4 | 0.9513 ± 0.0124 | 0.6166 |
| Plain consensus | large | 4 | 0.9836 ± 0.0030 | 0.6643 |
| Single-best teacher | small | 4 | 0.8727 ± 0.0213 | 0.5313 |
| Single-best teacher | medium | 4 | 0.9414 ± 0.0047 | 0.6074 |
| Single-best teacher | large | 4 | 0.9806 ± 0.0029 | 0.6704 |
| Soft-Teacher-style | small | 8 | 0.8763 ± 0.0175 | 0.5405 |
| Soft-Teacher-style | medium | 8 | 0.9506 ± 0.0113 | 0.6173 |
| Soft-Teacher-style | large | 8 | 0.9829 ± 0.0021 | 0.6727 |
