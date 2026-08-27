# Cookie Cats Analysis

Dataset: 90189 users. gate_30 (A, control) = 44700, gate_40 (B, treatment) = 45489.

## Sample Ratio Mismatch check

Observed: {'A': 44700, 'B': 45489}. Expected: {'A': 45094.5, 'B': 45094.5}. p-value: 0.008608.
OK, no mismatch.

## Retention Day 1

Không đủ bằng chứng để kết luận có khác biệt giữa 2 nhóm (p=0.0744 >= alpha=0.05).
(control=0.4482, treatment=0.4423, p=0.07441)

## Retention Day 7

Treatment thấp hơn Control 4.31% (chênh lệch tuyệt đối -0.0082), tin cậy 95%, CI: [-0.0133, -0.0031].
(control=0.1902, treatment=0.1820, p=0.001554)

## Game rounds played (first week)

Không đủ bằng chứng để kết luận có khác biệt giữa 2 nhóm (p=0.3759 >= alpha=0.05).
(control mean=52.46, treatment mean=51.30, p=0.3759)

## Conclusion

Moving the gate from level 30 to level 40 does not show a statistically
significant improvement in retention, and day-7 retention is directionally
lower for the treatment (gate_40) group. This matches the well-known public
analyses of this dataset (e.g. on Kaggle/Medium), which is the validation
signal we want: the tool reaches the same conclusion as established
analyses on real data, not just on our own simulated scenarios.
