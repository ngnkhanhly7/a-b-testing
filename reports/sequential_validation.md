# Sequential Testing Validation

Scenario: no true difference (p_control = p_treatment = 0.1), 10 looks of 200 users/group each, 3000 simulated experiments.

| Method | False positive rate | Nominal alpha |
|---|---|---|
| Naive peeking (fixed alpha every look) | 0.1747 | 0.05 |
| O'Brien-Fleming (alpha-spending) | 0.0650 | 0.05 |

Naive peeking inflates the false-positive rate well above the nominal alpha because every daily check is an independent chance to cross the p<0.05 threshold by chance alone. The O'Brien-Fleming boundary spends the alpha budget across looks (very strict early, relaxing to the standard critical value only at the final look), bringing the false positive rate back down close to the nominal alpha.
