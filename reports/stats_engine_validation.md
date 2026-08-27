# Stats Engine Validation

## scenario1_clear_diff
- True rates: control=0.1, treatment=0.15
- Observed rates: control=0.1010, treatment=0.1472
- p-value: 4.378e-23, significant: True
- Conclusion: Treatment cao hơn Control 45.69% (chênh lệch tuyệt đối +0.0462), tin cậy 95%, CI: [+0.0370, +0.0553].

## scenario2_no_diff
- True rates: control=0.1, treatment=0.1
- Observed rates: control=0.1010, treatment=0.1003
- p-value: 0.8604, significant: False
- Conclusion: Không đủ bằng chứng để kết luận có khác biệt giữa 2 nhóm (p=0.8604 >= alpha=0.05).

## scenario3_small_diff
- True rates: control=0.1, treatment=0.103
- Observed rates: control=0.1010, treatment=0.1020
- p-value: 0.8255, significant: False
- Conclusion: Không đủ bằng chứng để kết luận có khác biệt giữa 2 nhóm (p=0.8255 >= alpha=0.05).
