

| Run | CSV File Name | Initial Loss Function | Initial Combination | Initial LR | Initial Extra Strategies |
| :-- | :---------------------------------- | :-------------------- | :------------------ | :--------- | :------------------------------- |
| 1 | `DeeplabV2_ce` | Cross-Entropy (CE) | --- | 0.0001 | --- |
| 2 | `DeepLabV2_ce07_tv03` | CE + Lovász | 0.5 / 0.5 | 0.0001 | --- |
| 3 | `DeepLabV2_ce05_lv05` | CE + Tversky | 0.7 / 0.3 | 0.0001 | --- |
| 4 | `DeepLabV2_cv07_di03` | CE + Dice | 0.7 / 0.3 | 0.0001 | --- |
| 5 | `DeepLabV2_ce07_l03_warnup_lr_0.0002` | CE + Focal + Lovász | 0.6 / 0.2 / 0.2 | $0.0001$ | --- |
| 6 | `DeepLabV2_ce05_f05_warnup_lr_0.0003` | CE + Lovász | 0.7 / 0.3 $\gamma=2.0$| $0.0002$ | Warmup 500 steps |
| 7 | `DeepLabV2_ce05_l0.25_di0.25_no_warnup_lr_0.0002` | CE + Lovász + Dice | 0.5 / 0.25 / 0.25 | $0.0002$ | --- |
| 8 | `DeepLabV2_ce06_l0.2_fo0.2_no_warnup_lr_0.0001` | CE + Focal | 0.6 / 0.2 / 0.2 $\gamma=2.0$ | $0.0003$ | Warmup 500 steps |
| 9 | `DeepLabV2_ce05_f05_warmup1500_lr_0.0005` | CE + Focal | 0.5 / 0.5 | $0.0005$ | Warmup 1500 steps + epoch=41 add weights|
| 10 | `DeepLabV2_ce05_f05_warmup2500_lr_0.0005_ALL_WHEIGHTED` | CE + Focal | 0.5 / 0.5 | $0.0005$ | Warmup 2500 steps + epoch=31 focal_gamma=3 + epoch=43 focal_gamma=4 |

| **CE** (Cross-Entropy)                  | 49.97     | **364.63**     | Fastest training, lowest performance       |
| **CE + Tversky (0.7/0.3)**              | **51.04** | 670.73     | Best mIoU, 84% slower than CE              |
| **CE + Lovász + Tversky (0.6/0.2/0.2)** | 50.96     | **710.77** | Nearly same mIoU as CE+Tversky, but slower |