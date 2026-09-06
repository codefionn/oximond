Load 22:42:45 up 1 day,  1:12,  1 user,  load average: 10,46, 14,37, 12,11; cpu 5 (sibling 13); oximond sha256 830d2ac50f98d7fb71e0cdf00eda8f7764dcb3ac319e97ec8317e5d0ee6f10fe

| Benchmark | Mode | instructions:u | vs interpreter | tables | elided | elided bytes | traces | entries | iterations | exits | side exits | aborts |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- | :-- |
| allocation/table_churn | interpreter | 68,720,681 | 1.000 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | - |
| allocation/table_churn | tiered | 6,340,920 | 0.092 | 23 | 79,999 | 17,279,784 | 1 | 1 | 79,934 | LoopComplete=1 | - | - |
| allocation/table_churn | before-interpreter | 68,470,652 | 0.996 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/table_churn | before-tiered | 68,191,462 | 0.992 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_array | interpreter | 380,127,106 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_array | tiered | 7,555,148 | 0.020 | 87 | 149,935 | 32,385,960 | 1 | 1 | 149,934 | LoopComplete=1 | - | - |
| allocation/ctor_array | before-interpreter | 380,019,199 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_array | before-tiered | 379,681,306 | 0.999 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_fields | interpreter | 313,042,126 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_fields | tiered | 8,217,849 | 0.026 | 87 | 149,935 | 32,385,960 | 1 | 1 | 149,934 | LoopComplete=1 | - | - |
| allocation/ctor_fields | before-interpreter | 312,889,925 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_fields | before-tiered | 312,560,921 | 0.998 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_mixed | interpreter | 356,705,638 | 1.000 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_mixed | tiered | 7,962,281 | 0.022 | 87 | 119,935 | 25,905,960 | 1 | 1 | 119,934 | LoopComplete=1 | - | - |
| allocation/ctor_mixed | before-interpreter | 356,931,703 | 1.001 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_mixed | before-tiered | 355,809,790 | 0.997 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_side_exit | interpreter | 346,959,792 | 1.000 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_side_exit | tiered | 37,481,505 | 0.108 | 2,487 | 117,536 | 25,387,776 | 1 | 2,399 | 117,536 | GuardFail=2399 | Branch=2399 | UnsupportedOperand=3 |
| allocation/ctor_side_exit | before-interpreter | 346,802,939 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_side_exit | before-tiered | 346,593,520 | 0.999 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |

| Benchmark | Mode | before | after | change |
| :-- | :-- | --: | --: | --: |
| allocation/table_churn | interpreter | 68,470,652 | 68,720,681 | +0.37% |
| allocation/table_churn | tiered | 68,191,462 | 6,340,920 | -90.70% |
| allocation/ctor_array | interpreter | 380,019,199 | 380,127,106 | +0.03% |
| allocation/ctor_array | tiered | 379,681,306 | 7,555,148 | -98.01% |
| allocation/ctor_fields | interpreter | 312,889,925 | 313,042,126 | +0.05% |
| allocation/ctor_fields | tiered | 312,560,921 | 8,217,849 | -97.37% |
| allocation/ctor_mixed | interpreter | 356,931,703 | 356,705,638 | -0.06% |
| allocation/ctor_mixed | tiered | 355,809,790 | 7,962,281 | -97.76% |
| allocation/ctor_side_exit | interpreter | 346,802,939 | 346,959,792 | +0.05% |
| allocation/ctor_side_exit | tiered | 346,593,520 | 37,481,505 | -89.19% |
