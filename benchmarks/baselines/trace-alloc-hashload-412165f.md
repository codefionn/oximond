Load 19:53:40 up 22:23,  1 user,  load average: 3,62, 3,35, 3,79; cpu 7 (sibling 15); oximond sha256 b36cbee85b2dd0065689ea882f51aa15187d9f8ef70b7eaf7d36c7d26a170cae

| Benchmark | Mode | instructions:u | warmed per scale | vs interpreter | tables | elided | elided bytes | traces | entries | iterations | exits | side exits | aborts |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- | :-- |
| table/hash_read | interpreter | 224,912,358 | 211,605,484 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read | tiered | 223,356,478 | 209,874,862 | 0.993 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3, UnsupportedOperand=3 |
| allocation/table_churn | interpreter | 68,876,298 | 65,092,540 | 1.000 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | - |
| allocation/table_churn | tiered | 68,594,580 | 64,717,230 | 0.996 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| algorithm/nbody | interpreter | 76,702,310 | 71,061,272 | 1.000 | 28 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/nbody | tiered | 109,533,684 | 30,929,359 | 1.428 | 28 | 0 | 0 | 2 | 2,511 | 5,054 | LoopComplete=2511 | - | InnerLoop=6, LoopNotClosed=2 |
| algorithm/spectral_norm | interpreter | 763,107,868 | 758,311,979 | 1.000 | 44 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/spectral_norm | tiered | 104,046,774 | 56,340,551 | 0.136 | 44 | 0 | 0 | 2 | 4,854 | 544,424 | GuardFail=4593, LoopComplete=266 | ArrayBounds=4593 | InnerLoop=2, ShortLoop=4 |
| algorithm/binary_trees | interpreter | 323,428,264 | 327,210,966 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/binary_trees | tiered | 323,268,070 | 327,016,639 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_small | interpreter | 361,170,499 | 357,279,069 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_small | tiered | 358,262,605 | 354,290,393 | 0.992 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| table/hash_read_shaped | interpreter | 366,115,780 | 362,161,287 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_shaped | tiered | 363,109,096 | 359,051,596 | 0.992 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | InnerLoop=1, UnsupportedOperand=3 |
| table/hash_read_indexed | interpreter | 298,335,955 | 270,899,553 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_indexed | tiered | 296,677,455 | 268,990,092 | 0.994 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=6, UnsupportedOperand=3 |
| table/heap_redundancy | interpreter | 408,023,946 | 403,950,278 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/heap_redundancy | tiered | 21,117,483 | 7,345,516 | 0.052 | 24 | 0 | 0 | 1 | 1 | 147,228 | LoopComplete=1 | - | - |
| allocation/ctor_array | interpreter | 380,125,409 | 376,506,407 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_array | tiered | 379,853,295 | 376,149,391 | 0.999 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| allocation/ctor_fields | interpreter | 313,037,996 | 309,205,801 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_fields | tiered | 312,676,930 | 308,758,794 | 0.999 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| allocation/ctor_mixed | interpreter | 356,701,042 | 352,105,119 | 1.000 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_mixed | tiered | 356,112,496 | 351,964,558 | 0.998 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| allocation/ctor_side_exit | interpreter | 346,955,252 | 353,412,213 | 1.000 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_side_exit | tiered | 346,533,398 | 353,168,090 | 0.999 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
