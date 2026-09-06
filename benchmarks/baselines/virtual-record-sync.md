Load 22:30:30 up 1 day,  1:00,  1 user,  load average: 5,17, 6,70, 11,68; cpu 7 (sibling 15); oximond sha256 34e6c446d3ebf8c3de23ba0c439c77da373a07317f6adf37ec8410c6d3dec225

| Benchmark | Mode | instructions:u | vs interpreter | tables | elided | elided bytes | traces | entries | iterations | exits | side exits | aborts |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- | :-- |
| table/array_read | interpreter | 327,965,614 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/array_read | tiered | 60,665,692 | 0.185 | 23 | 0 | 0 | 1 | 21 | 1,599,263 | LoopComplete=17, Poll=4 | - | ShortLoop=3 |
| table/array_read | before-interpreter | 327,968,285 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/array_read | before-tiered | 60,653,349 | 0.185 | 23 | 0 | 0 | 1 | 21 | 1,599,263 | LoopComplete=17, Poll=4 | - | ShortLoop=3 |
| table/array_write | interpreter | 560,619,623 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/array_write | tiered | 54,816,024 | 0.098 | 23 | 0 | 0 | 1 | 4 | 799,934 | LoopComplete=1, Poll=3 | - | ShortLoop=3 |
| table/array_write | before-interpreter | 560,621,736 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/array_write | before-tiered | 54,807,355 | 0.098 | 23 | 0 | 0 | 1 | 4 | 799,934 | LoopComplete=1, Poll=3 | - | ShortLoop=3 |
| table/hash_read | interpreter | 224,916,254 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read | tiered | 141,807,186 | 0.630 | 24 | 0 | 0 | 1 | 19 | 599,705 | LoopComplete=17, Poll=2 | - | UnsupportedOpcode=3 |
| table/hash_read | before-interpreter | 224,930,814 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read | before-tiered | 141,803,033 | 0.630 | 24 | 0 | 0 | 1 | 19 | 599,705 | LoopComplete=17, Poll=2 | - | UnsupportedOpcode=3 |
| table/field_read | interpreter | 401,299,329 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/field_read | tiered | 17,290,921 | 0.043 | 23 | 0 | 0 | 1 | 2 | 499,934 | LoopComplete=1, Poll=1 | - | - |
| table/field_read | before-interpreter | 401,302,427 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/field_read | before-tiered | 17,289,571 | 0.043 | 23 | 0 | 0 | 1 | 2 | 499,934 | LoopComplete=1, Poll=1 | - | - |
| table/metamethod_index | interpreter | 304,589,454 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/metamethod_index | tiered | 9,441,732 | 0.031 | 25 | 0 | 0 | 1 | 1 | 199,934 | LoopComplete=1 | - | - |
| table/metamethod_index | before-interpreter | 304,591,889 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/metamethod_index | before-tiered | 9,439,412 | 0.031 | 25 | 0 | 0 | 1 | 1 | 199,934 | LoopComplete=1 | - | - |
| table/iteration | interpreter | 508,404,506 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/iteration | tiered | 32,071,553 | 0.063 | 23 | 0 | 0 | 1 | 80 | 797,395 | GuardFail=80 | ArrayBounds=80 | ShortLoop=3, UnsupportedOperand=1 |
| table/iteration | before-interpreter | 508,406,343 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/iteration | before-tiered | 32,060,996 | 0.063 | 23 | 0 | 0 | 1 | 80 | 797,395 | GuardFail=80 | ArrayBounds=80 | ShortLoop=3, UnsupportedOperand=1 |
| allocation/table_churn | interpreter | 68,560,208 | 1.000 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | - |
| allocation/table_churn | tiered | 68,285,822 | 0.996 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | Divergence=3 |
| allocation/table_churn | before-interpreter | 68,561,477 | 1.000 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | - |
| allocation/table_churn | before-tiered | 68,283,747 | 0.996 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| allocation/gc_cycles | interpreter | 144,925,483 | 1.000 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/gc_cycles | tiered | 145,015,525 | 1.001 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | VirtualEscape=3 |
| allocation/gc_cycles | before-interpreter | 144,950,506 | 1.000 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/gc_cycles | before-tiered | 144,934,385 | 1.000 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| algorithm/sieve | interpreter | 276,556,490 | 1.000 | 27 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/sieve | tiered | 126,597,509 | 0.458 | 27 | 0 | 0 | 2 | 469 | 715,775 | GuardFail=224, LoopComplete=245 | ArrayBounds=36, Branch=188 | - |
| algorithm/sieve | before-interpreter | 276,558,184 | 1.000 | 27 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/sieve | before-tiered | 126,548,023 | 0.458 | 27 | 0 | 0 | 2 | 469 | 715,775 | GuardFail=224, LoopComplete=245 | ArrayBounds=36, Branch=188 | - |
| algorithm/nbody | interpreter | 76,705,556 | 1.000 | 28 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/nbody | tiered | 74,288,654 | 0.968 | 28 | 0 | 0 | 2 | 1,977 | 12,730 | LoopComplete=1977 | - | InnerLoop=6, LoopNotClosed=2 |
| algorithm/nbody | before-interpreter | 76,706,734 | 1.000 | 28 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/nbody | before-tiered | 74,223,488 | 0.968 | 28 | 0 | 0 | 2 | 1,977 | 12,730 | LoopComplete=1977 | - | InnerLoop=6, LoopNotClosed=2 |
| algorithm/mandelbrot | interpreter | 798,497,815 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/mandelbrot | tiered | 61,661,448 | 0.077 | 22 | 0 | 0 | 1 | 64 | 822,874 | GuardFail=16, LoopComplete=46, Poll=2 | Branch=16 | - |
| algorithm/mandelbrot | before-interpreter | 798,499,892 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/mandelbrot | before-tiered | 61,634,337 | 0.077 | 22 | 0 | 0 | 1 | 64 | 822,874 | GuardFail=16, LoopComplete=46, Poll=2 | Branch=16 | - |
| algorithm/spectral_norm | interpreter | 763,112,160 | 1.000 | 44 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/spectral_norm | tiered | 74,842,303 | 0.098 | 44 | 0 | 0 | 2 | 4,868 | 571,000 | GuardFail=4796, LoopComplete=72 | ArrayBounds=4796 | ShortLoop=4 |
| algorithm/spectral_norm | before-interpreter | 763,112,643 | 1.000 | 44 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/spectral_norm | before-tiered | 74,812,757 | 0.098 | 44 | 0 | 0 | 2 | 4,868 | 571,000 | GuardFail=4796, LoopComplete=72 | ArrayBounds=4796 | ShortLoop=4 |
| algorithm/binary_trees | interpreter | 323,402,356 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/binary_trees | tiered | 323,277,624 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/binary_trees | before-interpreter | 323,413,860 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/binary_trees | before-tiered | 323,323,039 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/captured_call | interpreter | 85,979,777 | 1.000 | 31 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/captured_call | tiered | 26,691,045 | 0.310 | 31 | 0 | 0 | 1 | 1 | 4,934 | LoopComplete=1 | - | - |
| mineclonia/captured_call | before-interpreter | 85,981,234 | 1.000 | 31 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/captured_call | before-tiered | 26,638,295 | 0.310 | 31 | 0 | 0 | 1 | 1 | 4,934 | LoopComplete=1 | - | - |
| mineclonia/module_calls | interpreter | 268,627,238 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/module_calls | tiered | 55,875,812 | 0.208 | 25 | 0 | 0 | 1 | 1 | 9,934 | LoopComplete=1 | - | - |
| mineclonia/module_calls | before-interpreter | 268,629,627 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/module_calls | before-tiered | 55,893,838 | 0.208 | 25 | 0 | 0 | 1 | 1 | 9,934 | LoopComplete=1 | - | - |
| mineclonia/open_results | interpreter | 179,116,190 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/open_results | tiered | 29,426,684 | 0.164 | 23 | 0 | 0 | 2 | 1,492 | 29,877 | CommittedEffect=748, LoopComplete=744 | Branch=748 | - |
| mineclonia/open_results | before-interpreter | 179,118,106 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/open_results | before-tiered | 29,413,516 | 0.164 | 23 | 0 | 0 | 2 | 1,492 | 29,877 | CommittedEffect=748, LoopComplete=744 | Branch=748 | - |
| mineclonia/ull_kernel | interpreter | 146,057,089 | 1.000 | 32 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/ull_kernel | tiered | 60,707,077 | 0.416 | 32 | 0 | 0 | 1 | 1 | 3,934 | LoopComplete=1 | - | - |
| mineclonia/ull_kernel | before-interpreter | 146,059,103 | 1.000 | 32 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/ull_kernel | before-tiered | 60,682,082 | 0.415 | 32 | 0 | 0 | 1 | 1 | 3,934 | LoopComplete=1 | - | - |
| mineclonia/ull_arith | interpreter | 142,297,623 | 1.000 | 30 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/ull_arith | tiered | 14,148,516 | 0.099 | 30 | 0 | 0 | 1 | 1 | 49,934 | LoopComplete=1 | - | - |
| mineclonia/ull_arith | before-interpreter | 142,299,469 | 1.000 | 30 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/ull_arith | before-tiered | 14,144,128 | 0.099 | 30 | 0 | 0 | 1 | 1 | 49,934 | LoopComplete=1 | - | - |
| mineclonia/bit_wide | interpreter | 205,514,591 | 1.000 | 29 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/bit_wide | tiered | 12,488,701 | 0.061 | 29 | 0 | 0 | 1 | 1 | 49,934 | LoopComplete=1 | - | - |
| mineclonia/bit_wide | before-interpreter | 205,516,381 | 1.000 | 29 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/bit_wide | before-tiered | 12,489,022 | 0.061 | 29 | 0 | 0 | 1 | 1 | 49,934 | LoopComplete=1 | - | - |
| table/hash_read_small | interpreter | 361,173,846 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_small | tiered | 223,480,940 | 0.619 | 24 | 0 | 0 | 1 | 20 | 959,917 | LoopComplete=17, Poll=3 | - | - |
| table/hash_read_small | before-interpreter | 361,175,662 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_small | before-tiered | 223,482,463 | 0.619 | 24 | 0 | 0 | 1 | 20 | 959,917 | LoopComplete=17, Poll=3 | - | - |
| table/hash_read_shaped | interpreter | 366,122,751 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_shaped | tiered | 213,220,723 | 0.582 | 24 | 0 | 0 | 1 | 20 | 959,884 | LoopComplete=17, Poll=3 | - | LoopNotClosed=1 |
| table/hash_read_shaped | before-interpreter | 366,123,147 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_shaped | before-tiered | 213,217,333 | 0.582 | 24 | 0 | 0 | 1 | 20 | 959,884 | LoopComplete=17, Poll=3 | - | LoopNotClosed=1 |
| table/hash_read_indexed | interpreter | 298,340,279 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_indexed | tiered | 202,073,549 | 0.677 | 24 | 0 | 0 | 1 | 35 | 653,774 | GuardFail=16, LoopComplete=17, Poll=2 | ArrayBounds=16 | UnsupportedOpcode=6 |
| table/hash_read_indexed | before-interpreter | 298,318,935 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_indexed | before-tiered | 202,055,459 | 0.677 | 24 | 0 | 0 | 1 | 35 | 653,774 | GuardFail=16, LoopComplete=17, Poll=2 | ArrayBounds=16 | UnsupportedOpcode=6 |
| table/heap_redundancy | interpreter | 408,027,003 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/heap_redundancy | tiered | 13,904,448 | 0.034 | 24 | 0 | 0 | 1 | 1 | 149,934 | LoopComplete=1 | - | - |
| table/heap_redundancy | before-interpreter | 408,028,954 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/heap_redundancy | before-tiered | 13,903,364 | 0.034 | 24 | 0 | 0 | 1 | 1 | 149,934 | LoopComplete=1 | - | - |
| allocation/ctor_array | interpreter | 380,128,939 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_array | tiered | 7,555,309 | 0.020 | 87 | 149,935 | 32,385,960 | 1 | 1 | 149,934 | LoopComplete=1 | - | - |
| allocation/ctor_array | before-interpreter | 380,150,569 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_array | before-tiered | 379,867,849 | 0.999 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| allocation/ctor_fields | interpreter | 313,042,095 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_fields | tiered | 8,217,552 | 0.026 | 87 | 149,935 | 32,385,960 | 1 | 1 | 149,934 | LoopComplete=1 | - | - |
| allocation/ctor_fields | before-interpreter | 313,048,282 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_fields | before-tiered | 312,680,984 | 0.999 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| allocation/ctor_mixed | interpreter | 356,705,504 | 1.000 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_mixed | tiered | 7,961,495 | 0.022 | 87 | 119,935 | 25,905,960 | 1 | 1 | 119,934 | LoopComplete=1 | - | - |
| allocation/ctor_mixed | before-interpreter | 356,603,934 | 1.000 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_mixed | before-tiered | 355,943,629 | 0.998 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| allocation/ctor_side_exit | interpreter | 346,959,988 | 1.000 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_side_exit | tiered | 37,480,166 | 0.108 | 2,487 | 117,536 | 25,387,776 | 1 | 2,399 | 117,536 | GuardFail=2399 | Branch=2399 | UnsupportedOperand=3 |
| allocation/ctor_side_exit | before-interpreter | 346,985,343 | 1.000 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_side_exit | before-tiered | 346,621,296 | 0.999 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |

| Benchmark | Mode | before | after | change |
| :-- | :-- | --: | --: | --: |
| table/array_read | interpreter | 327,968,285 | 327,965,614 | -0.00% |
| table/array_read | tiered | 60,653,349 | 60,665,692 | +0.02% |
| table/array_write | interpreter | 560,621,736 | 560,619,623 | -0.00% |
| table/array_write | tiered | 54,807,355 | 54,816,024 | +0.02% |
| table/hash_read | interpreter | 224,930,814 | 224,916,254 | -0.01% |
| table/hash_read | tiered | 141,803,033 | 141,807,186 | +0.00% |
| table/field_read | interpreter | 401,302,427 | 401,299,329 | -0.00% |
| table/field_read | tiered | 17,289,571 | 17,290,921 | +0.01% |
| table/metamethod_index | interpreter | 304,591,889 | 304,589,454 | -0.00% |
| table/metamethod_index | tiered | 9,439,412 | 9,441,732 | +0.02% |
| table/iteration | interpreter | 508,406,343 | 508,404,506 | -0.00% |
| table/iteration | tiered | 32,060,996 | 32,071,553 | +0.03% |
| allocation/table_churn | interpreter | 68,561,477 | 68,560,208 | -0.00% |
| allocation/table_churn | tiered | 68,283,747 | 68,285,822 | +0.00% |
| allocation/gc_cycles | interpreter | 144,950,506 | 144,925,483 | -0.02% |
| allocation/gc_cycles | tiered | 144,934,385 | 145,015,525 | +0.06% |
| algorithm/sieve | interpreter | 276,558,184 | 276,556,490 | -0.00% |
| algorithm/sieve | tiered | 126,548,023 | 126,597,509 | +0.04% |
| algorithm/nbody | interpreter | 76,706,734 | 76,705,556 | -0.00% |
| algorithm/nbody | tiered | 74,223,488 | 74,288,654 | +0.09% |
| algorithm/mandelbrot | interpreter | 798,499,892 | 798,497,815 | -0.00% |
| algorithm/mandelbrot | tiered | 61,634,337 | 61,661,448 | +0.04% |
| algorithm/spectral_norm | interpreter | 763,112,643 | 763,112,160 | -0.00% |
| algorithm/spectral_norm | tiered | 74,812,757 | 74,842,303 | +0.04% |
| algorithm/binary_trees | interpreter | 323,413,860 | 323,402,356 | -0.00% |
| algorithm/binary_trees | tiered | 323,323,039 | 323,277,624 | -0.01% |
| mineclonia/captured_call | interpreter | 85,981,234 | 85,979,777 | -0.00% |
| mineclonia/captured_call | tiered | 26,638,295 | 26,691,045 | +0.20% |
| mineclonia/module_calls | interpreter | 268,629,627 | 268,627,238 | -0.00% |
| mineclonia/module_calls | tiered | 55,893,838 | 55,875,812 | -0.03% |
| mineclonia/open_results | interpreter | 179,118,106 | 179,116,190 | -0.00% |
| mineclonia/open_results | tiered | 29,413,516 | 29,426,684 | +0.04% |
| mineclonia/ull_kernel | interpreter | 146,059,103 | 146,057,089 | -0.00% |
| mineclonia/ull_kernel | tiered | 60,682,082 | 60,707,077 | +0.04% |
| mineclonia/ull_arith | interpreter | 142,299,469 | 142,297,623 | -0.00% |
| mineclonia/ull_arith | tiered | 14,144,128 | 14,148,516 | +0.03% |
| mineclonia/bit_wide | interpreter | 205,516,381 | 205,514,591 | -0.00% |
| mineclonia/bit_wide | tiered | 12,489,022 | 12,488,701 | -0.00% |
| table/hash_read_small | interpreter | 361,175,662 | 361,173,846 | -0.00% |
| table/hash_read_small | tiered | 223,482,463 | 223,480,940 | -0.00% |
| table/hash_read_shaped | interpreter | 366,123,147 | 366,122,751 | -0.00% |
| table/hash_read_shaped | tiered | 213,217,333 | 213,220,723 | +0.00% |
| table/hash_read_indexed | interpreter | 298,318,935 | 298,340,279 | +0.01% |
| table/hash_read_indexed | tiered | 202,055,459 | 202,073,549 | +0.01% |
| table/heap_redundancy | interpreter | 408,028,954 | 408,027,003 | -0.00% |
| table/heap_redundancy | tiered | 13,903,364 | 13,904,448 | +0.01% |
| allocation/ctor_array | interpreter | 380,150,569 | 380,128,939 | -0.01% |
| allocation/ctor_array | tiered | 379,867,849 | 7,555,309 | -98.01% |
| allocation/ctor_fields | interpreter | 313,048,282 | 313,042,095 | -0.00% |
| allocation/ctor_fields | tiered | 312,680,984 | 8,217,552 | -97.37% |
| allocation/ctor_mixed | interpreter | 356,603,934 | 356,705,504 | +0.03% |
| allocation/ctor_mixed | tiered | 355,943,629 | 7,961,495 | -97.76% |
| allocation/ctor_side_exit | interpreter | 346,985,343 | 346,959,988 | -0.01% |
| allocation/ctor_side_exit | tiered | 346,621,296 | 37,480,166 | -89.19% |
