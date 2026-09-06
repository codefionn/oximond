Load 22:13:22 up 1 day, 43 min,  1 user,  load average: 5,05, 13,14, 17,64; cpu 7 (sibling 15); oximond sha256 51f51778bae0d71c58ccf37a61a0ae6728c8bfaef2a536c7792b6d64e9c4fc69

| Benchmark | Mode | instructions:u | warmed per scale | vs interpreter | tables | elided | elided bytes | traces | entries | iterations | exits | side exits | aborts |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- | :-- |
| table/array_read | interpreter | 327,967,951 | 315,234,023 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/array_read | tiered | 73,713,081 | 44,667,766 | 0.225 | 23 | 0 | 0 | 1 | 21 | 1,528,019 | LoopComplete=17, Poll=4 | - | ShortLoop=3 |
| table/array_read | before-interpreter | 327,958,906 | 315,233,027 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/array_read | before-tiered | 68,405,539 | 41,639,273 | 0.209 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/array_write | interpreter | 560,621,046 | 552,000,061 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/array_write | tiered | 57,985,116 | 36,774,913 | 0.103 | 23 | 0 | 0 | 1 | 4 | 790,859 | LoopComplete=1, Poll=3 | - | ShortLoop=3 |
| table/array_write | before-interpreter | 560,612,340 | 551,999,296 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/array_write | before-tiered | 57,875,466 | 36,866,447 | 0.103 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read | interpreter | 224,930,929 | 211,621,005 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read | tiered | 148,457,155 | 116,685,389 | 0.660 | 24 | 0 | 0 | 1 | 50 | 559,267 | LoopComplete=49, Poll=1 | - | UnsupportedOpcode=3 |
| table/hash_read | before-interpreter | 224,920,138 | 211,621,196 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read | before-tiered | 223,351,662 | 209,874,099 | 0.993 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/field_read | interpreter | 401,302,426 | 397,498,635 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/field_read | tiered | 25,138,980 | 11,535,210 | 0.063 | 23 | 0 | 0 | 1 | 2 | 489,968 | LoopComplete=1, Poll=1 | - | - |
| table/field_read | before-interpreter | 401,292,244 | 397,500,168 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/field_read | before-tiered | 24,662,830 | 11,821,474 | 0.061 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/metamethod_index | interpreter | 304,591,376 | 300,800,751 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/metamethod_index | tiered | 16,945,405 | 3,406,052 | 0.056 | 25 | 0 | 0 | 1 | 1 | 194,984 | LoopComplete=1 | - | - |
| table/metamethod_index | before-interpreter | 304,582,388 | 300,799,962 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/metamethod_index | before-tiered | 16,766,115 | 4,150,619 | 0.055 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/iteration | interpreter | 508,406,651 | 500,206,206 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/iteration | tiered | 41,311,484 | 23,036,308 | 0.081 | 23 | 0 | 0 | 1 | 79 | 782,349 | GuardFail=79 | ArrayBounds=79 | ShortLoop=3, UnsupportedOperand=1 |
| table/iteration | before-interpreter | 508,396,803 | 500,206,169 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/iteration | before-tiered | 36,232,002 | 16,294,065 | 0.071 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/table_churn | interpreter | 68,560,134 | 64,768,415 | 1.000 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | - |
| allocation/table_churn | tiered | 68,288,615 | 64,399,480 | 0.996 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | Divergence=3 |
| allocation/table_churn | before-interpreter | 68,469,969 | 64,688,957 | 0.999 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/table_churn | before-tiered | 68,190,767 | 64,318,178 | 0.995 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/gc_cycles | interpreter | 144,931,811 | 139,896,765 | 1.000 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/gc_cycles | tiered | 144,952,105 | 139,809,920 | 1.000 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | VirtualEscape=3 |
| allocation/gc_cycles | before-interpreter | 144,841,973 | 139,816,493 | 0.999 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/gc_cycles | before-tiered | 144,842,295 | 139,728,175 | 0.999 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/sieve | interpreter | 276,557,656 | 271,898,211 | 1.000 | 27 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/sieve | tiered | 136,157,171 | 113,901,064 | 0.492 | 27 | 0 | 0 | 2 | 3,684 | 703,459 | GuardFail=3439, LoopComplete=245 | ArrayBounds=3251, Branch=188 | - |
| algorithm/sieve | before-interpreter | 276,549,147 | 271,898,894 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/sieve | before-tiered | 136,030,841 | 111,683,151 | 0.492 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/nbody | interpreter | 76,708,404 | 71,062,606 | 1.000 | 28 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/nbody | tiered | 110,386,666 | 24,071,475 | 1.439 | 28 | 0 | 0 | 2 | 2,575 | 5,179 | LoopComplete=2575 | - | InnerLoop=6, LoopNotClosed=2 |
| algorithm/nbody | before-interpreter | 76,697,260 | 71,062,001 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/nbody | before-tiered | 108,315,322 | 37,196,807 | 1.412 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/mandelbrot | interpreter | 798,500,071 | 794,368,066 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/mandelbrot | tiered | 95,751,960 | 19,705,509 | 0.120 | 22 | 0 | 0 | 1 | 5,695 | 805,275 | GuardFail=5026, LoopComplete=685 | Branch=5026 | InnerLoop=5 |
| algorithm/mandelbrot | before-interpreter | 798,491,517 | 794,367,084 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/mandelbrot | before-tiered | 89,289,838 | 27,428,413 | 0.112 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/spectral_norm | interpreter | 763,114,162 | 758,312,794 | 1.000 | 44 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/spectral_norm | tiered | 103,282,458 | 58,776,181 | 0.135 | 44 | 0 | 0 | 2 | 4,876 | 545,660 | GuardFail=4603, LoopComplete=273 | ArrayBounds=4603 | InnerLoop=2, ShortLoop=4 |
| algorithm/spectral_norm | before-interpreter | 763,104,263 | 758,312,706 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/spectral_norm | before-tiered | 99,904,875 | 60,400,370 | 0.131 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/binary_trees | interpreter | 323,382,534 | 327,149,179 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/binary_trees | tiered | 323,266,725 | 327,093,811 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/binary_trees | before-interpreter | 323,291,452 | 327,065,719 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/binary_trees | before-tiered | 323,177,992 | 326,874,228 | 0.999 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/captured_call | interpreter | 85,981,078 | 77,829,501 | 1.000 | 31 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/captured_call | tiered | 47,888,444 | 1,015,194 | 0.557 | 31 | 0 | 0 | 1 | 1 | 3,548 | LoopComplete=1 | - | - |
| mineclonia/captured_call | before-interpreter | 85,970,678 | 77,829,449 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/captured_call | before-tiered | 48,059,334 | 2,014,726 | 0.559 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/module_calls | interpreter | 268,629,320 | 262,558,265 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/module_calls | tiered | 115,483,066 | 8,235,832 | 0.430 | 25 | 0 | 0 | 1 | 1 | 7,591 | LoopComplete=1 | - | - |
| mineclonia/module_calls | before-interpreter | 268,619,107 | 262,558,632 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/module_calls | before-tiered | 121,512,441 | 8,183,940 | 0.452 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/open_results | interpreter | 179,119,119 | 174,486,085 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/open_results | tiered | 63,385,487 | 12,096,040 | 0.354 | 23 | 0 | 0 | 2 | 1,185 | 23,683 | CommittedEffect=592, LoopComplete=593 | Branch=592 | - |
| mineclonia/open_results | before-interpreter | 179,108,197 | 174,486,293 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/open_results | before-tiered | 64,410,703 | 10,752,348 | 0.360 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/ull_kernel | interpreter | 146,058,769 | 138,094,158 | 1.000 | 32 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/ull_kernel | tiered | 150,504,792 | 13,358,411 | 1.030 | 32 | 0 | 0 | 1 | 1 | 1,294 | LoopComplete=1 | - | - |
| mineclonia/ull_kernel | before-interpreter | 146,048,679 | 138,094,176 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/ull_kernel | before-tiered | 151,787,023 | 7,748,879 | 1.039 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/ull_arith | interpreter | 142,300,565 | 137,799,908 | 1.000 | 30 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/ull_arith | tiered | 45,850,688 | 4,314,534 | 0.322 | 30 | 0 | 0 | 1 | 1 | 38,120 | LoopComplete=1 | - | - |
| mineclonia/ull_arith | before-interpreter | 142,291,091 | 137,800,063 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/ull_arith | before-tiered | 45,955,893 | 4,400,240 | 0.323 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/bit_wide | interpreter | 205,516,229 | 200,550,015 | 1.000 | 29 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/bit_wide | tiered | 51,006,183 | -14,892,151 | 0.248 | 29 | 0 | 0 | 1 | 1 | 40,232 | LoopComplete=1 | - | - |
| mineclonia/bit_wide | before-interpreter | 205,505,428 | 200,550,337 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/bit_wide | before-tiered | 50,914,249 | 2,505,374 | 0.248 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read_small | interpreter | 361,175,517 | 357,281,434 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_small | tiered | 239,925,684 | 198,545,744 | 0.664 | 24 | 0 | 0 | 1 | 2,963 | 938,900 | LoopComplete=2960, Poll=3 | - | InnerLoop=3 |
| table/hash_read_small | before-interpreter | 361,166,076 | 357,280,350 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read_small | before-tiered | 358,259,813 | 354,289,137 | 0.992 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read_shaped | interpreter | 366,122,738 | 362,160,612 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_shaped | tiered | 231,623,248 | 186,549,180 | 0.633 | 24 | 0 | 0 | 1 | 3,281 | 939,520 | LoopComplete=3278, Poll=3 | - | InnerLoop=3, LoopNotClosed=1 |
| table/hash_read_shaped | before-interpreter | 366,113,400 | 362,160,415 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read_shaped | before-tiered | 363,104,976 | 359,052,401 | 0.992 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read_indexed | interpreter | 298,319,859 | 270,901,417 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_indexed | tiered | 214,959,525 | 156,610,949 | 0.721 | 24 | 0 | 0 | 1 | 2,950 | 645,778 | GuardFail=2931, LoopComplete=17, Poll=2 | ArrayBounds=2931 | UnsupportedOpcode=6 |
| table/hash_read_indexed | before-interpreter | 298,308,406 | 270,900,015 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read_indexed | before-tiered | 296,671,794 | 268,989,872 | 0.994 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/heap_redundancy | interpreter | 408,030,197 | 403,950,364 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/heap_redundancy | tiered | 20,534,312 | 6,908,485 | 0.050 | 24 | 0 | 0 | 1 | 1 | 147,459 | LoopComplete=1 | - | - |
| table/heap_redundancy | before-interpreter | 408,020,518 | 403,950,186 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/heap_redundancy | before-tiered | 20,674,363 | 6,995,327 | 0.051 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_array | interpreter | 380,178,544 | 376,555,477 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_array | tiered | 12,141,747 | 2,524,072 | 0.032 | 1,869 | 148,153 | 32,001,048 | 1 | 1 | 148,152 | LoopComplete=1 | - | - |
| allocation/ctor_array | before-interpreter | 380,018,690 | 376,405,566 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_array | before-tiered | 379,680,359 | 375,981,546 | 0.999 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_fields | interpreter | 313,049,417 | 309,209,705 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_fields | tiered | 13,403,576 | 2,981,346 | 0.043 | 2,562 | 147,460 | 31,851,360 | 1 | 1 | 147,459 | LoopComplete=1 | - | - |
| allocation/ctor_fields | before-interpreter | 312,888,586 | 309,060,187 | 0.999 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_fields | before-tiered | 312,560,069 | 308,646,652 | 0.998 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_mixed | interpreter | 357,028,725 | 353,217,680 | 1.000 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_mixed | tiered | 12,507,721 | 2,085,559 | 0.035 | 1,605 | 118,417 | 25,578,072 | 1 | 1 | 118,416 | LoopComplete=1 | - | - |
| allocation/ctor_mixed | before-interpreter | 356,930,512 | 353,066,846 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_mixed | before-tiered | 355,808,420 | 351,855,060 | 0.997 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_side_exit | interpreter | 346,930,863 | 353,403,382 | 1.000 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_side_exit | tiered | 40,300,352 | 49,237,465 | 0.116 | 3,490 | 116,533 | 25,171,128 | 1 | 2,379 | 116,533 | GuardFail=2379 | Branch=2379 | UnsupportedOperand=3 |
| allocation/ctor_side_exit | before-interpreter | 346,801,917 | 353,279,876 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| allocation/ctor_side_exit | before-tiered | 346,496,446 | 353,265,994 | 0.999 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |

| Benchmark | Mode | before | after | change |
| :-- | :-- | --: | --: | --: |
| table/array_read | interpreter | 327,958,906 | 327,967,951 | +0.00% |
| table/array_read | tiered | 68,405,539 | 73,713,081 | +7.76% |
| table/array_write | interpreter | 560,612,340 | 560,621,046 | +0.00% |
| table/array_write | tiered | 57,875,466 | 57,985,116 | +0.19% |
| table/hash_read | interpreter | 224,920,138 | 224,930,929 | +0.00% |
| table/hash_read | tiered | 223,351,662 | 148,457,155 | -33.53% |
| table/field_read | interpreter | 401,292,244 | 401,302,426 | +0.00% |
| table/field_read | tiered | 24,662,830 | 25,138,980 | +1.93% |
| table/metamethod_index | interpreter | 304,582,388 | 304,591,376 | +0.00% |
| table/metamethod_index | tiered | 16,766,115 | 16,945,405 | +1.07% |
| table/iteration | interpreter | 508,396,803 | 508,406,651 | +0.00% |
| table/iteration | tiered | 36,232,002 | 41,311,484 | +14.02% |
| allocation/table_churn | interpreter | 68,469,969 | 68,560,134 | +0.13% |
| allocation/table_churn | tiered | 68,190,767 | 68,288,615 | +0.14% |
| allocation/gc_cycles | interpreter | 144,841,973 | 144,931,811 | +0.06% |
| allocation/gc_cycles | tiered | 144,842,295 | 144,952,105 | +0.08% |
| algorithm/sieve | interpreter | 276,549,147 | 276,557,656 | +0.00% |
| algorithm/sieve | tiered | 136,030,841 | 136,157,171 | +0.09% |
| algorithm/nbody | interpreter | 76,697,260 | 76,708,404 | +0.01% |
| algorithm/nbody | tiered | 108,315,322 | 110,386,666 | +1.91% |
| algorithm/mandelbrot | interpreter | 798,491,517 | 798,500,071 | +0.00% |
| algorithm/mandelbrot | tiered | 89,289,838 | 95,751,960 | +7.24% |
| algorithm/spectral_norm | interpreter | 763,104,263 | 763,114,162 | +0.00% |
| algorithm/spectral_norm | tiered | 99,904,875 | 103,282,458 | +3.38% |
| algorithm/binary_trees | interpreter | 323,291,452 | 323,382,534 | +0.03% |
| algorithm/binary_trees | tiered | 323,177,992 | 323,266,725 | +0.03% |
| mineclonia/captured_call | interpreter | 85,970,678 | 85,981,078 | +0.01% |
| mineclonia/captured_call | tiered | 48,059,334 | 47,888,444 | -0.36% |
| mineclonia/module_calls | interpreter | 268,619,107 | 268,629,320 | +0.00% |
| mineclonia/module_calls | tiered | 121,512,441 | 115,483,066 | -4.96% |
| mineclonia/open_results | interpreter | 179,108,197 | 179,119,119 | +0.01% |
| mineclonia/open_results | tiered | 64,410,703 | 63,385,487 | -1.59% |
| mineclonia/ull_kernel | interpreter | 146,048,679 | 146,058,769 | +0.01% |
| mineclonia/ull_kernel | tiered | 151,787,023 | 150,504,792 | -0.84% |
| mineclonia/ull_arith | interpreter | 142,291,091 | 142,300,565 | +0.01% |
| mineclonia/ull_arith | tiered | 45,955,893 | 45,850,688 | -0.23% |
| mineclonia/bit_wide | interpreter | 205,505,428 | 205,516,229 | +0.01% |
| mineclonia/bit_wide | tiered | 50,914,249 | 51,006,183 | +0.18% |
| table/hash_read_small | interpreter | 361,166,076 | 361,175,517 | +0.00% |
| table/hash_read_small | tiered | 358,259,813 | 239,925,684 | -33.03% |
| table/hash_read_shaped | interpreter | 366,113,400 | 366,122,738 | +0.00% |
| table/hash_read_shaped | tiered | 363,104,976 | 231,623,248 | -36.21% |
| table/hash_read_indexed | interpreter | 298,308,406 | 298,319,859 | +0.00% |
| table/hash_read_indexed | tiered | 296,671,794 | 214,959,525 | -27.54% |
| table/heap_redundancy | interpreter | 408,020,518 | 408,030,197 | +0.00% |
| table/heap_redundancy | tiered | 20,674,363 | 20,534,312 | -0.68% |
| allocation/ctor_array | interpreter | 380,018,690 | 380,178,544 | +0.04% |
| allocation/ctor_array | tiered | 379,680,359 | 12,141,747 | -96.80% |
| allocation/ctor_fields | interpreter | 312,888,586 | 313,049,417 | +0.05% |
| allocation/ctor_fields | tiered | 312,560,069 | 13,403,576 | -95.71% |
| allocation/ctor_mixed | interpreter | 356,930,512 | 357,028,725 | +0.03% |
| allocation/ctor_mixed | tiered | 355,808,420 | 12,507,721 | -96.48% |
| allocation/ctor_side_exit | interpreter | 346,801,917 | 346,930,863 | +0.04% |
| allocation/ctor_side_exit | tiered | 346,496,446 | 40,300,352 | -88.37% |
