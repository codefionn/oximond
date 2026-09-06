Load 23:00:11 up 1 day,  1:30,  1 user,  load average: 8,55, 11,94, 11,41; cpu 7 (sibling 15); oximond sha256 7d11c2e8d985b7e8b5b0fbd07a9a25642bc0a17264b86943d0be79e32b1bf1a0

| Benchmark | Mode | instructions:u | vs interpreter | tables | elided | elided bytes | traces | entries | iterations | exits | side exits | aborts | loads_forwarded | loads_merged | loads_hoisted | alloc_count | alloc_bytes | alloc_peak_bytes |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: |
| startup/empty | interpreter | 3,529,821 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 508 | 77,140 | 53,092 |
| startup/empty | tiered | 3,529,288 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 507 | 77,082 | 53,064 |
| startup/empty | before-interpreter | 3,526,514 | 0.999 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 508 | 77,018 | 53,010 |
| startup/empty | before-tiered | 3,525,532 | 0.999 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 507 | 76,960 | 52,982 |
| numeric/for_sum | interpreter | 197,691,174 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 715 | 144,683 | 63,450 |
| numeric/for_sum | tiered | 24,941,953 | 0.126 | 22 | 0 | 0 | 1 | 8 | 1,999,934 | LoopComplete=1, Poll=7 | - | - | 0 | 0 | 0 | 1,487 | 402,532 | 201,590 |
| numeric/for_sum | before-interpreter | 197,688,068 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 715 | 144,561 | 63,378 |
| numeric/for_sum | before-tiered | 24,925,363 | 0.126 | 22 | 0 | 0 | 1 | 8 | 1,999,934 | LoopComplete=1, Poll=7 | - | - | 0 | 0 | 0 | 1,469 | 399,245 | 201,222 |
| numeric/while_mix | interpreter | 656,552,475 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 789 | 178,544 | 68,473 |
| numeric/while_mix | tiered | 32,677,215 | 0.050 | 22 | 0 | 0 | 1 | 5 | 1,199,934 | LoopComplete=1, Poll=4 | - | - | 0 | 0 | 0 | 1,621 | 453,758 | 190,600 |
| numeric/while_mix | before-interpreter | 656,549,806 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 789 | 178,422 | 68,401 |
| numeric/while_mix | before-tiered | 32,659,986 | 0.050 | 22 | 0 | 0 | 1 | 5 | 1,199,934 | LoopComplete=1, Poll=4 | - | - | 0 | 0 | 0 | 1,603 | 449,364 | 190,256 |
| control/branching | interpreter | 527,807,672 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 790 | 190,281 | 67,964 |
| control/branching | tiered | 40,558,947 | 0.077 | 22 | 0 | 0 | 1 | 21 | 1,199,917 | GuardFail=16, LoopComplete=1, Poll=4 | Branch=16 | - | 0 | 0 | 0 | 3,493 | 1,218,454 | 382,981 |
| control/branching | before-interpreter | 527,802,835 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 790 | 190,159 | 67,892 |
| control/branching | before-tiered | 40,490,908 | 0.077 | 22 | 0 | 0 | 1 | 21 | 1,199,917 | GuardFail=16, LoopComplete=1, Poll=4 | Branch=16 | - | 0 | 0 | 0 | 3,437 | 1,202,601 | 382,027 |
| control/nested_loops | interpreter | 171,081,755 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 746 | 171,943 | 65,025 |
| control/nested_loops | tiered | 18,364,182 | 0.107 | 22 | 0 | 0 | 1 | 19 | 639,917 | LoopComplete=17, Poll=2 | - | - | 0 | 0 | 0 | 3,417 | 1,195,410 | 396,369 |
| control/nested_loops | before-interpreter | 171,078,786 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 746 | 171,821 | 64,953 |
| control/nested_loops | before-tiered | 18,289,575 | 0.107 | 22 | 0 | 0 | 1 | 19 | 639,917 | LoopComplete=17, Poll=2 | - | - | 0 | 0 | 0 | 3,361 | 1,178,889 | 395,320 |
| control/nested_calls | interpreter | 3,437,859,990 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 951 | 251,647 | 77,331 |
| control/nested_calls | tiered | 163,812,953 | 0.048 | 22 | 0 | 0 | 1 | 63 | 7,999,867 | GuardFail=16, LoopComplete=17, Poll=30 | Branch=16 | LoopNotClosed=1 | 0 | 0 | 0 | 6,499 | 2,665,710 | 607,840 |
| control/nested_calls | before-interpreter | 3,437,855,900 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 951 | 251,525 | 77,259 |
| control/nested_calls | before-tiered | 163,633,772 | 0.048 | 22 | 0 | 0 | 1 | 63 | 7,999,867 | GuardFail=16, LoopComplete=17, Poll=30 | Branch=16 | LoopNotClosed=1 | 0 | 0 | 0 | 6,389 | 2,621,896 | 605,834 |
| calls/function | interpreter | 322,165,749 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 819 | 177,686 | 69,795 |
| calls/function | tiered | 9,789,938 | 0.030 | 22 | 0 | 0 | 1 | 2 | 399,934 | LoopComplete=1, Poll=1 | - | - | 0 | 0 | 0 | 1,890 | 530,451 | 232,209 |
| calls/function | before-interpreter | 322,162,456 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 819 | 177,564 | 69,723 |
| calls/function | before-tiered | 9,767,185 | 0.030 | 22 | 0 | 0 | 1 | 2 | 399,934 | LoopComplete=1, Poll=1 | - | - | 0 | 0 | 0 | 1,872 | 524,973 | 231,817 |
| calls/builtin | interpreter | 64,279,239 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 706 | 149,459 | 64,823 |
| calls/builtin | tiered | 10,674,036 | 0.166 | 22 | 0 | 0 | 1 | 1 | 199,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 1,799 | 495,781 | 227,664 |
| calls/builtin | before-interpreter | 64,275,761 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 706 | 149,337 | 64,751 |
| calls/builtin | before-tiered | 10,652,427 | 0.166 | 22 | 0 | 0 | 1 | 1 | 199,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 1,781 | 490,666 | 227,272 |
| calls/method | interpreter | 352,574,773 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 915 | 210,629 | 73,498 |
| calls/method | tiered | 10,112,387 | 0.029 | 23 | 0 | 0 | 1 | 1 | 249,934 | LoopComplete=1 | - | - | 1 | 0 | 0 | 2,041 | 603,236 | 238,386 |
| calls/method | before-interpreter | 352,571,049 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 915 | 210,507 | 73,426 |
| calls/method | before-tiered | 10,086,127 | 0.029 | 23 | 0 | 0 | 1 | 1 | 249,934 | LoopComplete=1 | - | - | 1 | 0 | 0 | 2,023 | 596,880 | 237,994 |
| calls/closure | interpreter | 304,419,889 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 930 | 225,678 | 75,848 |
| calls/closure | tiered | 10,500,777 | 0.034 | 22 | 0 | 0 | 1 | 2 | 299,934 | LoopComplete=1, Poll=1 | - | - | 0 | 0 | 0 | 2,024 | 605,658 | 251,363 |
| calls/closure | before-interpreter | 304,416,761 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 930 | 225,556 | 75,776 |
| calls/closure | before-tiered | 10,477,018 | 0.034 | 22 | 0 | 0 | 1 | 2 | 299,934 | LoopComplete=1, Poll=1 | - | - | 0 | 0 | 0 | 2,006 | 599,617 | 250,971 |
| calls/recursive_fib | interpreter | 209,676,574 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 904 | 224,431 | 73,061 |
| calls/recursive_fib | tiered | 49,197,249 | 0.235 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 2,441 | 1,265,372 | 613,560 |
| calls/recursive_fib | before-interpreter | 209,672,466 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 904 | 224,309 | 72,989 |
| calls/recursive_fib | before-tiered | 49,194,297 | 0.235 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 2,441 | 1,265,250 | 613,478 |
| calls/tail_call | interpreter | 283,484,055 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 988 | 245,966 | 76,086 |
| calls/tail_call | tiered | 76,132,097 | 0.269 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 4,639 | 2,591,627 | 668,405 |
| calls/tail_call | before-interpreter | 283,480,959 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 988 | 245,844 | 76,014 |
| calls/tail_call | before-tiered | 76,128,095 | 0.269 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 4,639 | 2,591,505 | 668,323 |
| calls/varargs | interpreter | 378,817,547 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 892 | 216,961 | 71,350 |
| calls/varargs | tiered | 9,238,599 | 0.024 | 22 | 0 | 0 | 1 | 1 | 249,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 1,983 | 649,988 | 248,550 |
| calls/varargs | before-interpreter | 378,814,109 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 892 | 216,839 | 71,278 |
| calls/varargs | before-tiered | 9,212,150 | 0.024 | 22 | 0 | 0 | 1 | 1 | 249,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 1,965 | 642,068 | 248,158 |
| calls/varargs_forward | interpreter | 463,391,616 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 989 | 228,516 | 74,472 |
| calls/varargs_forward | tiered | 9,568,586 | 0.021 | 22 | 0 | 0 | 1 | 1 | 249,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 2,205 | 731,203 | 271,942 |
| calls/varargs_forward | before-interpreter | 464,388,677 | 1.002 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 989 | 228,394 | 74,400 |
| calls/varargs_forward | before-tiered | 9,537,067 | 0.021 | 22 | 0 | 0 | 1 | 1 | 249,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 2,187 | 720,583 | 271,430 |
| table/array_read | interpreter | 327,967,862 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 838 | 1,250,567 | 585,400 |
| table/array_read | tiered | 60,704,791 | 0.185 | 23 | 0 | 0 | 1 | 21 | 1,599,263 | LoopComplete=17, Poll=4 | - | ShortLoop=3 | 0 | 0 | 0 | 4,030 | 2,468,672 | 971,720 |
| table/array_read | before-interpreter | 327,965,677 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 838 | 1,250,445 | 585,318 |
| table/array_read | before-tiered | 60,606,511 | 0.185 | 23 | 0 | 0 | 1 | 21 | 1,599,263 | LoopComplete=17, Poll=4 | - | ShortLoop=3 | 0 | 0 | 0 | 3,962 | 2,449,193 | 970,622 |
| table/array_write | interpreter | 560,621,975 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 845 | 1,247,184 | 585,454 |
| table/array_write | tiered | 54,821,694 | 0.098 | 23 | 0 | 0 | 1 | 4 | 799,934 | LoopComplete=1, Poll=3 | - | ShortLoop=3 | 0 | 0 | 0 | 2,410 | 1,860,212 | 618,723 |
| table/array_write | before-interpreter | 560,619,757 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 845 | 1,247,062 | 585,372 |
| table/array_write | before-tiered | 54,777,164 | 0.098 | 23 | 0 | 0 | 1 | 4 | 799,934 | LoopComplete=1, Poll=3 | - | ShortLoop=3 | 0 | 0 | 0 | 2,380 | 1,852,072 | 617,685 |
| table/hash_read | interpreter | 224,919,224 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 3,256 | 736,863 | 440,019 |
| table/hash_read | tiered | 141,846,061 | 0.631 | 24 | 0 | 0 | 1 | 19 | 599,705 | LoopComplete=17, Poll=2 | - | UnsupportedOpcode=3 | 0 | 0 | 0 | 6,687 | 2,169,938 | 842,839 |
| table/hash_read | before-interpreter | 224,916,109 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 3,256 | 736,741 | 439,937 |
| table/hash_read | before-tiered | 223,361,370 | 0.993 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3, UnsupportedOperand=3 | 0 | 0 | 0 | 3,303 | 746,707 | 440,477 |
| table/field_read | interpreter | 401,301,917 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 858 | 196,450 | 69,952 |
| table/field_read | tiered | 15,326,985 | 0.038 | 23 | 0 | 0 | 1 | 2 | 499,934 | LoopComplete=1, Poll=1 | - | - | 0 | 0 | 4 | 1,919 | 600,993 | 241,390 |
| table/field_read | before-interpreter | 401,298,779 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 858 | 196,328 | 69,880 |
| table/field_read | before-tiered | 15,302,867 | 0.038 | 23 | 0 | 0 | 1 | 2 | 499,934 | LoopComplete=1, Poll=1 | - | - | 0 | 0 | 4 | 1,901 | 595,080 | 241,022 |
| table/metamethod_index | interpreter | 304,592,275 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 852 | 192,796 | 70,454 |
| table/metamethod_index | tiered | 9,067,449 | 0.030 | 25 | 0 | 0 | 1 | 1 | 199,934 | LoopComplete=1 | - | - | 0 | 0 | 2 | 1,917 | 562,251 | 233,119 |
| table/metamethod_index | before-interpreter | 304,589,378 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 852 | 192,674 | 70,382 |
| table/metamethod_index | before-tiered | 9,045,272 | 0.030 | 25 | 0 | 0 | 1 | 1 | 199,934 | LoopComplete=1 | - | - | 0 | 0 | 2 | 1,899 | 556,866 | 232,751 |
| table/iteration | interpreter | 508,407,443 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 997 | 731,226 | 323,866 |
| table/iteration | tiered | 32,089,975 | 0.063 | 23 | 0 | 0 | 1 | 80 | 797,395 | GuardFail=80 | ArrayBounds=80 | ShortLoop=3, UnsupportedOperand=1 | 0 | 0 | 0 | 2,320 | 1,129,713 | 495,043 |
| table/iteration | before-interpreter | 508,403,443 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 997 | 731,104 | 323,784 |
| table/iteration | before-tiered | 32,039,722 | 0.063 | 23 | 0 | 0 | 1 | 80 | 797,395 | GuardFail=80 | ArrayBounds=80 | ShortLoop=3, UnsupportedOperand=1 | 0 | 0 | 0 | 2,290 | 1,122,569 | 494,675 |
| allocation/table_churn | interpreter | 68,723,038 | 1.000 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 948 | 190,212 | 65,451 |
| allocation/table_churn | tiered | 6,346,618 | 0.092 | 23 | 79,999 | 17,279,784 | 1 | 1 | 79,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 1,787 | 532,462 | 221,872 |
| allocation/table_churn | before-interpreter | 68,560,372 | 0.998 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 948 | 190,090 | 65,379 |
| allocation/table_churn | before-tiered | 68,278,459 | 0.994 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 | 0 | 0 | 0 | 876 | 187,594 | 65,374 |
| allocation/gc_cycles | interpreter | 144,927,103 | 1.000 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 81,845 | 16,565,615 | 1,227,898 |
| allocation/gc_cycles | tiered | 145,017,708 | 1.001 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | VirtualEscape=3 | 0 | 0 | 0 | 81,871 | 16,571,957 | 1,228,438 |
| allocation/gc_cycles | before-interpreter | 144,924,607 | 1.000 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 81,845 | 16,565,493 | 1,227,816 |
| allocation/gc_cycles | before-tiered | 144,989,652 | 1.000 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 | 0 | 0 | 0 | 81,862 | 16,569,699 | 1,228,356 |
| string/naive_concat | interpreter | 151,514,301 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 255,304 | 32,997,844 | 68,443 |
| string/naive_concat | tiered | 151,447,652 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 255,357 | 33,007,882 | 68,438 |
| string/naive_concat | before-interpreter | 151,519,855 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 255,304 | 32,997,722 | 68,371 |
| string/naive_concat | before-tiered | 151,446,415 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 255,357 | 33,007,616 | 68,366 |
| string/table_concat | interpreter | 183,781,701 | 1.000 | 102 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 2,378 | 5,942,292 | 379,890 |
| string/table_concat | tiered | 183,441,943 | 0.998 | 102 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 2,398 | 5,947,602 | 380,430 |
| string/table_concat | before-interpreter | 183,784,057 | 1.000 | 102 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 2,378 | 5,942,170 | 379,808 |
| string/table_concat | before-tiered | 183,445,024 | 0.998 | 102 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 2,398 | 5,947,336 | 380,348 |
| string/plain_find | interpreter | 31,592,820 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 804 | 188,666 | 68,374 |
| string/plain_find | tiered | 31,646,272 | 1.002 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 866 | 198,926 | 68,369 |
| string/plain_find | before-interpreter | 31,730,622 | 1.004 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 804 | 188,544 | 68,302 |
| string/plain_find | before-tiered | 31,783,734 | 1.006 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 866 | 198,660 | 68,297 |
| string/pattern_gsub | interpreter | 76,593,090 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 6,815 | 1,679,218 | 69,092 |
| string/pattern_gsub | tiered | 76,706,955 | 1.001 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 6,850 | 1,686,256 | 69,087 |
| string/pattern_gsub | before-interpreter | 76,589,030 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 6,815 | 1,679,096 | 69,020 |
| string/pattern_gsub | before-tiered | 76,703,824 | 1.001 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 6,850 | 1,685,990 | 69,015 |
| string/format | interpreter | 241,105,053 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 240,786 | 8,088,279 | 65,020 |
| string/format | tiered | 240,991,225 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 240,845 | 8,098,419 | 65,015 |
| string/format | before-interpreter | 241,101,654 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 240,786 | 8,088,157 | 64,948 |
| string/format | before-tiered | 240,987,676 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 240,845 | 8,098,153 | 64,943 |
| stdlib/math | interpreter | 440,906,444 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 852 | 198,945 | 69,538 |
| stdlib/math | tiered | 53,551,644 | 0.121 | 22 | 0 | 0 | 1 | 1 | 249,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 2,064 | 665,903 | 254,420 |
| stdlib/math | before-interpreter | 440,902,748 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 852 | 198,823 | 69,466 |
| stdlib/math | before-tiered | 53,517,320 | 0.121 | 22 | 0 | 0 | 1 | 1 | 249,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 2,046 | 657,473 | 253,884 |
| stdlib/native_calls | interpreter | 149,269,422 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 50,923 | 2,631,448 | 72,805 |
| stdlib/native_calls | tiered | 149,261,036 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 51,003 | 2,647,966 | 72,800 |
| stdlib/native_calls | before-interpreter | 149,266,053 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 50,923 | 2,631,326 | 72,733 |
| stdlib/native_calls | before-tiered | 149,255,871 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 51,003 | 2,647,412 | 72,728 |
| stdlib/sort | interpreter | 76,793,026 | 1.000 | 34 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,146 | 3,400,909 | 459,494 |
| stdlib/sort | tiered | 69,098,554 | 0.900 | 34 | 0 | 0 | 1 | 12 | 95,923 | LoopComplete=12 | - | - | 0 | 0 | 0 | 2,542 | 3,974,314 | 610,607 |
| stdlib/sort | before-interpreter | 76,789,480 | 1.000 | 34 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,146 | 3,400,787 | 459,412 |
| stdlib/sort | before-tiered | 69,075,060 | 0.899 | 34 | 0 | 0 | 1 | 12 | 95,923 | LoopComplete=12 | - | - | 0 | 0 | 0 | 2,524 | 3,968,902 | 610,227 |
| algorithm/sieve | interpreter | 276,560,579 | 1.000 | 27 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,224 | 16,029,998 | 4,807,359 |
| algorithm/sieve | tiered | 126,639,940 | 0.458 | 27 | 0 | 0 | 2 | 469 | 715,775 | GuardFail=224, LoopComplete=245 | ArrayBounds=36, Branch=188 | - | 0 | 0 | 0 | 6,645 | 17,948,298 | 7,986,731 |
| algorithm/sieve | before-interpreter | 276,555,386 | 1.000 | 27 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,224 | 16,029,876 | 4,807,277 |
| algorithm/sieve | before-tiered | 126,465,517 | 0.457 | 27 | 0 | 0 | 2 | 469 | 715,775 | GuardFail=224, LoopComplete=245 | ArrayBounds=36, Branch=188 | - | 0 | 0 | 0 | 6,571 | 17,917,285 | 7,985,352 |
| algorithm/nbody | interpreter | 76,707,873 | 1.000 | 28 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 3,177 | 1,167,678 | 139,763 |
| algorithm/nbody | tiered | 74,015,113 | 0.965 | 28 | 0 | 0 | 2 | 1,977 | 12,730 | LoopComplete=1977 | - | InnerLoop=6, LoopNotClosed=2 | 0 | 0 | 4 | 17,484 | 9,086,462 | 1,852,068 |
| algorithm/nbody | before-interpreter | 76,704,882 | 1.000 | 28 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 3,177 | 1,167,556 | 139,691 |
| algorithm/nbody | before-tiered | 73,197,665 | 0.954 | 28 | 0 | 0 | 2 | 1,977 | 12,730 | LoopComplete=1977 | - | InnerLoop=6, LoopNotClosed=2 | 0 | 0 | 4 | 17,410 | 8,967,808 | 1,846,283 |
| algorithm/mandelbrot | interpreter | 798,500,824 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,290 | 455,518 | 83,888 |
| algorithm/mandelbrot | tiered | 61,656,770 | 0.077 | 22 | 0 | 0 | 1 | 64 | 822,874 | GuardFail=16, LoopComplete=46, Poll=2 | Branch=16 | - | 0 | 0 | 0 | 9,750 | 5,620,380 | 966,420 |
| algorithm/mandelbrot | before-interpreter | 798,497,046 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,290 | 455,396 | 83,816 |
| algorithm/mandelbrot | before-tiered | 61,307,947 | 0.077 | 22 | 0 | 0 | 1 | 64 | 822,874 | GuardFail=16, LoopComplete=46, Poll=2 | Branch=16 | - | 0 | 0 | 0 | 9,570 | 5,541,615 | 964,284 |
| algorithm/spectral_norm | interpreter | 763,115,657 | 1.000 | 44 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 2,447 | 892,669 | 132,313 |
| algorithm/spectral_norm | tiered | 74,964,075 | 0.098 | 44 | 0 | 0 | 2 | 4,868 | 571,000 | GuardFail=4796, LoopComplete=72 | ArrayBounds=4796 | ShortLoop=4 | 0 | 0 | 0 | 21,608 | 5,343,530 | 718,268 |
| algorithm/spectral_norm | before-interpreter | 763,111,069 | 1.000 | 44 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 2,447 | 892,547 | 132,231 |
| algorithm/spectral_norm | before-tiered | 74,206,384 | 0.097 | 44 | 0 | 0 | 2 | 4,868 | 571,000 | GuardFail=4796, LoopComplete=72 | ArrayBounds=4796 | ShortLoop=4 | 0 | 0 | 0 | 21,480 | 5,272,048 | 715,937 |
| algorithm/binary_trees | interpreter | 323,405,070 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 85,140 | 18,339,810 | 5,771,469 |
| algorithm/binary_trees | tiered | 323,280,594 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 85,164 | 18,345,518 | 5,774,317 |
| algorithm/binary_trees | before-interpreter | 323,401,951 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 85,140 | 18,339,688 | 5,771,387 |
| algorithm/binary_trees | before-tiered | 323,271,094 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 85,164 | 18,345,396 | 5,774,235 |
| mineclonia/captured_call | interpreter | 85,982,953 | 1.000 | 31 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 21,807 | 2,290,293 | 259,886 |
| mineclonia/captured_call | tiered | 24,664,132 | 0.287 | 31 | 0 | 0 | 1 | 1 | 4,934 | LoopComplete=1 | - | - | 4 | 0 | 0 | 12,248 | 5,084,756 | 1,058,506 |
| mineclonia/captured_call | before-interpreter | 85,979,115 | 1.000 | 31 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 21,807 | 2,290,171 | 259,814 |
| mineclonia/captured_call | before-tiered | 24,354,177 | 0.283 | 31 | 0 | 0 | 1 | 1 | 4,934 | LoopComplete=1 | - | - | 4 | 0 | 0 | 12,230 | 5,022,746 | 1,056,602 |
| mineclonia/module_calls | interpreter | 268,630,225 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 4,021 | 1,345,692 | 189,169 |
| mineclonia/module_calls | tiered | 48,954,285 | 0.182 | 25 | 0 | 0 | 1 | 1 | 9,934 | LoopComplete=1 | - | - | 13 | 0 | 0 | 14,686 | 8,170,144 | 2,344,246 |
| mineclonia/module_calls | before-interpreter | 268,626,266 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 4,021 | 1,345,570 | 189,097 |
| mineclonia/module_calls | before-tiered | 48,277,038 | 0.180 | 25 | 0 | 0 | 1 | 1 | 9,934 | LoopComplete=1 | - | - | 13 | 0 | 0 | 14,668 | 8,031,631 | 2,340,302 |
| mineclonia/open_results | interpreter | 179,119,484 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 41,012 | 1,481,821 | 117,060 |
| mineclonia/open_results | tiered | 28,182,704 | 0.157 | 23 | 0 | 0 | 2 | 1,492 | 29,877 | CommittedEffect=748, LoopComplete=744 | Branch=748 | - | 2 | 0 | 0 | 8,600 | 3,161,060 | 568,516 |
| mineclonia/open_results | before-interpreter | 179,116,047 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 41,012 | 1,481,699 | 116,988 |
| mineclonia/open_results | before-tiered | 27,847,205 | 0.155 | 23 | 0 | 0 | 2 | 1,492 | 29,877 | CommittedEffect=748, LoopComplete=744 | Branch=748 | - | 2 | 0 | 0 | 8,564 | 3,116,238 | 567,476 |
| mineclonia/ull_kernel | interpreter | 146,060,916 | 1.000 | 32 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 6,450 | 2,217,547 | 210,832 |
| mineclonia/ull_kernel | tiered | 52,254,682 | 0.358 | 32 | 0 | 0 | 1 | 1 | 3,934 | LoopComplete=1 | - | - | 15 | 0 | 0 | 19,370 | 10,133,313 | 2,568,618 |
| mineclonia/ull_kernel | before-interpreter | 146,056,705 | 1.000 | 32 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 6,450 | 2,217,425 | 210,760 |
| mineclonia/ull_kernel | before-tiered | 51,370,075 | 0.352 | 32 | 0 | 0 | 1 | 1 | 3,934 | LoopComplete=1 | - | - | 15 | 0 | 0 | 19,352 | 9,955,731 | 2,562,658 |
| mineclonia/ull_arith | interpreter | 142,300,855 | 1.000 | 30 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,692 | 527,222 | 104,756 |
| mineclonia/ull_arith | tiered | 12,485,678 | 0.088 | 30 | 0 | 0 | 1 | 1 | 49,934 | LoopComplete=1 | - | - | 3 | 0 | 0 | 3,249 | 1,447,963 | 443,870 |
| mineclonia/ull_arith | before-interpreter | 142,297,762 | 1.000 | 30 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,692 | 527,100 | 104,674 |
| mineclonia/ull_arith | before-tiered | 12,425,558 | 0.087 | 30 | 0 | 0 | 1 | 1 | 49,934 | LoopComplete=1 | - | - | 3 | 0 | 0 | 3,231 | 1,430,303 | 442,974 |
| mineclonia/bit_wide | interpreter | 205,518,162 | 1.000 | 29 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 2,393 | 765,784 | 111,090 |
| mineclonia/bit_wide | tiered | 12,488,489 | 0.061 | 29 | 0 | 0 | 1 | 1 | 49,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 4,361 | 1,841,905 | 432,756 |
| mineclonia/bit_wide | before-interpreter | 205,512,962 | 1.000 | 29 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 2,393 | 765,662 | 111,018 |
| mineclonia/bit_wide | before-tiered | 12,400,199 | 0.060 | 29 | 0 | 0 | 1 | 1 | 49,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 4,343 | 1,817,177 | 431,908 |
| table/hash_read_small | interpreter | 361,176,396 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 938 | 246,848 | 74,030 |
| table/hash_read_small | tiered | 223,525,077 | 0.619 | 24 | 0 | 0 | 1 | 20 | 959,917 | LoopComplete=17, Poll=3 | - | - | 0 | 0 | 0 | 4,347 | 1,675,329 | 517,321 |
| table/hash_read_small | before-interpreter | 361,173,387 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 938 | 246,726 | 73,958 |
| table/hash_read_small | before-tiered | 358,265,689 | 0.992 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 | 0 | 0 | 0 | 961 | 251,832 | 73,953 |
| table/hash_read_shaped | interpreter | 366,123,601 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,032 | 254,605 | 75,639 |
| table/hash_read_shaped | tiered | 213,265,099 | 0.582 | 24 | 0 | 0 | 1 | 20 | 959,884 | LoopComplete=17, Poll=3 | - | LoopNotClosed=1 | 0 | 0 | 0 | 4,454 | 1,689,338 | 518,093 |
| table/hash_read_shaped | before-interpreter | 366,119,288 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,032 | 254,483 | 75,567 |
| table/hash_read_shaped | before-tiered | 363,112,895 | 0.992 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | InnerLoop=1, UnsupportedOperand=3 | 0 | 0 | 0 | 1,062 | 261,761 | 75,562 |
| table/hash_read_indexed | interpreter | 298,342,821 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 8,827 | 1,234,557 | 701,220 |
| table/hash_read_indexed | tiered | 202,134,110 | 0.678 | 24 | 0 | 0 | 1 | 35 | 653,774 | GuardFail=16, LoopComplete=17, Poll=2 | ArrayBounds=16 | UnsupportedOpcode=6 | 0 | 0 | 0 | 15,262 | 4,285,574 | 1,435,811 |
| table/hash_read_indexed | before-interpreter | 298,337,954 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 8,827 | 1,234,435 | 701,138 |
| table/hash_read_indexed | before-tiered | 296,680,897 | 0.994 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=6, UnsupportedOperand=3 | 0 | 0 | 0 | 8,910 | 1,256,929 | 702,702 |
| table/heap_redundancy | interpreter | 408,029,958 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,195 | 325,694 | 86,001 |
| table/heap_redundancy | tiered | 11,505,432 | 0.028 | 24 | 0 | 0 | 1 | 1 | 149,934 | LoopComplete=1 | - | - | 1 | 6 | 4 | 2,384 | 937,165 | 307,261 |
| table/heap_redundancy | before-interpreter | 408,026,283 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,195 | 325,572 | 85,929 |
| table/heap_redundancy | before-tiered | 11,474,861 | 0.028 | 24 | 0 | 0 | 1 | 1 | 149,934 | LoopComplete=1 | - | - | 1 | 6 | 4 | 2,366 | 927,719 | 306,893 |
| allocation/ctor_array | interpreter | 380,132,510 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 151,135 | 10,245,986 | 410,482 |
| allocation/ctor_array | tiered | 7,560,464 | 0.020 | 87 | 149,935 | 32,385,960 | 1 | 1 | 149,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 1,829 | 564,354 | 237,146 |
| allocation/ctor_array | before-interpreter | 380,128,242 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 151,135 | 10,245,864 | 410,400 |
| allocation/ctor_array | before-tiered | 379,856,226 | 0.999 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 | 0 | 0 | 0 | 151,152 | 10,249,830 | 410,940 |
| allocation/ctor_fields | interpreter | 313,044,581 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 151,280 | 9,061,154 | 367,851 |
| allocation/ctor_fields | tiered | 8,223,575 | 0.026 | 87 | 149,935 | 32,385,960 | 1 | 1 | 149,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 1,876 | 551,276 | 235,711 |
| allocation/ctor_fields | before-interpreter | 313,041,763 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 151,280 | 9,061,032 | 367,769 |
| allocation/ctor_fields | before-tiered | 312,680,144 | 0.999 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 | 0 | 0 | 0 | 151,297 | 9,064,968 | 368,309 |
| allocation/ctor_mixed | interpreter | 356,708,426 | 1.000 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 241,219 | 13,147,075 | 436,537 |
| allocation/ctor_mixed | tiered | 7,966,417 | 0.022 | 87 | 119,935 | 25,905,960 | 1 | 1 | 119,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 1,975 | 592,243 | 243,737 |
| allocation/ctor_mixed | before-interpreter | 356,705,110 | 1.000 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 241,219 | 13,146,953 | 436,455 |
| allocation/ctor_mixed | before-tiered | 356,115,874 | 0.998 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 | 0 | 0 | 0 | 241,236 | 13,150,979 | 436,995 |
| allocation/ctor_side_exit | interpreter | 346,963,981 | 1.000 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 242,392 | 29,484,219 | 2,149,985 |
| allocation/ctor_side_exit | tiered | 37,485,260 | 0.108 | 2,487 | 117,536 | 25,387,776 | 1 | 2,399 | 117,536 | GuardFail=2399 | Branch=2399 | UnsupportedOperand=3 | 0 | 0 | 0 | 13,480 | 2,750,130 | 1,251,120 |
| allocation/ctor_side_exit | before-interpreter | 346,961,815 | 1.000 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 242,392 | 29,484,097 | 2,149,903 |
| allocation/ctor_side_exit | before-tiered | 346,690,378 | 0.999 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 | 0 | 0 | 0 | 242,409 | 29,488,153 | 2,150,443 |
| opt/field_reload | interpreter | 698,079,598 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,527 | 274,581 | 78,178 |
| opt/field_reload | tiered | 17,138,251 | 0.025 | 23 | 0 | 0 | 1 | 2 | 399,934 | LoopComplete=1, Poll=1 | - | - | 2 | 2 | 2 | 2,641 | 755,984 | 259,276 |
| opt/field_reload | before-interpreter | 698,076,129 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,527 | 274,459 | 78,096 |
| opt/field_reload | before-tiered | 17,112,461 | 0.025 | 23 | 0 | 0 | 1 | 2 | 399,934 | LoopComplete=1, Poll=1 | - | - | 2 | 2 | 2 | 2,623 | 748,585 | 258,908 |
| opt/store_load | interpreter | 789,994,052 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,131 | 301,375 | 83,784 |
| opt/store_load | tiered | 22,136,890 | 0.028 | 24 | 0 | 0 | 1 | 2 | 299,934 | LoopComplete=1, Poll=1 | - | - | 4 | 0 | 0 | 2,670 | 1,087,695 | 389,602 |
| opt/store_load | before-interpreter | 789,991,135 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,131 | 301,253 | 83,712 |
| opt/store_load | before-tiered | 22,085,144 | 0.028 | 24 | 0 | 0 | 1 | 2 | 299,934 | LoopComplete=1, Poll=1 | - | - | 4 | 0 | 0 | 2,652 | 1,073,529 | 388,922 |
| opt/alias_holder | interpreter | 876,643,578 | 1.000 | 26 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,033 | 257,849 | 78,414 |
| opt/alias_holder | tiered | 14,183,365 | 0.016 | 26 | 0 | 0 | 1 | 2 | 299,934 | LoopComplete=1, Poll=1 | - | - | 1 | 0 | 3 | 2,155 | 733,021 | 265,892 |
| opt/alias_holder | before-interpreter | 876,640,125 | 1.000 | 26 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,033 | 257,727 | 78,342 |
| opt/alias_holder | before-tiered | 14,161,115 | 0.016 | 26 | 0 | 0 | 1 | 2 | 299,934 | LoopComplete=1, Poll=1 | - | - | 1 | 0 | 3 | 2,137 | 726,247 | 265,524 |
| opt/invariant_field | interpreter | 419,818,228 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 884 | 208,411 | 71,079 |
| opt/invariant_field | tiered | 17,424,716 | 0.042 | 23 | 0 | 0 | 1 | 2 | 499,934 | LoopComplete=1, Poll=1 | - | - | 0 | 1 | 3 | 1,971 | 619,363 | 245,949 |
| opt/invariant_field | before-interpreter | 419,815,230 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 884 | 208,289 | 71,007 |
| opt/invariant_field | before-tiered | 17,397,705 | 0.041 | 23 | 0 | 0 | 1 | 2 | 499,934 | LoopComplete=1, Poll=1 | - | - | 0 | 1 | 3 | 1,953 | 613,216 | 245,581 |

| Benchmark | Mode | before | after | change |
| :-- | :-- | --: | --: | --: |
| startup/empty | interpreter | 3,526,514 | 3,529,821 | +0.09% |
| startup/empty | tiered | 3,525,532 | 3,529,288 | +0.11% |
| numeric/for_sum | interpreter | 197,688,068 | 197,691,174 | +0.00% |
| numeric/for_sum | tiered | 24,925,363 | 24,941,953 | +0.07% |
| numeric/while_mix | interpreter | 656,549,806 | 656,552,475 | +0.00% |
| numeric/while_mix | tiered | 32,659,986 | 32,677,215 | +0.05% |
| control/branching | interpreter | 527,802,835 | 527,807,672 | +0.00% |
| control/branching | tiered | 40,490,908 | 40,558,947 | +0.17% |
| control/nested_loops | interpreter | 171,078,786 | 171,081,755 | +0.00% |
| control/nested_loops | tiered | 18,289,575 | 18,364,182 | +0.41% |
| control/nested_calls | interpreter | 3,437,855,900 | 3,437,859,990 | +0.00% |
| control/nested_calls | tiered | 163,633,772 | 163,812,953 | +0.11% |
| calls/function | interpreter | 322,162,456 | 322,165,749 | +0.00% |
| calls/function | tiered | 9,767,185 | 9,789,938 | +0.23% |
| calls/builtin | interpreter | 64,275,761 | 64,279,239 | +0.01% |
| calls/builtin | tiered | 10,652,427 | 10,674,036 | +0.20% |
| calls/method | interpreter | 352,571,049 | 352,574,773 | +0.00% |
| calls/method | tiered | 10,086,127 | 10,112,387 | +0.26% |
| calls/closure | interpreter | 304,416,761 | 304,419,889 | +0.00% |
| calls/closure | tiered | 10,477,018 | 10,500,777 | +0.23% |
| calls/recursive_fib | interpreter | 209,672,466 | 209,676,574 | +0.00% |
| calls/recursive_fib | tiered | 49,194,297 | 49,197,249 | +0.01% |
| calls/tail_call | interpreter | 283,480,959 | 283,484,055 | +0.00% |
| calls/tail_call | tiered | 76,128,095 | 76,132,097 | +0.01% |
| calls/varargs | interpreter | 378,814,109 | 378,817,547 | +0.00% |
| calls/varargs | tiered | 9,212,150 | 9,238,599 | +0.29% |
| calls/varargs_forward | interpreter | 464,388,677 | 463,391,616 | -0.21% |
| calls/varargs_forward | tiered | 9,537,067 | 9,568,586 | +0.33% |
| table/array_read | interpreter | 327,965,677 | 327,967,862 | +0.00% |
| table/array_read | tiered | 60,606,511 | 60,704,791 | +0.16% |
| table/array_write | interpreter | 560,619,757 | 560,621,975 | +0.00% |
| table/array_write | tiered | 54,777,164 | 54,821,694 | +0.08% |
| table/hash_read | interpreter | 224,916,109 | 224,919,224 | +0.00% |
| table/hash_read | tiered | 223,361,370 | 141,846,061 | -36.49% |
| table/field_read | interpreter | 401,298,779 | 401,301,917 | +0.00% |
| table/field_read | tiered | 15,302,867 | 15,326,985 | +0.16% |
| table/metamethod_index | interpreter | 304,589,378 | 304,592,275 | +0.00% |
| table/metamethod_index | tiered | 9,045,272 | 9,067,449 | +0.25% |
| table/iteration | interpreter | 508,403,443 | 508,407,443 | +0.00% |
| table/iteration | tiered | 32,039,722 | 32,089,975 | +0.16% |
| allocation/table_churn | interpreter | 68,560,372 | 68,723,038 | +0.24% |
| allocation/table_churn | tiered | 68,278,459 | 6,346,618 | -90.70% |
| allocation/gc_cycles | interpreter | 144,924,607 | 144,927,103 | +0.00% |
| allocation/gc_cycles | tiered | 144,989,652 | 145,017,708 | +0.02% |
| string/naive_concat | interpreter | 151,519,855 | 151,514,301 | -0.00% |
| string/naive_concat | tiered | 151,446,415 | 151,447,652 | +0.00% |
| string/table_concat | interpreter | 183,784,057 | 183,781,701 | -0.00% |
| string/table_concat | tiered | 183,445,024 | 183,441,943 | -0.00% |
| string/plain_find | interpreter | 31,730,622 | 31,592,820 | -0.43% |
| string/plain_find | tiered | 31,783,734 | 31,646,272 | -0.43% |
| string/pattern_gsub | interpreter | 76,589,030 | 76,593,090 | +0.01% |
| string/pattern_gsub | tiered | 76,703,824 | 76,706,955 | +0.00% |
| string/format | interpreter | 241,101,654 | 241,105,053 | +0.00% |
| string/format | tiered | 240,987,676 | 240,991,225 | +0.00% |
| stdlib/math | interpreter | 440,902,748 | 440,906,444 | +0.00% |
| stdlib/math | tiered | 53,517,320 | 53,551,644 | +0.06% |
| stdlib/native_calls | interpreter | 149,266,053 | 149,269,422 | +0.00% |
| stdlib/native_calls | tiered | 149,255,871 | 149,261,036 | +0.00% |
| stdlib/sort | interpreter | 76,789,480 | 76,793,026 | +0.00% |
| stdlib/sort | tiered | 69,075,060 | 69,098,554 | +0.03% |
| algorithm/sieve | interpreter | 276,555,386 | 276,560,579 | +0.00% |
| algorithm/sieve | tiered | 126,465,517 | 126,639,940 | +0.14% |
| algorithm/nbody | interpreter | 76,704,882 | 76,707,873 | +0.00% |
| algorithm/nbody | tiered | 73,197,665 | 74,015,113 | +1.12% |
| algorithm/mandelbrot | interpreter | 798,497,046 | 798,500,824 | +0.00% |
| algorithm/mandelbrot | tiered | 61,307,947 | 61,656,770 | +0.57% |
| algorithm/spectral_norm | interpreter | 763,111,069 | 763,115,657 | +0.00% |
| algorithm/spectral_norm | tiered | 74,206,384 | 74,964,075 | +1.02% |
| algorithm/binary_trees | interpreter | 323,401,951 | 323,405,070 | +0.00% |
| algorithm/binary_trees | tiered | 323,271,094 | 323,280,594 | +0.00% |
| mineclonia/captured_call | interpreter | 85,979,115 | 85,982,953 | +0.00% |
| mineclonia/captured_call | tiered | 24,354,177 | 24,664,132 | +1.27% |
| mineclonia/module_calls | interpreter | 268,626,266 | 268,630,225 | +0.00% |
| mineclonia/module_calls | tiered | 48,277,038 | 48,954,285 | +1.40% |
| mineclonia/open_results | interpreter | 179,116,047 | 179,119,484 | +0.00% |
| mineclonia/open_results | tiered | 27,847,205 | 28,182,704 | +1.20% |
| mineclonia/ull_kernel | interpreter | 146,056,705 | 146,060,916 | +0.00% |
| mineclonia/ull_kernel | tiered | 51,370,075 | 52,254,682 | +1.72% |
| mineclonia/ull_arith | interpreter | 142,297,762 | 142,300,855 | +0.00% |
| mineclonia/ull_arith | tiered | 12,425,558 | 12,485,678 | +0.48% |
| mineclonia/bit_wide | interpreter | 205,512,962 | 205,518,162 | +0.00% |
| mineclonia/bit_wide | tiered | 12,400,199 | 12,488,489 | +0.71% |
| table/hash_read_small | interpreter | 361,173,387 | 361,176,396 | +0.00% |
| table/hash_read_small | tiered | 358,265,689 | 223,525,077 | -37.61% |
| table/hash_read_shaped | interpreter | 366,119,288 | 366,123,601 | +0.00% |
| table/hash_read_shaped | tiered | 363,112,895 | 213,265,099 | -41.27% |
| table/hash_read_indexed | interpreter | 298,337,954 | 298,342,821 | +0.00% |
| table/hash_read_indexed | tiered | 296,680,897 | 202,134,110 | -31.87% |
| table/heap_redundancy | interpreter | 408,026,283 | 408,029,958 | +0.00% |
| table/heap_redundancy | tiered | 11,474,861 | 11,505,432 | +0.27% |
| allocation/ctor_array | interpreter | 380,128,242 | 380,132,510 | +0.00% |
| allocation/ctor_array | tiered | 379,856,226 | 7,560,464 | -98.01% |
| allocation/ctor_fields | interpreter | 313,041,763 | 313,044,581 | +0.00% |
| allocation/ctor_fields | tiered | 312,680,144 | 8,223,575 | -97.37% |
| allocation/ctor_mixed | interpreter | 356,705,110 | 356,708,426 | +0.00% |
| allocation/ctor_mixed | tiered | 356,115,874 | 7,966,417 | -97.76% |
| allocation/ctor_side_exit | interpreter | 346,961,815 | 346,963,981 | +0.00% |
| allocation/ctor_side_exit | tiered | 346,690,378 | 37,485,260 | -89.19% |
| opt/field_reload | interpreter | 698,076,129 | 698,079,598 | +0.00% |
| opt/field_reload | tiered | 17,112,461 | 17,138,251 | +0.15% |
| opt/store_load | interpreter | 789,991,135 | 789,994,052 | +0.00% |
| opt/store_load | tiered | 22,085,144 | 22,136,890 | +0.23% |
| opt/alias_holder | interpreter | 876,640,125 | 876,643,578 | +0.00% |
| opt/alias_holder | tiered | 14,161,115 | 14,183,365 | +0.16% |
| opt/invariant_field | interpreter | 419,815,230 | 419,818,228 | +0.00% |
| opt/invariant_field | tiered | 17,397,705 | 17,424,716 | +0.16% |
