Load 22:15:38 up 1 day, 45 min,  1 user,  load average: 5,40, 10,07, 15,84; cpu 7 (sibling 15); oximond sha256 51f51778bae0d71c58ccf37a61a0ae6728c8bfaef2a536c7792b6d64e9c4fc69

| Benchmark | Mode | instructions:u | warmed per scale | vs interpreter | tables | elided | elided bytes | traces | entries | iterations | exits | side exits | aborts |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- | :-- |
| numeric/for_sum | interpreter | 197,690,831 | 193,999,642 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| numeric/for_sum | tiered | 34,560,648 | 19,948,275 | 0.175 | 22 | 0 | 0 | 1 | 8 | 1,902,551 | LoopComplete=1, Poll=7 | - | - |
| numeric/for_sum | before-interpreter | 197,686,621 | 193,999,941 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| numeric/for_sum | before-tiered | 33,783,999 | 20,042,854 | 0.171 | 22 | 0 | 0 | 1 | 8 | 1,910,372 | LoopComplete=1, Poll=7 | - | - |
| numeric/while_mix | interpreter | 656,552,472 | 652,799,906 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| numeric/while_mix | tiered | 39,639,012 | 27,350,597 | 0.060 | 22 | 0 | 0 | 1 | 5 | 1,186,866 | LoopComplete=1, Poll=4 | - | - |
| numeric/while_mix | before-interpreter | 656,549,712 | 652,798,462 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| numeric/while_mix | before-tiered | 39,817,166 | 27,590,252 | 0.061 | 22 | 0 | 0 | 1 | 5 | 1,186,569 | LoopComplete=1, Poll=4 | - | - |
| control/branching | interpreter | 527,807,315 | 524,056,471 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| control/branching | tiered | 48,809,275 | 62,743,894 | 0.092 | 22 | 0 | 0 | 1 | 21 | 1,180,447 | GuardFail=16, LoopComplete=1, Poll=4 | Branch=16 | - |
| control/branching | before-interpreter | 527,803,418 | 524,056,370 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| control/branching | before-tiered | 48,845,111 | 31,400,779 | 0.093 | 22 | 0 | 0 | 1 | 21 | 1,180,348 | GuardFail=16, LoopComplete=1, Poll=4 | Branch=16 | - |
| control/nested_loops | interpreter | 171,081,717 | 167,367,962 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| control/nested_loops | tiered | 27,254,887 | 9,046,511 | 0.159 | 22 | 0 | 0 | 1 | 19 | 605,605 | LoopComplete=17, Poll=2 | - | InnerLoop=0 |
| control/nested_loops | before-interpreter | 171,077,022 | 167,368,567 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| control/nested_loops | before-tiered | 26,674,967 | 10,025,175 | 0.156 | 22 | 0 | 0 | 1 | 19 | 607,814 | LoopComplete=17, Poll=2 | - | - |
| control/nested_calls | interpreter | 3,437,858,485 | 3,434,000,169 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| control/nested_calls | tiered | 198,296,900 | 149,708,501 | 0.058 | 22 | 0 | 0 | 1 | 6,607 | 7,976,015 | GuardFail=6560, LoopComplete=17, Poll=30 | Branch=6560 | InnerLoop=0, LoopNotClosed=1 |
| control/nested_calls | before-interpreter | 3,437,855,054 | 3,434,000,036 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| control/nested_calls | before-tiered | 199,827,093 | 148,952,725 | 0.058 | 22 | 0 | 0 | 1 | 7,099 | 7,975,583 | GuardFail=7052, LoopComplete=17, Poll=30 | Branch=7052 | LoopNotClosed=1 |
| calls/function | interpreter | 322,166,518 | 318,399,541 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/function | tiered | 18,587,641 | 4,376,999 | 0.058 | 22 | 0 | 0 | 1 | 2 | 389,143 | LoopComplete=1, Poll=1 | - | - |
| calls/function | before-interpreter | 322,162,053 | 318,400,526 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/function | before-tiered | 18,528,289 | 4,242,897 | 0.058 | 22 | 0 | 0 | 1 | 2 | 389,209 | LoopComplete=1, Poll=1 | - | - |
| calls/builtin | interpreter | 64,279,110 | 60,599,357 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/builtin | tiered | 20,434,511 | 4,897,536 | 0.318 | 22 | 0 | 0 | 1 | 1 | 166,208 | LoopComplete=1 | - | - |
| calls/builtin | before-interpreter | 64,274,461 | 60,600,166 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/builtin | before-tiered | 20,210,039 | 5,497,545 | 0.314 | 22 | 0 | 0 | 1 | 1 | 166,967 | LoopComplete=1 | - | - |
| calls/method | interpreter | 350,574,717 | 346,749,903 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/method | tiered | 17,299,571 | 4,825,983 | 0.049 | 23 | 0 | 0 | 1 | 1 | 244,786 | LoopComplete=1 | - | - |
| calls/method | before-interpreter | 350,570,310 | 346,750,420 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/method | before-tiered | 17,431,104 | 4,734,711 | 0.050 | 23 | 0 | 0 | 1 | 1 | 244,687 | LoopComplete=1 | - | - |
| calls/closure | interpreter | 304,419,904 | 300,600,352 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/closure | tiered | 17,993,590 | 5,035,774 | 0.059 | 22 | 0 | 0 | 1 | 2 | 292,575 | LoopComplete=1, Poll=1 | - | - |
| calls/closure | before-interpreter | 304,414,896 | 300,600,351 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/closure | before-tiered | 17,922,525 | 5,168,816 | 0.059 | 22 | 0 | 0 | 1 | 2 | 292,641 | LoopComplete=1, Poll=1 | - | - |
| calls/recursive_fib | interpreter | 209,676,128 | 205,868,929 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/recursive_fib | tiered | 49,202,831 | 37,608,453 | 0.235 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/recursive_fib | before-interpreter | 209,671,816 | 205,868,448 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/recursive_fib | before-tiered | 49,198,742 | 37,608,492 | 0.235 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/tail_call | interpreter | 283,483,439 | 279,625,346 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/tail_call | tiered | 76,125,433 | 55,000,963 | 0.269 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/tail_call | before-interpreter | 283,479,780 | 279,625,501 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/tail_call | before-tiered | 76,121,332 | 55,001,058 | 0.269 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/varargs | interpreter | 378,817,953 | 375,000,380 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/varargs | tiered | 17,309,504 | 4,002,634 | 0.046 | 22 | 0 | 0 | 1 | 1 | 244,621 | LoopComplete=1 | - | - |
| calls/varargs | before-interpreter | 378,813,960 | 375,000,204 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/varargs | before-tiered | 17,355,697 | 3,600,820 | 0.046 | 22 | 0 | 0 | 1 | 1 | 244,588 | LoopComplete=1 | - | - |
| calls/varargs_forward | interpreter | 463,390,947 | 459,500,428 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/varargs_forward | tiered | 15,671,129 | 4,022,675 | 0.034 | 22 | 0 | 0 | 1 | 1 | 246,469 | LoopComplete=1 | - | - |
| calls/varargs_forward | before-interpreter | 463,387,073 | 459,500,427 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| calls/varargs_forward | before-tiered | 16,301,468 | 3,501,767 | 0.035 | 22 | 0 | 0 | 1 | 1 | 246,106 | LoopComplete=1 | - | - |
| table/array_read | interpreter | 327,968,362 | 315,233,183 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/array_read | tiered | 69,848,149 | 39,627,575 | 0.213 | 23 | 0 | 0 | 1 | 52 | 1,553,428 | LoopComplete=50, Poll=2 | - | InnerLoop=0, ShortLoop=3 |
| table/array_read | before-interpreter | 327,965,112 | 315,232,594 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/array_read | before-tiered | 70,564,336 | 43,721,241 | 0.215 | 23 | 0 | 0 | 1 | 56 | 1,550,417 | LoopComplete=55, Poll=1 | - | ShortLoop=3 |
| table/array_write | interpreter | 560,622,656 | 551,999,226 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/array_write | tiered | 58,020,619 | 36,601,440 | 0.103 | 23 | 0 | 0 | 1 | 4 | 790,760 | LoopComplete=1, Poll=3 | - | ShortLoop=3 |
| table/array_write | before-interpreter | 560,617,802 | 552,000,865 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/array_write | before-tiered | 57,914,763 | 36,625,440 | 0.103 | 23 | 0 | 0 | 1 | 4 | 791,024 | LoopComplete=1, Poll=3 | - | ShortLoop=3 |
| table/hash_read | interpreter | 224,930,784 | 211,621,679 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read | tiered | 148,553,156 | 120,191,950 | 0.660 | 24 | 0 | 0 | 1 | 19 | 558,706 | LoopComplete=17, Poll=2 | - | UnsupportedOpcode=3 |
| table/hash_read | before-interpreter | 224,926,298 | 211,623,250 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read | before-tiered | 143,824,866 | 122,350,950 | 0.639 | 24 | 0 | 0 | 1 | 58 | 583,993 | LoopComplete=57, Poll=1 | - | InnerLoop=0, UnsupportedOpcode=3 |
| table/field_read | interpreter | 401,302,411 | 397,500,177 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/field_read | tiered | 25,319,444 | 11,252,431 | 0.063 | 23 | 0 | 0 | 1 | 2 | 489,737 | LoopComplete=1, Poll=1 | - | - |
| table/field_read | before-interpreter | 401,298,697 | 397,499,733 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/field_read | before-tiered | 25,233,737 | 11,562,846 | 0.063 | 23 | 0 | 0 | 1 | 2 | 489,836 | LoopComplete=1, Poll=1 | - | - |
| table/metamethod_index | interpreter | 304,590,984 | 300,801,654 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/metamethod_index | tiered | 16,994,667 | 3,955,523 | 0.056 | 25 | 0 | 0 | 1 | 1 | 194,951 | LoopComplete=1 | - | - |
| table/metamethod_index | before-interpreter | 304,587,593 | 300,800,892 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/metamethod_index | before-tiered | 15,944,344 | 4,699,570 | 0.052 | 25 | 0 | 0 | 1 | 1 | 195,644 | LoopComplete=1 | - | - |
| table/iteration | interpreter | 508,406,076 | 500,207,230 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/iteration | tiered | 50,264,743 | 19,267,045 | 0.099 | 23 | 0 | 0 | 1 | 77 | 767,701 | GuardFail=77 | ArrayBounds=77 | ShortLoop=3, UnsupportedOperand=1 |
| table/iteration | before-interpreter | 508,402,716 | 500,206,688 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/iteration | before-tiered | 36,232,962 | 23,725,260 | 0.071 | 23 | 0 | 0 | 1 | 80 | 790,630 | GuardFail=80 | ArrayBounds=80 | ShortLoop=3, UnsupportedOperand=1 |
| allocation/table_churn | interpreter | 68,560,424 | 64,769,288 | 1.000 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | - |
| allocation/table_churn | tiered | 68,288,851 | 64,399,403 | 0.996 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | Divergence=3 |
| allocation/table_churn | before-interpreter | 68,555,869 | 64,770,461 | 1.000 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | - |
| allocation/table_churn | before-tiered | 68,277,690 | 64,397,617 | 0.996 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| allocation/gc_cycles | interpreter | 144,933,220 | 139,896,959 | 1.000 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/gc_cycles | tiered | 144,951,863 | 139,809,883 | 1.000 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | VirtualEscape=3 |
| allocation/gc_cycles | before-interpreter | 144,928,963 | 139,897,425 | 1.000 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/gc_cycles | before-tiered | 144,939,270 | 139,808,250 | 1.000 | 80,042 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| string/naive_concat | interpreter | 151,491,010 | 147,658,564 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| string/naive_concat | tiered | 151,431,890 | 147,488,242 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| string/naive_concat | before-interpreter | 151,486,896 | 147,658,490 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| string/naive_concat | before-tiered | 151,436,924 | 147,496,049 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| string/table_concat | interpreter | 183,790,268 | 179,835,421 | 1.000 | 102 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| string/table_concat | tiered | 183,439,832 | 179,388,529 | 0.998 | 102 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| string/table_concat | before-interpreter | 183,787,255 | 179,834,282 | 1.000 | 102 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| string/table_concat | before-tiered | 183,433,912 | 179,387,640 | 0.998 | 102 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| string/plain_find | interpreter | 31,733,692 | 27,959,387 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| string/plain_find | tiered | 31,786,885 | 27,902,325 | 1.002 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| string/plain_find | before-interpreter | 31,729,368 | 27,959,880 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| string/plain_find | before-tiered | 31,782,383 | 27,902,539 | 1.002 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| string/pattern_gsub | interpreter | 76,593,085 | 72,814,405 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| string/pattern_gsub | tiered | 76,707,584 | 72,832,254 | 1.001 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| string/pattern_gsub | before-interpreter | 76,588,490 | 72,815,014 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| string/pattern_gsub | before-tiered | 76,703,492 | 72,832,813 | 1.001 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| string/format | interpreter | 241,104,556 | 238,366,015 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| string/format | tiered | 240,990,948 | 238,137,057 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| string/format | before-interpreter | 241,099,934 | 238,366,352 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| string/format | before-tiered | 240,985,746 | 238,137,632 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| stdlib/math | interpreter | 440,905,185 | 437,098,234 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| stdlib/math | tiered | 60,584,735 | 47,093,588 | 0.137 | 22 | 0 | 0 | 1 | 1 | 245,479 | LoopComplete=1 | - | - |
| stdlib/math | before-interpreter | 440,901,443 | 437,097,282 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| stdlib/math | before-tiered | 60,214,801 | 47,403,233 | 0.137 | 22 | 0 | 0 | 1 | 1 | 245,710 | LoopComplete=1 | - | - |
| stdlib/native_calls | interpreter | 149,268,833 | 145,399,780 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| stdlib/native_calls | tiered | 149,259,573 | 145,259,803 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| stdlib/native_calls | before-interpreter | 149,264,818 | 145,398,968 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| stdlib/native_calls | before-tiered | 149,254,522 | 145,259,824 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOperand=3 |
| stdlib/sort | interpreter | 76,792,453 | 72,908,592 | 1.000 | 34 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| stdlib/sort | tiered | 70,065,228 | 62,229,771 | 0.912 | 34 | 0 | 0 | 1 | 11 | 87,972 | LoopComplete=11 | - | - |
| stdlib/sort | before-interpreter | 76,788,051 | 72,908,786 | 1.000 | 34 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| stdlib/sort | before-tiered | 70,059,141 | 62,228,427 | 0.912 | 34 | 0 | 0 | 1 | 11 | 87,972 | LoopComplete=11 | - | - |
| algorithm/sieve | interpreter | 276,522,441 | 271,934,401 | 1.000 | 27 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/sieve | tiered | 136,383,074 | 112,292,458 | 0.493 | 27 | 0 | 0 | 2 | 2,562 | 703,796 | GuardFail=2317, LoopComplete=245 | ArrayBounds=2129, Branch=188 | - |
| algorithm/sieve | before-interpreter | 276,554,847 | 271,898,110 | 1.000 | 27 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/sieve | before-tiered | 136,795,798 | 102,360,593 | 0.495 | 27 | 0 | 0 | 2 | 3,918 | 703,779 | GuardFail=3673, LoopComplete=245 | ArrayBounds=3485, Branch=188 | - |
| algorithm/nbody | interpreter | 76,709,319 | 71,061,275 | 1.000 | 28 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/nbody | tiered | 110,387,476 | 38,588,567 | 1.439 | 28 | 0 | 0 | 2 | 2,621 | 5,244 | LoopComplete=2621 | - | InnerLoop=6, LoopNotClosed=2 |
| algorithm/nbody | before-interpreter | 76,704,921 | 71,062,036 | 1.000 | 28 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/nbody | before-tiered | 108,754,973 | 43,911,054 | 1.418 | 28 | 0 | 0 | 2 | 2,646 | 5,289 | LoopComplete=2646 | - | InnerLoop=6, LoopNotClosed=2 |
| algorithm/mandelbrot | interpreter | 798,500,691 | 794,366,912 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/mandelbrot | tiered | 87,871,057 | 28,108,421 | 0.110 | 22 | 0 | 0 | 1 | 2,991 | 810,285 | GuardFail=16, LoopComplete=2627 | Branch=16 | InnerLoop=3 |
| algorithm/mandelbrot | before-interpreter | 798,496,944 | 794,366,532 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/mandelbrot | before-tiered | 87,905,009 | 27,826,000 | 0.110 | 22 | 0 | 0 | 1 | 3,137 | 810,309 | GuardFail=16, LoopComplete=3119, Poll=0 | Branch=16 | InnerLoop=3 |
| algorithm/spectral_norm | interpreter | 763,114,223 | 758,313,193 | 1.000 | 44 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/spectral_norm | tiered | 100,884,586 | 61,616,072 | 0.132 | 44 | 0 | 0 | 2 | 4,871 | 547,388 | GuardFail=4615, LoopComplete=272 | ArrayBounds=4615 | InnerLoop=2, ShortLoop=4 |
| algorithm/spectral_norm | before-interpreter | 763,110,241 | 758,312,548 | 1.000 | 44 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/spectral_norm | before-tiered | 98,195,294 | 66,696,077 | 0.129 | 44 | 0 | 0 | 2 | 4,882 | 551,410 | GuardFail=4650, LoopComplete=276 | ArrayBounds=4650 | InnerLoop=2, ShortLoop=4 |
| algorithm/binary_trees | interpreter | 323,383,716 | 327,147,646 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/binary_trees | tiered | 323,266,648 | 327,094,476 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/binary_trees | before-interpreter | 323,379,163 | 327,148,087 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/binary_trees | before-tiered | 323,263,695 | 327,093,582 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/captured_call | interpreter | 85,981,700 | 77,829,582 | 1.000 | 31 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/captured_call | tiered | 48,388,822 | 7,552,387 | 0.563 | 31 | 0 | 0 | 1 | 1 | 3,515 | LoopComplete=1 | - | - |
| mineclonia/captured_call | before-interpreter | 85,977,664 | 77,829,241 | 1.000 | 31 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/captured_call | before-tiered | 48,335,762 | 1,516,876 | 0.562 | 31 | 0 | 0 | 1 | 1 | 3,515 | LoopComplete=1 | - | - |
| mineclonia/module_calls | interpreter | 268,629,488 | 262,557,987 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/module_calls | tiered | 148,274,413 | -24,557,065 | 0.552 | 25 | 0 | 0 | 1 | 1 | 6,304 | LoopComplete=1 | - | - |
| mineclonia/module_calls | before-interpreter | 268,625,219 | 262,558,799 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/module_calls | before-tiered | 116,300,893 | 29,218,689 | 0.433 | 25 | 0 | 0 | 1 | 1 | 7,558 | LoopComplete=1 | - | - |
| mineclonia/open_results | interpreter | 179,118,471 | 174,486,567 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/open_results | tiered | 63,572,426 | 14,112,626 | 0.355 | 23 | 0 | 0 | 2 | 1,184 | 23,651 | CommittedEffect=592, LoopComplete=592 | Branch=592 | - |
| mineclonia/open_results | before-interpreter | 179,115,254 | 174,486,716 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/open_results | before-tiered | 65,409,440 | 11,053,393 | 0.365 | 23 | 0 | 0 | 2 | 1,166 | 23,306 | CommittedEffect=583, LoopComplete=583 | Branch=583 | - |
| mineclonia/ull_kernel | interpreter | 146,059,793 | 138,093,563 | 1.000 | 32 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/ull_kernel | tiered | 149,384,075 | 3,264,899 | 1.023 | 32 | 0 | 0 | 1 | 1 | 1,327 | LoopComplete=1 | - | - |
| mineclonia/ull_kernel | before-interpreter | 146,055,011 | 138,094,861 | 1.000 | 32 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/ull_kernel | before-tiered | 156,098,714 | -2,345,658 | 1.069 | 32 | 0 | 0 | 1 | 1 | 1,129 | LoopComplete=1 | - | - |
| mineclonia/ull_arith | interpreter | 142,300,602 | 137,799,585 | 1.000 | 30 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/ull_arith | tiered | 36,483,257 | 13,860,624 | 0.256 | 30 | 0 | 0 | 1 | 1 | 41,618 | LoopComplete=1 | - | - |
| mineclonia/ull_arith | before-interpreter | 142,296,840 | 137,799,738 | 1.000 | 30 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/ull_arith | before-tiered | 45,936,680 | 5,107,047 | 0.323 | 30 | 0 | 0 | 1 | 1 | 38,087 | LoopComplete=1 | - | - |
| mineclonia/bit_wide | interpreter | 205,516,115 | 200,550,898 | 1.000 | 29 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/bit_wide | tiered | 51,138,309 | 3,031,232 | 0.249 | 29 | 0 | 0 | 1 | 1 | 40,199 | LoopComplete=1 | - | - |
| mineclonia/bit_wide | before-interpreter | 205,513,217 | 200,549,834 | 1.000 | 29 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/bit_wide | before-tiered | 51,391,019 | 2,636,797 | 0.250 | 29 | 0 | 0 | 1 | 1 | 40,133 | LoopComplete=1 | - | - |
| table/hash_read_small | interpreter | 361,176,231 | 357,280,523 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_small | tiered | 239,936,510 | 198,508,744 | 0.664 | 24 | 0 | 0 | 1 | 2,961 | 939,868 | LoopComplete=2958, Poll=3 | - | InnerLoop=3 |
| table/hash_read_small | before-interpreter | 361,172,137 | 357,280,385 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_small | before-tiered | 233,893,788 | 210,726,929 | 0.648 | 24 | 0 | 0 | 1 | 1,922 | 939,847 | LoopComplete=1919, Poll=3 | - | InnerLoop=3 |
| table/hash_read_shaped | interpreter | 366,123,185 | 362,160,444 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_shaped | tiered | 216,769,694 | 201,297,400 | 0.592 | 24 | 0 | 0 | 1 | 20 | 940,523 | LoopComplete=17, Poll=3 | - | LoopNotClosed=1 |
| table/hash_read_shaped | before-interpreter | 366,119,646 | 362,159,484 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_shaped | before-tiered | 216,832,807 | 201,234,015 | 0.592 | 24 | 0 | 0 | 1 | 20 | 940,406 | LoopComplete=17, Poll=3 | - | InnerLoop=0, LoopNotClosed=1 |
| table/hash_read_indexed | interpreter | 298,319,878 | 270,899,221 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_indexed | tiered | 215,001,164 | 156,190,740 | 0.721 | 24 | 0 | 0 | 1 | 2,985 | 644,984 | GuardFail=2966, LoopComplete=17, Poll=2 | ArrayBounds=2966 | UnsupportedOpcode=6 |
| table/hash_read_indexed | before-interpreter | 298,316,218 | 270,899,032 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_indexed | before-tiered | 215,575,257 | 156,328,429 | 0.723 | 24 | 0 | 0 | 1 | 3,134 | 644,376 | GuardFail=3115, LoopComplete=17, Poll=2 | ArrayBounds=3115 | UnsupportedOpcode=6 |
| table/heap_redundancy | interpreter | 408,030,811 | 403,950,142 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/heap_redundancy | tiered | 20,447,617 | 7,433,875 | 0.050 | 24 | 0 | 0 | 1 | 1 | 147,492 | LoopComplete=1 | - | - |
| table/heap_redundancy | before-interpreter | 408,027,144 | 403,949,655 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/heap_redundancy | before-tiered | 20,703,774 | 6,910,511 | 0.051 | 24 | 0 | 0 | 1 | 1 | 147,393 | LoopComplete=1 | - | - |
| allocation/ctor_array | interpreter | 380,179,104 | 376,554,795 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_array | tiered | 12,307,195 | 1,946,748 | 0.032 | 1,935 | 148,087 | 31,986,792 | 1 | 1 | 148,086 | LoopComplete=1 | - | - |
| allocation/ctor_array | before-interpreter | 380,175,034 | 376,555,207 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_array | before-tiered | 379,837,716 | 376,130,839 | 0.999 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| allocation/ctor_fields | interpreter | 313,049,980 | 309,209,855 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_fields | tiered | 13,890,769 | 2,360,345 | 0.044 | 2,793 | 147,229 | 31,801,464 | 1 | 1 | 147,228 | LoopComplete=1 | - | - |
| allocation/ctor_fields | before-interpreter | 313,045,109 | 309,210,966 | 1.000 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_fields | before-tiered | 312,717,431 | 308,796,469 | 0.999 | 150,022 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| allocation/ctor_mixed | interpreter | 357,028,721 | 353,218,179 | 1.000 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_mixed | tiered | 12,795,264 | 1,894,881 | 0.036 | 1,704 | 118,318 | 25,556,688 | 1 | 1 | 118,317 | LoopComplete=1 | - | - |
| allocation/ctor_mixed | before-interpreter | 357,056,773 | 353,186,075 | 1.000 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_mixed | before-tiered | 355,934,580 | 351,975,313 | 0.997 | 120,022 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |
| allocation/ctor_side_exit | interpreter | 346,931,791 | 353,262,733 | 1.000 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_side_exit | tiered | 40,650,105 | 49,071,300 | 0.117 | 3,619 | 116,404 | 25,143,264 | 1 | 2,376 | 116,404 | GuardFail=2376 | Branch=2376 | UnsupportedOperand=3 |
| allocation/ctor_side_exit | before-interpreter | 346,929,664 | 353,409,822 | 1.000 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| allocation/ctor_side_exit | before-tiered | 346,616,445 | 353,519,505 | 0.999 | 120,023 | 0 | 0 | 0 | 0 | 0 | - | - | UnsupportedOpcode=3 |

| Benchmark | Mode | before | after | change |
| :-- | :-- | --: | --: | --: |
| numeric/for_sum | interpreter | 197,686,621 | 197,690,831 | +0.00% |
| numeric/for_sum | tiered | 33,783,999 | 34,560,648 | +2.30% |
| numeric/while_mix | interpreter | 656,549,712 | 656,552,472 | +0.00% |
| numeric/while_mix | tiered | 39,817,166 | 39,639,012 | -0.45% |
| control/branching | interpreter | 527,803,418 | 527,807,315 | +0.00% |
| control/branching | tiered | 48,845,111 | 48,809,275 | -0.07% |
| control/nested_loops | interpreter | 171,077,022 | 171,081,717 | +0.00% |
| control/nested_loops | tiered | 26,674,967 | 27,254,887 | +2.17% |
| control/nested_calls | interpreter | 3,437,855,054 | 3,437,858,485 | +0.00% |
| control/nested_calls | tiered | 199,827,093 | 198,296,900 | -0.77% |
| calls/function | interpreter | 322,162,053 | 322,166,518 | +0.00% |
| calls/function | tiered | 18,528,289 | 18,587,641 | +0.32% |
| calls/builtin | interpreter | 64,274,461 | 64,279,110 | +0.01% |
| calls/builtin | tiered | 20,210,039 | 20,434,511 | +1.11% |
| calls/method | interpreter | 350,570,310 | 350,574,717 | +0.00% |
| calls/method | tiered | 17,431,104 | 17,299,571 | -0.75% |
| calls/closure | interpreter | 304,414,896 | 304,419,904 | +0.00% |
| calls/closure | tiered | 17,922,525 | 17,993,590 | +0.40% |
| calls/recursive_fib | interpreter | 209,671,816 | 209,676,128 | +0.00% |
| calls/recursive_fib | tiered | 49,198,742 | 49,202,831 | +0.01% |
| calls/tail_call | interpreter | 283,479,780 | 283,483,439 | +0.00% |
| calls/tail_call | tiered | 76,121,332 | 76,125,433 | +0.01% |
| calls/varargs | interpreter | 378,813,960 | 378,817,953 | +0.00% |
| calls/varargs | tiered | 17,355,697 | 17,309,504 | -0.27% |
| calls/varargs_forward | interpreter | 463,387,073 | 463,390,947 | +0.00% |
| calls/varargs_forward | tiered | 16,301,468 | 15,671,129 | -3.87% |
| table/array_read | interpreter | 327,965,112 | 327,968,362 | +0.00% |
| table/array_read | tiered | 70,564,336 | 69,848,149 | -1.01% |
| table/array_write | interpreter | 560,617,802 | 560,622,656 | +0.00% |
| table/array_write | tiered | 57,914,763 | 58,020,619 | +0.18% |
| table/hash_read | interpreter | 224,926,298 | 224,930,784 | +0.00% |
| table/hash_read | tiered | 143,824,866 | 148,553,156 | +3.29% |
| table/field_read | interpreter | 401,298,697 | 401,302,411 | +0.00% |
| table/field_read | tiered | 25,233,737 | 25,319,444 | +0.34% |
| table/metamethod_index | interpreter | 304,587,593 | 304,590,984 | +0.00% |
| table/metamethod_index | tiered | 15,944,344 | 16,994,667 | +6.59% |
| table/iteration | interpreter | 508,402,716 | 508,406,076 | +0.00% |
| table/iteration | tiered | 36,232,962 | 50,264,743 | +38.73% |
| allocation/table_churn | interpreter | 68,555,869 | 68,560,424 | +0.01% |
| allocation/table_churn | tiered | 68,277,690 | 68,288,851 | +0.02% |
| allocation/gc_cycles | interpreter | 144,928,963 | 144,933,220 | +0.00% |
| allocation/gc_cycles | tiered | 144,939,270 | 144,951,863 | +0.01% |
| string/naive_concat | interpreter | 151,486,896 | 151,491,010 | +0.00% |
| string/naive_concat | tiered | 151,436,924 | 151,431,890 | -0.00% |
| string/table_concat | interpreter | 183,787,255 | 183,790,268 | +0.00% |
| string/table_concat | tiered | 183,433,912 | 183,439,832 | +0.00% |
| string/plain_find | interpreter | 31,729,368 | 31,733,692 | +0.01% |
| string/plain_find | tiered | 31,782,383 | 31,786,885 | +0.01% |
| string/pattern_gsub | interpreter | 76,588,490 | 76,593,085 | +0.01% |
| string/pattern_gsub | tiered | 76,703,492 | 76,707,584 | +0.01% |
| string/format | interpreter | 241,099,934 | 241,104,556 | +0.00% |
| string/format | tiered | 240,985,746 | 240,990,948 | +0.00% |
| stdlib/math | interpreter | 440,901,443 | 440,905,185 | +0.00% |
| stdlib/math | tiered | 60,214,801 | 60,584,735 | +0.61% |
| stdlib/native_calls | interpreter | 149,264,818 | 149,268,833 | +0.00% |
| stdlib/native_calls | tiered | 149,254,522 | 149,259,573 | +0.00% |
| stdlib/sort | interpreter | 76,788,051 | 76,792,453 | +0.01% |
| stdlib/sort | tiered | 70,059,141 | 70,065,228 | +0.01% |
| algorithm/sieve | interpreter | 276,554,847 | 276,522,441 | -0.01% |
| algorithm/sieve | tiered | 136,795,798 | 136,383,074 | -0.30% |
| algorithm/nbody | interpreter | 76,704,921 | 76,709,319 | +0.01% |
| algorithm/nbody | tiered | 108,754,973 | 110,387,476 | +1.50% |
| algorithm/mandelbrot | interpreter | 798,496,944 | 798,500,691 | +0.00% |
| algorithm/mandelbrot | tiered | 87,905,009 | 87,871,057 | -0.04% |
| algorithm/spectral_norm | interpreter | 763,110,241 | 763,114,223 | +0.00% |
| algorithm/spectral_norm | tiered | 98,195,294 | 100,884,586 | +2.74% |
| algorithm/binary_trees | interpreter | 323,379,163 | 323,383,716 | +0.00% |
| algorithm/binary_trees | tiered | 323,263,695 | 323,266,648 | +0.00% |
| mineclonia/captured_call | interpreter | 85,977,664 | 85,981,700 | +0.00% |
| mineclonia/captured_call | tiered | 48,335,762 | 48,388,822 | +0.11% |
| mineclonia/module_calls | interpreter | 268,625,219 | 268,629,488 | +0.00% |
| mineclonia/module_calls | tiered | 116,300,893 | 148,274,413 | +27.49% |
| mineclonia/open_results | interpreter | 179,115,254 | 179,118,471 | +0.00% |
| mineclonia/open_results | tiered | 65,409,440 | 63,572,426 | -2.81% |
| mineclonia/ull_kernel | interpreter | 146,055,011 | 146,059,793 | +0.00% |
| mineclonia/ull_kernel | tiered | 156,098,714 | 149,384,075 | -4.30% |
| mineclonia/ull_arith | interpreter | 142,296,840 | 142,300,602 | +0.00% |
| mineclonia/ull_arith | tiered | 45,936,680 | 36,483,257 | -20.58% |
| mineclonia/bit_wide | interpreter | 205,513,217 | 205,516,115 | +0.00% |
| mineclonia/bit_wide | tiered | 51,391,019 | 51,138,309 | -0.49% |
| table/hash_read_small | interpreter | 361,172,137 | 361,176,231 | +0.00% |
| table/hash_read_small | tiered | 233,893,788 | 239,936,510 | +2.58% |
| table/hash_read_shaped | interpreter | 366,119,646 | 366,123,185 | +0.00% |
| table/hash_read_shaped | tiered | 216,832,807 | 216,769,694 | -0.03% |
| table/hash_read_indexed | interpreter | 298,316,218 | 298,319,878 | +0.00% |
| table/hash_read_indexed | tiered | 215,575,257 | 215,001,164 | -0.27% |
| table/heap_redundancy | interpreter | 408,027,144 | 408,030,811 | +0.00% |
| table/heap_redundancy | tiered | 20,703,774 | 20,447,617 | -1.24% |
| allocation/ctor_array | interpreter | 380,175,034 | 380,179,104 | +0.00% |
| allocation/ctor_array | tiered | 379,837,716 | 12,307,195 | -96.76% |
| allocation/ctor_fields | interpreter | 313,045,109 | 313,049,980 | +0.00% |
| allocation/ctor_fields | tiered | 312,717,431 | 13,890,769 | -95.56% |
| allocation/ctor_mixed | interpreter | 357,056,773 | 357,028,721 | -0.01% |
| allocation/ctor_mixed | tiered | 355,934,580 | 12,795,264 | -96.41% |
| allocation/ctor_side_exit | interpreter | 346,929,664 | 346,931,791 | +0.00% |
| allocation/ctor_side_exit | tiered | 346,616,445 | 40,650,105 | -88.27% |
