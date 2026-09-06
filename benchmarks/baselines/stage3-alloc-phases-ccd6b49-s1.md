Load 19:27:39 up 1 day, 21:57,  1 user,  load average: 6,40, 7,76, 8,69; cpu 3 (sibling 11); oximond sha256 47ed62fa1f71f81e366d32e61f487701183e8ab03d7906bb3d00eead43b89e1a

| Benchmark | Mode | instructions:u | vs interpreter | tables | elided | elided bytes | traces | entries | iterations | exits | side exits | aborts | loads_forwarded | loads_merged | loads_hoisted | alloc_count | alloc_bytes | alloc_peak_bytes |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: |
| table/hash_read | interpreter | 227,035,619 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 3,256 | 736,574 | 439,730 |
| table/hash_read | tiered | 141,861,545 | 0.625 | 24 | 0 | 0 | 1 | 19 | 599,705 | LoopComplete=17, Poll=2 | - | UnsupportedOpcode=3 | 0 | 0 | 0 | 6,687 | 2,169,689 | 842,550 |
| table/hash_read | before-interpreter | 227,034,848 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 3,256 | 736,549 | 439,725 |
| table/hash_read | before-tiered | 141,862,271 | 0.625 | 24 | 0 | 0 | 1 | 19 | 599,705 | LoopComplete=17, Poll=2 | - | UnsupportedOpcode=3 | 0 | 0 | 0 | 6,687 | 2,169,664 | 842,545 |
| allocation/table_churn | interpreter | 68,809,306 | 1.000 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 948 | 189,923 | 65,159 |
| allocation/table_churn | tiered | 6,351,719 | 0.092 | 23 | 79,999 | 17,279,784 | 1 | 1 | 79,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 1,787 | 532,181 | 221,583 |
| allocation/table_churn | before-interpreter | 68,808,876 | 1.000 | 22 | 80,000 | 17,280,000 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 948 | 189,898 | 65,159 |
| allocation/table_churn | before-tiered | 6,351,083 | 0.092 | 23 | 79,999 | 17,279,784 | 1 | 1 | 79,934 | LoopComplete=1 | - | - | 0 | 0 | 0 | 1,787 | 532,156 | 221,578 |
| table/hash_read_small | interpreter | 361,179,042 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 938 | 246,559 | 73,738 |
| table/hash_read_small | tiered | 223,531,905 | 0.619 | 24 | 0 | 0 | 1 | 20 | 959,917 | LoopComplete=17, Poll=3 | - | - | 0 | 0 | 0 | 4,347 | 1,675,056 | 517,032 |
| table/hash_read_small | before-interpreter | 361,180,803 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 938 | 246,534 | 73,738 |
| table/hash_read_small | before-tiered | 223,530,971 | 0.619 | 24 | 0 | 0 | 1 | 20 | 959,917 | LoopComplete=17, Poll=3 | - | - | 0 | 0 | 0 | 4,347 | 1,675,031 | 517,027 |
| table/hash_read_shaped | interpreter | 366,127,390 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,032 | 254,316 | 75,347 |
| table/hash_read_shaped | tiered | 213,271,378 | 0.583 | 24 | 0 | 0 | 1 | 20 | 959,884 | LoopComplete=17, Poll=3 | - | LoopNotClosed=1 | 0 | 0 | 0 | 4,454 | 1,689,073 | 517,804 |
| table/hash_read_shaped | before-interpreter | 366,127,518 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,032 | 254,291 | 75,347 |
| table/hash_read_shaped | before-tiered | 213,271,039 | 0.583 | 24 | 0 | 0 | 1 | 20 | 959,884 | LoopComplete=17, Poll=3 | - | LoopNotClosed=1 | 0 | 0 | 0 | 4,454 | 1,689,048 | 517,799 |
| table/hash_read_indexed | interpreter | 299,434,718 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 8,827 | 1,234,268 | 700,931 |
| table/hash_read_indexed | tiered | 202,173,773 | 0.675 | 24 | 0 | 0 | 1 | 35 | 653,774 | GuardFail=16, LoopComplete=17, Poll=2 | ArrayBounds=16 | UnsupportedOpcode=6 | 0 | 0 | 0 | 15,262 | 4,285,357 | 1,435,522 |
| table/hash_read_indexed | before-interpreter | 299,439,300 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 8,827 | 1,234,243 | 700,926 |
| table/hash_read_indexed | before-tiered | 202,173,147 | 0.675 | 24 | 0 | 0 | 1 | 35 | 653,774 | GuardFail=16, LoopComplete=17, Poll=2 | ArrayBounds=16 | UnsupportedOpcode=6 | 0 | 0 | 0 | 15,262 | 4,285,332 | 1,435,517 |

Oracles:

- checked against lua5.1: table/hash_read, allocation/table_churn, table/hash_read_small, table/hash_read_shaped, table/hash_read_indexed

| Benchmark | Mode | before | after | change |
| :-- | :-- | --: | --: | --: |
| table/hash_read | interpreter | 227,034,848 | 227,035,619 | +0.00% |
| table/hash_read | tiered | 141,862,271 | 141,861,545 | -0.00% |
| allocation/table_churn | interpreter | 68,808,876 | 68,809,306 | +0.00% |
| allocation/table_churn | tiered | 6,351,083 | 6,351,719 | +0.01% |
| table/hash_read_small | interpreter | 361,180,803 | 361,179,042 | -0.00% |
| table/hash_read_small | tiered | 223,530,971 | 223,531,905 | +0.00% |
| table/hash_read_shaped | interpreter | 366,127,518 | 366,127,390 | -0.00% |
| table/hash_read_shaped | tiered | 213,271,039 | 213,271,378 | +0.00% |
| table/hash_read_indexed | interpreter | 299,439,300 | 299,434,718 | -0.00% |
| table/hash_read_indexed | tiered | 202,173,147 | 202,173,773 | +0.00% |

## Allocation phases (alloc-phases diagnostic builds)

### table/hash_read interpreter (alloc-phases diagnostic build)

Totals: 3,257 allocations, 736,631 bytes, global peak 439,731 bytes, live at the snapshot 381,459 bytes. Header overhead 52,112 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 849 | 30 | 227,811 | 73,129 | 73,009 | 51,098 | 174,069 | 2,644 |
| recording | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| optimization | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| codegen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| installation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| execution | 2,405 | 0 | 508,471 | 439,731 | 385,989 | 330,309 | 178,162 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 3 | 0 | 349 | 381,703 | 296 | 52 | 297 | 0 |

### table/hash_read tiered (alloc-phases diagnostic build)

Totals: 6,689 allocations, 2,170,338 bytes, global peak 842,551 bytes, live at the snapshot 419,141 bytes. Header overhead 115,184 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 849 | 30 | 227,791 | 73,124 | 73,004 | 51,098 | 174,054 | 2,639 |
| recording | 337 | 0 | 84,069 | 448,574 | 42,624 | 15,228 | 62,645 | 6,196 |
| optimization | 1,629 | 0 | 359,626 | 458,795 | 35,929 | 7,988 | 346,970 | 4,668 |
| codegen | 1,321 | 0 | 965,490 | 842,551 | 403,036 | 8,065 | 949,936 | 7,489 |
| installation | 97 | 0 | 22,046 | 447,299 | 7,994 | 5,200 | 16,846 | 0 |
| execution | 2,452 | 0 | 510,375 | 440,271 | 386,370 | 331,510 | 178,865 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 4 | 0 | 941 | 419,681 | 592 | 52 | 889 | 0 |

### allocation/table_churn interpreter (alloc-phases diagnostic build)

Totals: 949 allocations, 189,980 bytes, global peak 65,159 bytes, live at the snapshot 54,313 bytes. Header overhead 15,184 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 727 | 30 | 166,904 | 65,159 | 65,032 | 50,810 | 114,255 | 1,839 |
| recording | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| optimization | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| codegen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| installation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| execution | 219 | 0 | 22,727 | 61,732 | 9,083 | 3,451 | 19,276 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 3 | 0 | 349 | 54,557 | 296 | 52 | 297 | 0 |

### allocation/table_churn tiered (alloc-phases diagnostic build)

Totals: 1,789 allocations, 532,830 bytes, global peak 221,584 bytes, live at the snapshot 75,009 bytes. Header overhead 32,704 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 727 | 30 | 166,884 | 65,154 | 65,027 | 50,810 | 114,240 | 1,834 |
| recording | 75 | 0 | 25,459 | 74,269 | 12,913 | 7,532 | 17,927 | 0 |
| optimization | 433 | 0 | 98,031 | 85,171 | 16,283 | 3,044 | 94,955 | 32 |
| codegen | 444 | 0 | 224,672 | 221,584 | 145,431 | 7,069 | 217,603 | 0 |
| installation | 23 | 0 | 3,846 | 81,495 | 2,494 | 2,494 | 1,352 | 0 |
| execution | 83 | 0 | 12,997 | 82,183 | 9,400 | 4,008 | 8,989 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 4 | 0 | 941 | 75,549 | 592 | 52 | 889 | 0 |

### table/hash_read_small interpreter (alloc-phases diagnostic build)

Totals: 939 allocations, 246,616 bytes, global peak 73,738 bytes, live at the snapshot 55,607 bytes. Header overhead 15,024 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 856 | 30 | 231,863 | 73,738 | 73,612 | 51,098 | 177,883 | 2,882 |
| recording | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| optimization | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| codegen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| installation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| execution | 80 | 0 | 14,404 | 63,717 | 9,737 | 4,457 | 9,947 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 3 | 0 | 349 | 55,851 | 296 | 52 | 297 | 0 |

### table/hash_read_small tiered (alloc-phases diagnostic build)

Totals: 4,349 allocations, 1,675,705 bytes, global peak 517,033 bytes, live at the snapshot 93,385 bytes. Header overhead 77,744 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 856 | 30 | 231,843 | 73,733 | 73,607 | 51,098 | 177,868 | 2,877 |
| recording | 313 | 0 | 79,425 | 123,056 | 42,720 | 15,324 | 57,905 | 6,196 |
| optimization | 1,629 | 0 | 359,626 | 133,277 | 35,929 | 7,988 | 346,970 | 4,668 |
| codegen | 1,321 | 0 | 965,490 | 517,033 | 403,036 | 8,065 | 949,936 | 7,489 |
| installation | 97 | 0 | 22,046 | 121,781 | 7,994 | 5,200 | 16,846 | 0 |
| execution | 129 | 0 | 16,334 | 101,490 | 10,938 | 5,658 | 10,676 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 4 | 0 | 941 | 93,925 | 592 | 52 | 889 | 0 |

### table/hash_read_shaped interpreter (alloc-phases diagnostic build)

Totals: 1,033 allocations, 254,373 bytes, global peak 75,347 bytes, live at the snapshot 55,273 bytes. Header overhead 16,528 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 948 | 30 | 238,257 | 75,347 | 75,220 | 51,146 | 184,075 | 3,036 |
| recording | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| optimization | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| codegen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| installation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| execution | 82 | 0 | 15,767 | 64,343 | 10,161 | 4,075 | 11,692 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 3 | 0 | 349 | 55,517 | 296 | 52 | 297 | 0 |

### table/hash_read_shaped tiered (alloc-phases diagnostic build)

Totals: 4,456 allocations, 1,689,722 bytes, global peak 517,805 bytes, live at the snapshot 94,003 bytes. Header overhead 79,456 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 948 | 30 | 238,237 | 75,342 | 75,215 | 51,146 | 184,060 | 3,031 |
| recording | 325 | 0 | 85,365 | 123,828 | 43,488 | 16,092 | 63,077 | 6,196 |
| optimization | 1,629 | 0 | 359,626 | 134,049 | 35,929 | 7,988 | 346,970 | 4,668 |
| codegen | 1,321 | 0 | 965,490 | 517,805 | 403,036 | 8,065 | 949,936 | 7,489 |
| installation | 97 | 0 | 22,046 | 122,553 | 7,994 | 5,200 | 16,846 | 0 |
| execution | 132 | 0 | 18,017 | 102,262 | 10,740 | 5,460 | 12,557 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 4 | 0 | 941 | 94,543 | 592 | 52 | 889 | 0 |

### table/hash_read_indexed interpreter (alloc-phases diagnostic build)

Totals: 8,828 allocations, 1,234,325 bytes, global peak 700,932 bytes, live at the snapshot 692,043 bytes. Header overhead 141,248 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 1,148 | 30 | 380,642 | 93,012 | 92,884 | 51,626 | 325,387 | 3,629 |
| recording | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| optimization | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| codegen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| installation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| execution | 7,677 | 0 | 853,334 | 700,932 | 645,677 | 640,365 | 212,969 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 3 | 0 | 349 | 692,287 | 296 | 52 | 297 | 0 |

### table/hash_read_indexed tiered (alloc-phases diagnostic build)

Totals: 15,264 allocations, 4,286,006 bytes, global peak 1,435,523 bytes, live at the snapshot 750,003 bytes. Header overhead 256,464 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 1,148 | 30 | 380,622 | 93,007 | 92,879 | 51,626 | 325,372 | 3,624 |
| recording | 677 | 0 | 170,172 | 788,269 | 61,644 | 21,090 | 132,900 | 16,182 |
| optimization | 3,211 | 0 | 744,724 | 798,226 | 49,555 | 13,214 | 717,420 | 14,090 |
| codegen | 2,283 | 0 | 2,086,137 | 1,435,523 | 657,236 | 13,601 | 2,056,290 | 16,246 |
| installation | 184 | 0 | 46,900 | 791,438 | 13,827 | 8,402 | 38,498 | 0 |
| execution | 7,757 | 0 | 856,510 | 758,887 | 647,330 | 642,018 | 214,492 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 4 | 0 | 941 | 750,543 | 592 | 52 | 889 | 0 |
