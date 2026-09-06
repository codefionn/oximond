Load 19:27:42 up 1 day, 21:57,  1 user,  load average: 6,40, 7,76, 8,69; cpu 3 (sibling 11); oximond sha256 47ed62fa1f71f81e366d32e61f487701183e8ab03d7906bb3d00eead43b89e1a

| Benchmark | Mode | instructions:u | vs interpreter | tables | elided | elided bytes | traces | entries | iterations | exits | side exits | aborts | loads_forwarded | loads_merged | loads_hoisted | alloc_count | alloc_bytes | alloc_peak_bytes |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: |
| table/hash_read | interpreter | 868,812,000 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 3,256 | 736,577 | 439,730 |
| table/hash_read | tiered | 503,815,092 | 0.580 | 24 | 0 | 0 | 1 | 26 | 2,399,705 | LoopComplete=17, Poll=9 | - | UnsupportedOpcode=3 | 0 | 0 | 0 | 6,701 | 2,169,874 | 842,550 |
| allocation/table_churn | interpreter | 263,846,644 | 1.000 | 22 | 320,000 | 69,120,000 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,344 | 218,438 | 65,159 |
| allocation/table_churn | tiered | 9,491,023 | 0.036 | 23 | 319,999 | 69,119,784 | 1 | 2 | 319,934 | LoopComplete=1, Poll=1 | - | - | 0 | 0 | 0 | 1,802 | 533,576 | 221,583 |
| table/hash_read_small | interpreter | 1,433,021,427 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 938 | 246,559 | 73,738 |
| table/hash_read_small | tiered | 858,370,300 | 0.599 | 24 | 0 | 0 | 1 | 31 | 3,839,917 | LoopComplete=17, Poll=14 | - | - | 0 | 0 | 0 | 4,369 | 1,675,342 | 517,032 |
| table/hash_read_shaped | interpreter | 1,452,604,698 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 1,033 | 254,332 | 75,347 |
| table/hash_read_shaped | tiered | 817,031,810 | 0.562 | 24 | 0 | 0 | 1 | 31 | 3,839,884 | LoopComplete=17, Poll=14 | - | LoopNotClosed=1 | 0 | 0 | 0 | 4,477 | 1,689,375 | 517,804 |
| table/hash_read_indexed | interpreter | 1,115,483,034 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 0 | 0 | 8,827 | 1,234,271 | 700,931 |
| table/hash_read_indexed | tiered | 671,446,251 | 0.602 | 24 | 0 | 0 | 1 | 42 | 2,619,854 | GuardFail=16, LoopComplete=17, Poll=9 | ArrayBounds=16 | UnsupportedOpcode=6 | 0 | 0 | 0 | 15,276 | 4,285,542 | 1,435,522 |

Oracles:

- checked against lua5.1: table/hash_read, allocation/table_churn, table/hash_read_small, table/hash_read_shaped, table/hash_read_indexed

## Allocation phases (alloc-phases diagnostic builds)

### table/hash_read interpreter (alloc-phases diagnostic build)

Totals: 3,257 allocations, 736,634 bytes, global peak 439,731 bytes, live at the snapshot 381,459 bytes. Header overhead 52,112 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 849 | 30 | 227,811 | 73,129 | 73,009 | 51,098 | 174,069 | 2,644 |
| recording | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| optimization | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| codegen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| installation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| execution | 2,405 | 0 | 508,474 | 439,731 | 385,989 | 330,309 | 178,165 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 3 | 0 | 349 | 381,703 | 296 | 52 | 297 | 0 |

### table/hash_read tiered (alloc-phases diagnostic build)

Totals: 6,703 allocations, 2,170,523 bytes, global peak 842,551 bytes, live at the snapshot 419,141 bytes. Header overhead 115,408 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 849 | 30 | 227,791 | 73,124 | 73,004 | 51,098 | 174,054 | 2,639 |
| recording | 337 | 0 | 84,069 | 448,574 | 42,624 | 15,228 | 62,645 | 6,196 |
| optimization | 1,629 | 0 | 359,626 | 458,795 | 35,929 | 7,988 | 346,970 | 4,668 |
| codegen | 1,321 | 0 | 965,490 | 842,551 | 403,036 | 8,065 | 949,936 | 7,489 |
| installation | 97 | 0 | 22,046 | 447,299 | 7,994 | 5,200 | 16,846 | 0 |
| execution | 2,466 | 0 | 510,560 | 440,271 | 386,370 | 331,510 | 179,050 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 4 | 0 | 941 | 419,681 | 592 | 52 | 889 | 0 |

### allocation/table_churn interpreter (alloc-phases diagnostic build)

Totals: 1,345 allocations, 218,495 bytes, global peak 65,159 bytes, live at the snapshot 54,313 bytes. Header overhead 21,520 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 727 | 30 | 166,904 | 65,159 | 65,032 | 50,810 | 114,255 | 1,839 |
| recording | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| optimization | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| codegen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| installation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| execution | 615 | 0 | 51,242 | 61,732 | 9,083 | 3,451 | 47,791 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 3 | 0 | 349 | 54,557 | 296 | 52 | 297 | 0 |

### allocation/table_churn tiered (alloc-phases diagnostic build)

Totals: 1,804 allocations, 534,225 bytes, global peak 221,584 bytes, live at the snapshot 75,393 bytes. Header overhead 32,944 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 727 | 30 | 166,884 | 65,154 | 65,027 | 50,810 | 114,240 | 1,834 |
| recording | 75 | 0 | 25,459 | 74,269 | 12,913 | 7,532 | 17,927 | 0 |
| optimization | 433 | 0 | 98,031 | 85,171 | 16,283 | 3,044 | 94,955 | 32 |
| codegen | 444 | 0 | 224,672 | 221,584 | 145,431 | 7,069 | 217,603 | 0 |
| installation | 23 | 0 | 3,846 | 81,495 | 2,494 | 2,494 | 1,352 | 0 |
| execution | 98 | 0 | 14,392 | 82,151 | 9,368 | 4,392 | 10,000 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 4 | 0 | 941 | 75,933 | 592 | 52 | 889 | 0 |

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

Totals: 4,371 allocations, 1,675,991 bytes, global peak 517,033 bytes, live at the snapshot 93,385 bytes. Header overhead 78,096 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 856 | 30 | 231,843 | 73,733 | 73,607 | 51,098 | 177,868 | 2,877 |
| recording | 313 | 0 | 79,425 | 123,056 | 42,720 | 15,324 | 57,905 | 6,196 |
| optimization | 1,629 | 0 | 359,626 | 133,277 | 35,929 | 7,988 | 346,970 | 4,668 |
| codegen | 1,321 | 0 | 965,490 | 517,033 | 403,036 | 8,065 | 949,936 | 7,489 |
| installation | 97 | 0 | 22,046 | 121,781 | 7,994 | 5,200 | 16,846 | 0 |
| execution | 151 | 0 | 16,620 | 101,490 | 10,938 | 5,658 | 10,962 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 4 | 0 | 941 | 93,925 | 592 | 52 | 889 | 0 |

### table/hash_read_shaped interpreter (alloc-phases diagnostic build)

Totals: 1,034 allocations, 254,389 bytes, global peak 75,347 bytes, live at the snapshot 55,273 bytes. Header overhead 16,544 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 948 | 30 | 238,257 | 75,347 | 75,220 | 51,146 | 184,075 | 3,036 |
| recording | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| optimization | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| codegen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| installation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| execution | 83 | 0 | 15,783 | 64,343 | 10,161 | 4,075 | 11,708 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 3 | 0 | 349 | 55,517 | 296 | 52 | 297 | 0 |

### table/hash_read_shaped tiered (alloc-phases diagnostic build)

Totals: 4,479 allocations, 1,690,024 bytes, global peak 517,805 bytes, live at the snapshot 94,003 bytes. Header overhead 79,824 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 948 | 30 | 238,237 | 75,342 | 75,215 | 51,146 | 184,060 | 3,031 |
| recording | 325 | 0 | 85,365 | 123,828 | 43,488 | 16,092 | 63,077 | 6,196 |
| optimization | 1,629 | 0 | 359,626 | 134,049 | 35,929 | 7,988 | 346,970 | 4,668 |
| codegen | 1,321 | 0 | 965,490 | 517,805 | 403,036 | 8,065 | 949,936 | 7,489 |
| installation | 97 | 0 | 22,046 | 122,553 | 7,994 | 5,200 | 16,846 | 0 |
| execution | 155 | 0 | 18,319 | 102,262 | 10,740 | 5,460 | 12,859 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 4 | 0 | 941 | 94,543 | 592 | 52 | 889 | 0 |

### table/hash_read_indexed interpreter (alloc-phases diagnostic build)

Totals: 8,828 allocations, 1,234,328 bytes, global peak 700,932 bytes, live at the snapshot 692,043 bytes. Header overhead 141,248 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 1,148 | 30 | 380,642 | 93,012 | 92,884 | 51,626 | 325,387 | 3,629 |
| recording | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| optimization | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| codegen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| installation | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| execution | 7,677 | 0 | 853,337 | 700,932 | 645,677 | 640,365 | 212,972 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 3 | 0 | 349 | 692,287 | 296 | 52 | 297 | 0 |

### table/hash_read_indexed tiered (alloc-phases diagnostic build)

Totals: 15,278 allocations, 4,286,191 bytes, global peak 1,435,523 bytes, live at the snapshot 750,003 bytes. Header overhead 256,688 bytes, not counted.

Peaks are per phase and do not sum to the global peak.

| Phase | allocations | other threads | bytes | active peak | origin peak | origin live at snapshot | freed by itself | freed by other phases |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| unclassified | 1,148 | 30 | 380,622 | 93,007 | 92,879 | 51,626 | 325,372 | 3,624 |
| recording | 677 | 0 | 170,172 | 788,269 | 61,644 | 21,090 | 132,900 | 16,182 |
| optimization | 3,211 | 0 | 744,724 | 798,226 | 49,555 | 13,214 | 717,420 | 14,090 |
| codegen | 2,283 | 0 | 2,086,137 | 1,435,523 | 657,236 | 13,601 | 2,056,290 | 16,246 |
| installation | 184 | 0 | 46,900 | 791,438 | 13,827 | 8,402 | 38,498 | 0 |
| execution | 7,771 | 0 | 856,695 | 758,887 | 647,330 | 642,018 | 214,677 | 0 |
| exit_recovery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| report | 4 | 0 | 941 | 750,543 | 592 | 52 | 889 | 0 |
