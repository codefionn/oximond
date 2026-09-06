Heap allocations by workload scale, tiered, OXIMOND_JIT_SYNC=1, lua51, alloc-stats builds: before a881ecd 5ad7e636, after 7a2149be (d557b47). One run per point; counts are deterministic.

The rise in the gate is a fixed offset that does not grow with scale, so it is a one-time cost (trace recording, compile and install), not allocation by guest execution.

```
table/hash_read before | s1 count=3303 bytes=746196 peak=440185 | s2 count=3303 bytes=746196 peak=440185 | s4 count=3303 bytes=746199 peak=440185
table/hash_read after | s1 count=6683 bytes=2168972 peak=842474 | s2 count=6687 bytes=2169024 peak=842474 | s4 count=6697 bytes=2169157 peak=842474
table/hash_read_small before | s1 count=961 bytes=251321 peak=73734 | s2 count=975 bytes=255665 peak=73734 | s4 count=975 bytes=255665 peak=73734
table/hash_read_small after | s1 count=4343 bytes=1674363 peak=516956 | s2 count=4351 bytes=1674467 peak=516956 | s4 count=4365 bytes=1674649 peak=516956
table/hash_read_shaped before | s1 count=1062 bytes=261250 peak=75343 | s2 count=1070 bytes=263438 peak=75343 | s4 count=1077 bytes=265610 peak=75343
table/hash_read_shaped after | s1 count=4450 bytes=1688372 peak=517728 | s2 count=4459 bytes=1688492 peak=517728 | s4 count=4473 bytes=1688674 peak=517728
table/hash_read_indexed before | s1 count=8910 bytes=1256418 peak=702410 | s2 count=8910 bytes=1256418 peak=702410 | s4 count=8910 bytes=1256421 peak=702410
table/hash_read_indexed after | s1 count=15256 bytes=4284563 peak=1435446 | s2 count=15260 bytes=4284615 peak=1435446 | s4 count=15270 bytes=4284748 peak=1435446
allocation/table_churn before | s1 count=876 bytes=187083 peak=65155 | s2 count=915 bytes=189918 peak=65155 | s4 count=993 bytes=195534 peak=65155
allocation/table_churn after | s1 count=1785 bytes=531541 peak=221507 | s2 count=1785 bytes=531544 peak=221507 | s4 count=1800 bytes=532936 peak=221507
```
