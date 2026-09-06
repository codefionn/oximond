Load 22:42:55 up 1 day,  1:13,  1 user,  load average: 9,16, 13,96, 12,00; cpu 5 (sibling 13); oximond sha256 830d2ac50f98d7fb71e0cdf00eda8f7764dcb3ac319e97ec8317e5d0ee6f10fe

| Benchmark | Mode | instructions:u | vs interpreter | tables | elided | elided bytes | traces | entries | iterations | exits | side exits | aborts |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- | :-- |
| table/array_read | interpreter | 327,966,354 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/array_read | tiered | 60,667,108 | 0.185 | 23 | 0 | 0 | 1 | 21 | 1,599,263 | LoopComplete=17, Poll=4 | - | ShortLoop=3 |
| table/array_read | before-interpreter | 327,959,287 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/array_read | before-tiered | 65,154,160 | 0.199 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/array_write | interpreter | 560,619,842 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/array_write | tiered | 54,816,372 | 0.098 | 23 | 0 | 0 | 1 | 4 | 799,934 | LoopComplete=1, Poll=3 | - | ShortLoop=3 |
| table/array_write | before-interpreter | 560,613,389 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/array_write | before-tiered | 57,475,717 | 0.103 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read | interpreter | 224,927,493 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read | tiered | 141,808,344 | 0.630 | 24 | 0 | 0 | 1 | 19 | 599,705 | LoopComplete=17, Poll=2 | - | UnsupportedOpcode=3 |
| table/hash_read | before-interpreter | 224,925,151 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read | before-tiered | 223,353,288 | 0.993 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/field_read | interpreter | 401,299,488 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/field_read | tiered | 17,290,738 | 0.043 | 23 | 0 | 0 | 1 | 2 | 499,934 | LoopComplete=1, Poll=1 | - | - |
| table/field_read | before-interpreter | 401,293,672 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/field_read | before-tiered | 23,991,287 | 0.060 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/metamethod_index | interpreter | 304,590,536 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/metamethod_index | tiered | 9,441,783 | 0.031 | 25 | 0 | 0 | 1 | 1 | 199,934 | LoopComplete=1 | - | - |
| table/metamethod_index | before-interpreter | 304,582,458 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/metamethod_index | before-tiered | 17,564,884 | 0.058 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/iteration | interpreter | 508,403,905 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/iteration | tiered | 32,072,174 | 0.063 | 23 | 0 | 0 | 1 | 80 | 797,395 | GuardFail=80 | ArrayBounds=80 | ShortLoop=3, UnsupportedOperand=1 |
| table/iteration | before-interpreter | 508,397,193 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/iteration | before-tiered | 42,765,827 | 0.084 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/sieve | interpreter | 276,557,071 | 1.000 | 27 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/sieve | tiered | 126,597,647 | 0.458 | 27 | 0 | 0 | 2 | 469 | 715,775 | GuardFail=224, LoopComplete=245 | ArrayBounds=36, Branch=188 | - |
| algorithm/sieve | before-interpreter | 276,550,853 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/sieve | before-tiered | 136,362,836 | 0.493 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/nbody | interpreter | 76,705,873 | 1.000 | 28 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/nbody | tiered | 74,289,362 | 0.968 | 28 | 0 | 0 | 2 | 1,977 | 12,730 | LoopComplete=1977 | - | InnerLoop=6, LoopNotClosed=2 |
| algorithm/nbody | before-interpreter | 76,698,392 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/nbody | before-tiered | 109,635,374 | 1.429 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/mandelbrot | interpreter | 798,497,663 | 1.000 | 22 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/mandelbrot | tiered | 61,656,925 | 0.077 | 22 | 0 | 0 | 1 | 64 | 822,874 | GuardFail=16, LoopComplete=46, Poll=2 | Branch=16 | - |
| algorithm/mandelbrot | before-interpreter | 798,492,091 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/mandelbrot | before-tiered | 94,505,295 | 0.118 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/spectral_norm | interpreter | 763,111,978 | 1.000 | 44 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/spectral_norm | tiered | 74,843,339 | 0.098 | 44 | 0 | 0 | 2 | 4,868 | 571,000 | GuardFail=4796, LoopComplete=72 | ArrayBounds=4796 | ShortLoop=4 |
| algorithm/spectral_norm | before-interpreter | 763,104,170 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/spectral_norm | before-tiered | 104,483,666 | 0.137 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/binary_trees | interpreter | 323,401,572 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/binary_trees | tiered | 323,276,424 | 1.000 | 81,922 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| algorithm/binary_trees | before-interpreter | 323,291,987 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| algorithm/binary_trees | before-tiered | 323,178,104 | 0.999 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/captured_call | interpreter | 85,979,856 | 1.000 | 31 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/captured_call | tiered | 26,690,967 | 0.310 | 31 | 0 | 0 | 1 | 1 | 4,934 | LoopComplete=1 | - | - |
| mineclonia/captured_call | before-interpreter | 85,973,235 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/captured_call | before-tiered | 48,559,119 | 0.565 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/module_calls | interpreter | 268,627,980 | 1.000 | 25 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/module_calls | tiered | 55,876,695 | 0.208 | 25 | 0 | 0 | 1 | 1 | 9,934 | LoopComplete=1 | - | - |
| mineclonia/module_calls | before-interpreter | 268,620,177 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/module_calls | before-tiered | 115,584,557 | 0.430 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/open_results | interpreter | 179,116,432 | 1.000 | 23 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/open_results | tiered | 29,434,103 | 0.164 | 23 | 0 | 0 | 2 | 1,492 | 29,877 | CommittedEffect=748, LoopComplete=744 | Branch=748 | - |
| mineclonia/open_results | before-interpreter | 179,109,073 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/open_results | before-tiered | 64,252,193 | 0.359 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/ull_kernel | interpreter | 146,058,929 | 1.000 | 32 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/ull_kernel | tiered | 60,707,993 | 0.416 | 32 | 0 | 0 | 1 | 1 | 3,934 | LoopComplete=1 | - | - |
| mineclonia/ull_kernel | before-interpreter | 146,049,391 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/ull_kernel | before-tiered | 152,906,796 | 1.047 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/ull_arith | interpreter | 142,298,475 | 1.000 | 30 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/ull_arith | tiered | 14,154,912 | 0.099 | 30 | 0 | 0 | 1 | 1 | 49,934 | LoopComplete=1 | - | - |
| mineclonia/ull_arith | before-interpreter | 142,292,143 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/ull_arith | before-tiered | 46,663,828 | 0.328 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/bit_wide | interpreter | 205,515,736 | 1.000 | 29 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| mineclonia/bit_wide | tiered | 12,489,549 | 0.061 | 29 | 0 | 0 | 1 | 1 | 49,934 | LoopComplete=1 | - | - |
| mineclonia/bit_wide | before-interpreter | 205,506,978 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| mineclonia/bit_wide | before-tiered | 50,914,603 | 0.248 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read_small | interpreter | 361,173,054 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_small | tiered | 223,485,692 | 0.619 | 24 | 0 | 0 | 1 | 20 | 959,917 | LoopComplete=17, Poll=3 | - | - |
| table/hash_read_small | before-interpreter | 361,167,675 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read_small | before-tiered | 358,259,658 | 0.992 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read_shaped | interpreter | 366,120,835 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_shaped | tiered | 213,221,601 | 0.582 | 24 | 0 | 0 | 1 | 20 | 959,884 | LoopComplete=17, Poll=3 | - | LoopNotClosed=1 |
| table/hash_read_shaped | before-interpreter | 366,113,064 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read_shaped | before-tiered | 363,105,669 | 0.992 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read_indexed | interpreter | 298,340,100 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/hash_read_indexed | tiered | 202,073,874 | 0.677 | 24 | 0 | 0 | 1 | 35 | 653,774 | GuardFail=16, LoopComplete=17, Poll=2 | ArrayBounds=16 | UnsupportedOpcode=6 |
| table/hash_read_indexed | before-interpreter | 298,308,762 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/hash_read_indexed | before-tiered | 296,672,049 | 0.994 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/heap_redundancy | interpreter | 408,027,677 | 1.000 | 24 | 0 | 0 | 0 | 0 | 0 | - | - | - |
| table/heap_redundancy | tiered | 13,904,892 | 0.034 | 24 | 0 | 0 | 1 | 1 | 149,934 | LoopComplete=1 | - | - |
| table/heap_redundancy | before-interpreter | 408,021,657 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |
| table/heap_redundancy | before-tiered | 20,938,658 | 0.051 | n/a | n/a | n/a | n/a | n/a | n/a | - | - | - |

| Benchmark | Mode | before | after | change |
| :-- | :-- | --: | --: | --: |
| table/array_read | interpreter | 327,959,287 | 327,966,354 | +0.00% |
| table/array_read | tiered | 65,154,160 | 60,667,108 | -6.89% |
| table/array_write | interpreter | 560,613,389 | 560,619,842 | +0.00% |
| table/array_write | tiered | 57,475,717 | 54,816,372 | -4.63% |
| table/hash_read | interpreter | 224,925,151 | 224,927,493 | +0.00% |
| table/hash_read | tiered | 223,353,288 | 141,808,344 | -36.51% |
| table/field_read | interpreter | 401,293,672 | 401,299,488 | +0.00% |
| table/field_read | tiered | 23,991,287 | 17,290,738 | -27.93% |
| table/metamethod_index | interpreter | 304,582,458 | 304,590,536 | +0.00% |
| table/metamethod_index | tiered | 17,564,884 | 9,441,783 | -46.25% |
| table/iteration | interpreter | 508,397,193 | 508,403,905 | +0.00% |
| table/iteration | tiered | 42,765,827 | 32,072,174 | -25.01% |
| algorithm/sieve | interpreter | 276,550,853 | 276,557,071 | +0.00% |
| algorithm/sieve | tiered | 136,362,836 | 126,597,647 | -7.16% |
| algorithm/nbody | interpreter | 76,698,392 | 76,705,873 | +0.01% |
| algorithm/nbody | tiered | 109,635,374 | 74,289,362 | -32.24% |
| algorithm/mandelbrot | interpreter | 798,492,091 | 798,497,663 | +0.00% |
| algorithm/mandelbrot | tiered | 94,505,295 | 61,656,925 | -34.76% |
| algorithm/spectral_norm | interpreter | 763,104,170 | 763,111,978 | +0.00% |
| algorithm/spectral_norm | tiered | 104,483,666 | 74,843,339 | -28.37% |
| algorithm/binary_trees | interpreter | 323,291,987 | 323,401,572 | +0.03% |
| algorithm/binary_trees | tiered | 323,178,104 | 323,276,424 | +0.03% |
| mineclonia/captured_call | interpreter | 85,973,235 | 85,979,856 | +0.01% |
| mineclonia/captured_call | tiered | 48,559,119 | 26,690,967 | -45.03% |
| mineclonia/module_calls | interpreter | 268,620,177 | 268,627,980 | +0.00% |
| mineclonia/module_calls | tiered | 115,584,557 | 55,876,695 | -51.66% |
| mineclonia/open_results | interpreter | 179,109,073 | 179,116,432 | +0.00% |
| mineclonia/open_results | tiered | 64,252,193 | 29,434,103 | -54.19% |
| mineclonia/ull_kernel | interpreter | 146,049,391 | 146,058,929 | +0.01% |
| mineclonia/ull_kernel | tiered | 152,906,796 | 60,707,993 | -60.30% |
| mineclonia/ull_arith | interpreter | 142,292,143 | 142,298,475 | +0.00% |
| mineclonia/ull_arith | tiered | 46,663,828 | 14,154,912 | -69.67% |
| mineclonia/bit_wide | interpreter | 205,506,978 | 205,515,736 | +0.00% |
| mineclonia/bit_wide | tiered | 50,914,603 | 12,489,549 | -75.47% |
| table/hash_read_small | interpreter | 361,167,675 | 361,173,054 | +0.00% |
| table/hash_read_small | tiered | 358,259,658 | 223,485,692 | -37.62% |
| table/hash_read_shaped | interpreter | 366,113,064 | 366,120,835 | +0.00% |
| table/hash_read_shaped | tiered | 363,105,669 | 213,221,601 | -41.28% |
| table/hash_read_indexed | interpreter | 298,308,762 | 298,340,100 | +0.01% |
| table/hash_read_indexed | tiered | 296,672,049 | 202,073,874 | -31.89% |
| table/heap_redundancy | interpreter | 408,021,657 | 408,027,677 | +0.00% |
| table/heap_redundancy | tiered | 20,938,658 | 13,904,892 | -33.59% |
