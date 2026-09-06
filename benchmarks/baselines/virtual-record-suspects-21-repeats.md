Load 22:17:09 up 1 day, 47 min,  1 user,  load average: 2,99, 8,13, 14,62; cpu 7 (sibling 15); oximond sha256 51f51778bae0d71c58ccf37a61a0ae6728c8bfaef2a536c7792b6d64e9c4fc69

| Benchmark | Mode | instructions:u | vs interpreter | tables | elided | elided bytes | traces | entries | iterations | exits | side exits | aborts |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- | :-- |
| numeric/for_sum | tiered | 34,545,094 | n/a | 22 | 0 | 0 | 1 | 8 | 1,902,716 | LoopComplete=1, Poll=7 | - | - |
| numeric/for_sum | before-tiered | 34,616,891 | n/a | 22 | 0 | 0 | 1 | 8 | 1,901,924 | LoopComplete=1, Poll=7 | - | - |
| table/array_read | tiered | 69,224,441 | n/a | 23 | 0 | 0 | 1 | 54 | 1,557,220 | LoopComplete=53, Poll=1 | - | InnerLoop=0, ShortLoop=3 |
| table/array_read | before-tiered | 69,816,224 | n/a | 23 | 0 | 0 | 1 | 55 | 1,553,658 | LoopComplete=54, Poll=1 | - | InnerLoop=0, ShortLoop=3 |
| table/hash_read | tiered | 146,543,596 | n/a | 24 | 0 | 0 | 1 | 19 | 571,009 | LoopComplete=17, Poll=1 | - | InnerLoop=0, UnsupportedOpcode=3 |
| table/hash_read | before-tiered | 147,170,738 | n/a | 24 | 0 | 0 | 1 | 41 | 567,117 | LoopComplete=40, Poll=1 | - | InnerLoop=0, UnsupportedOpcode=3 |
| table/metamethod_index | tiered | 17,393,356 | n/a | 25 | 0 | 0 | 1 | 1 | 194,687 | LoopComplete=1 | - | - |
| table/metamethod_index | before-tiered | 17,489,214 | n/a | 25 | 0 | 0 | 1 | 1 | 194,621 | LoopComplete=1 | - | - |
| table/iteration | tiered | 32,756,312 | n/a | 23 | 0 | 0 | 1 | 80 | 796,339 | GuardFail=80 | ArrayBounds=80 | ShortLoop=3, UnsupportedOperand=1 |
| table/iteration | before-tiered | 32,639,442 | n/a | 23 | 0 | 0 | 1 | 80 | 796,504 | GuardFail=80 | ArrayBounds=80 | ShortLoop=3, UnsupportedOperand=1 |
| algorithm/nbody | tiered | 109,514,598 | n/a | 28 | 0 | 0 | 2 | 2,560 | 5,195 | LoopComplete=2560 | - | InnerLoop=6, LoopNotClosed=2 |
| algorithm/nbody | before-tiered | 109,876,421 | n/a | 28 | 0 | 0 | 2 | 2,544 | 5,179 | LoopComplete=2544 | - | InnerLoop=6, LoopNotClosed=2 |
| algorithm/spectral_norm | tiered | 102,502,077 | n/a | 44 | 0 | 0 | 2 | 4,688 | 546,176 | GuardFail=4606, LoopComplete=71 | ArrayBounds=4606 | InnerLoop=0, ShortLoop=4 |
| algorithm/spectral_norm | before-tiered | 101,894,694 | n/a | 44 | 0 | 0 | 2 | 4,866 | 546,631 | GuardFail=4611, LoopComplete=269 | ArrayBounds=4611 | InnerLoop=2, ShortLoop=4 |
| mineclonia/module_calls | tiered | 116,377,672 | n/a | 25 | 0 | 0 | 1 | 1 | 7,558 | LoopComplete=1 | - | - |
| mineclonia/module_calls | before-tiered | 118,869,869 | n/a | 25 | 0 | 0 | 1 | 1 | 7,459 | LoopComplete=1 | - | - |
| mineclonia/open_results | tiered | 63,733,924 | n/a | 23 | 0 | 0 | 2 | 1,182 | 23,620 | CommittedEffect=591, LoopComplete=591 | Branch=591 | - |
| mineclonia/open_results | before-tiered | 65,080,829 | n/a | 23 | 0 | 0 | 2 | 1,169 | 23,369 | CommittedEffect=584, LoopComplete=585 | Branch=584 | - |
| mineclonia/ull_arith | tiered | 45,767,155 | n/a | 30 | 0 | 0 | 1 | 1 | 38,153 | LoopComplete=1 | - | - |
| mineclonia/ull_arith | before-tiered | 46,108,853 | n/a | 30 | 0 | 0 | 1 | 1 | 38,021 | LoopComplete=1 | - | - |

| Benchmark | Mode | before | after | change |
| :-- | :-- | --: | --: | --: |
| numeric/for_sum | tiered | 34,616,891 | 34,545,094 | -0.21% |
| table/array_read | tiered | 69,816,224 | 69,224,441 | -0.85% |
| table/hash_read | tiered | 147,170,738 | 146,543,596 | -0.43% |
| table/metamethod_index | tiered | 17,489,214 | 17,393,356 | -0.55% |
| table/iteration | tiered | 32,639,442 | 32,756,312 | +0.36% |
| algorithm/nbody | tiered | 109,876,421 | 109,514,598 | -0.33% |
| algorithm/spectral_norm | tiered | 101,894,694 | 102,502,077 | +0.60% |
| mineclonia/module_calls | tiered | 118,869,869 | 116,377,672 | -2.10% |
| mineclonia/open_results | tiered | 65,080,829 | 63,733,924 | -2.07% |
| mineclonia/ull_arith | tiered | 46,108,853 | 45,767,155 | -0.74% |
