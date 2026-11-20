# Pre-registered hypotheses and controls (written 2026-11-19, before any student was trained)

Written while the teacher was generating, after the 576-candidate teacher pilot and the filter calibration on real
stories; no student model existed yet and no test data had been used.

## Fixed conditions (identical for every student)

| item | value |
|---|---|
| task | TinyStories-style stories (option A of the guide) |
| real data | 5,000 real training stories (970,921 tokens) = the scarce real pool; also the source of seed prompts |
| teacher | TinyStories-33M (MIT) @ 2ad0a164, trained by others on all TinyStories training stories; CPU, fp32 |
| teacher budget | 5,000 prompts x 3 candidates per pool (main, lowT, plain), max 320 teacher tokens each |
| student | Project 1 GPT, 4 layers / 4 heads / 256 wide (~5.3M parameters), from scratch; Project 1 tokenizer |
| training budget | 6.0M tokens = 1,464 updates x 16 x 256 tokens, AdamW, cosine schedule, learning rate from a pilot |
| checkpoint | the one with the lowest loss on 500 real validation stories (10 evaluations per run) |
| seeds | 3 training seeds (0, 1, 2) for the 7 main systems; ablations seed 0 |
| splits | seeds / fresh from TinyStories train; validation and test from TinyStories validation (never seen by the teacher); frozen in data_manifests/seeds.json |
| filter thresholds | calibrated on real seed stories (data_manifests/filter_calibration.json), fixed before training |

## Metrics

* **Primary: test loss** = mean loss per token (nats) on 2,000 held-out real test stories. Lower is better.
  Systems are compared with the per-story loss averaged over their 3 seeds, paired by story, with a 95% bootstrap
  interval (10,000 resamples). A hypothesis "holds" only if the interval is entirely on the predicted side.
* Secondary (generations from 300 held-out test openings): scorer loss under SmolLM2-135M (independent), distinct-2,
  repetitive share, ended share, copied share (near duplicate of a real seed story).
* Robustness: test loss on the test stories without non-ASCII characters (5.5% of real stories contain encoding
  debris such as "â€™" that the synthetic filter removes).

## Hypotheses

| # | hypothesis | test |
|---|---|---|
| H1 | **Main.** Real + filtered synthetic beats real-only at the same token budget: test loss mix50_filt < real_only | 3 seeds |
| H2 | **Main.** Filtering matters: mix50_filt < mix50_raw | 3 seeds |
| H3a | Too much synthetic hurts: syn_only is worse than the best of mix25/50/75_filt | 3 seeds |
| H3b | Low prompt diversity hurts: mix50_plain > mix50_filt | ablation rule |
| H4 | Synthetic data does not fully replace new real data: real_fresh < mix50_filt | 3 seeds |
| H5 | Generations improve too: scorer loss of mix50_filt < real_only | 3 seeds x 300 |
| H6 | Real-only memorises its 5,000 stories: copied share of real_only > mix50_filt | 3 seeds x 300 |
| H7 | Lower teacher temperature (0.6) gives less diverse data and a worse student: mix50_lowT > mix50_filt | ablation rule |
| H8 | 3 candidates per prompt beat 1 (more distinct data, less repetition): mix50_cand1 > mix50_filt | ablation rule |
| H9 | Deduplication barely matters here (few duplicates): \|mix50_nodedup − mix50_filt\| inside the seed range | ablation rule |
| H10 | The model-based judge filter helps further: mix50_judge < mix50_filt | ablation rule |
| H11 | A bigger student (13.9M) gains more from synthetic data, because it overfits 5,000 stories faster: (real_only − mix50_filt) larger for `small` than for `base` | seed 0 |

**Ablation rule** (one seed per ablation): an ablation effect counts only if (a) its paired 95% interval against
mix50_filt seed 0 excludes zero *and* (b) its test loss lies outside the range of the three mix50_filt seeds.
Otherwise it is reported as "no effect detectable beyond seed-to-seed variation".
