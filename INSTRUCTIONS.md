# Project 9 — Synthetic Data Generation and Training: phase-by-phase instructions

Source plan: `docs/Project_9_Synthetic_Data_Generation_and_Training_Guide.docx` (kept locally, not in the repository).

**Goal:** use a larger *teacher* model to write synthetic training stories, clean them with a multi-stage quality
filter, mix them with a small amount of real data, train small *student* models from scratch on the mixtures, and
measure (on held-out **real** stories) whether the synthetic data helps, how much filtering matters, and what it
costs.

**Research question:** when real data is scarce, does teacher-generated synthetic data improve a small student at a
fixed training budget, and which quality controls and mixture ratios matter?

```
5,000 real stories (scarce) ──► opening sentences / story frames ──► teacher TinyStories-33M writes 3 stories each
   ──► parse ──► 9-stage filter (schema, length, ending, repetition, language, contamination, dedup, copy, judge)
   ──► mixtures (0/25/50/75/100% synthetic, raw vs filtered, ablations), all 6M tokens
   ──► 5.3M-parameter student from scratch ──► test loss on 2,000 held-out real stories + generations
```

## How to use this file
1. Work through the phases in order. Each has **Learn**, **Files**, **Run**, **Check** and **Result**.
2. Run everything from the `project9` folder with `.venv` active (Windows: `.venv\Scripts\activate`).
3. Status changes from ⬜ to ✅ when the phase is done and checked.

## Progress

| # | Phase | Status |
|---|---|---|
| 1 | Environment and project scope | ✅ |
| 2 | Research hypotheses and controls | ✅ |
| 3 | Seed data and held-out evaluation | ✅ |
| 4 | Select and evaluate the teacher | ✅ |
| 5 | Prompt templates | ✅ |
| 6 | Generate synthetic examples | ✅ |
| 7 | Parse and normalise records | ✅ |
| 8 | Multi-stage quality filter | ✅ |
| 9 | Deduplication and contamination | ✅ |
| 10 | Dataset quality and diversity | ✅ |
| 11 | Training mixtures | ✅ |
| 12 | Train the student models | ✅ |
| 13 | Evaluate model improvement | ✅ |
| 14 | Ablation experiments | ✅ |
| 15 | Error analysis and failure modes | ✅ |
| 16 | Efficiency and reproducibility | ✅ |

## Decisions for this PC and this project (senior-researcher choices, with reasons)

- **CPU only** (no GPU on this PC), fp32, 24 threads, as in Projects 1–8. "GPU-hours" become CPU wall-clock hours.
- **Task A of the guide: short children's stories (TinyStories).** It reuses Project 1's cleaned data, tokenizer and
  GPT code, and stories can be checked by rules (length, ending, repetition, vocabulary) without a big judge model.
- **Make real data genuinely scarce.** With all 2M TinyStories available, synthetic data cannot matter. So the
  student gets only **5,000 real stories (0.97M tokens)** and a **6M-token training budget**: real-only training has
  to repeat its stories about 6 times, and synthetic stories are the only way to see *new* text. A reference system
  trained on 6M tokens of *other real stories* (`real_fresh`) shows what more real data would have given.
- **Teacher = TinyStories-33M** (GPT-Neo, MIT licence, revision `2ad0a164`), already downloaded in Project 4 and
  loaded read-only. It is ~6× the student's size and was trained on all of TinyStories, so it "knows" far more stories
  than our 5,000. TinyStories-Instruct-33M was rejected: no licence is listed on its model card.
- **An independent scorer for generations: SmolLM2-135M** (Apache-2.0, revision `93efa2f0`). It never saw the teacher's
  outputs and is a different model family, so "the student writes like the teacher" is not automatically rewarded as
  "fluent". (It still prefers bland text, see Phase 13.)
- **The primary metric is loss on real held-out stories,** never on synthetic text. Synthetic data only helps if it
  makes the student better at *real* stories.
- **Every threshold of the filter is calibrated on real stories** (rejects about 1% of real seed stories per rule)
  and frozen before any student is trained. Hypotheses were written down before training (`configs/hypotheses.md`).
- **Three seeds per main system,** one seed per ablation, and an ablation only "counts" if it moves the result
  outside the spread of the main system's three seeds (lesson from Project 8).

---
## Phase 1 — Environment and project scope

**Learn:** *synthetic data* is training text written by a model instead of people. A strong *teacher* model writes
examples; a smaller *student* learns from them. This is close to *distillation* (Project 7), but here the student
only sees the teacher's *text*, not its probabilities, so any model that can write can be a teacher. The risks are
that the teacher's mistakes and habits are copied, that the data is less varied than real data, and that the teacher
might have memorised the test set. The project is designed to measure each of these.

**Files:** `requirements.txt`, `requirements.lock.txt`, `synth/__init__.py` (paths, hashing, JSONL helpers; makes
Project 1's `src` importable read-only), `scripts/env_report.py`, `scripts/download_scorer.py`

**Run:**
```
python -m venv .venv
.venv\Scripts\pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
python scripts/download_scorer.py     # SmolLM2-135M at a pinned revision -> checkpoints/scorer_smollm2_135m
python scripts/env_report.py          # -> results/metrics/env.json
```

**Check:** `env.json` lists versions, licences, model revisions and sha256 hashes of every input file.

**Result:** Python 3.12.10, torch 2.14.1+cpu, transformers 5.22.2, 24 CPU threads, 63.7 GB RAM, no CUDA. Teacher
weights sha256 `41316d3b…6dad302`, scorer `80521b40…d55da1`, Project 1 tokenizer `158ade55…0f530`, judge model
(Project 1 small) `b4a9986a…611361`. Data licence: TinyStories, CDLA-Sharing-1.0.

---
## Phase 2 — Research hypotheses and controls

**Learn:** decide *before* training what you expect and how you will decide. Otherwise it is easy to find a story
for any result afterwards. **Controls** keep everything fixed except the one thing being studied: here every student
has the same architecture, token budget, optimiser, schedule, validation set and checkpoint rule, and only the
*training data* changes.

**Files:** `configs/hypotheses.md` (written 2026-11-19 while the teacher was still generating), `configs/systems.yaml`

**Result:** 11 hypotheses (H1–H11), the primary metric, the comparison method and the ablation rule are fixed in
`configs/hypotheses.md`. The main ones: **H1** real + filtered synthetic beats real-only; **H2** filtering beats
raw synthetic; **H3** synthetic-only and undiverse prompts hurt; **H4** synthetic does not fully replace new real
data. The verdicts are in Phase 13 and 14.

---
## Phase 3 — Seed data and held-out evaluation

**Learn:** *seed data* is the small real set the teacher is prompted with. *Held-out* data is never shown to the
teacher, the filter or the student and is only used to measure. Overlap between them would make the results
meaningless, so every split comes from a different source and the overlaps are checked by hash.

**Files:** `synth/prepare_seeds.py`

**Run:** `python -m synth.prepare_seeds` → `data/seeds/*.jsonl`, `data_manifests/seeds.json`

**Check:** zero pairwise overlap (exact text hash) between all splits; every story is "well formed" (starts with a
capital letter, 80–450 tokens, first sentence 4–30 words, at least 3 sentences).

**Result:**

| split | source | stories | tokens | use |
|---|---|---|---|---|
| seeds | TinyStories train, first 200k (shuffled, seed 9) | 5,000 | 970,921 | the scarce real data + teacher prompts |
| fresh | same window, disjoint | 40,000 | 7,787,570 | `real_fresh` reference only; vocabulary for the language rule |
| val | TinyStories validation (seed 10) | 500 | — | checkpoint choice, learning-rate pilot |
| test_loss | TinyStories validation (seed 11) | 2,000 | 381,797 | **primary metric** |
| test_prompt | TinyStories validation | 300 | — | opening sentences for generation tests |

128 candidate test prompts were skipped because their opening sentence was identical to a seed's opening.
**Data quirk found:** 5.5% of real TinyStories (2,488 of 45,000) contain encoding debris such as `â€™`. The synthetic
filter removes any non-ASCII text, so the student trained on synthetic data never sees it. Every test comparison is
therefore repeated on the ASCII-only test stories.

---
## Phase 4 — Select and evaluate the teacher

**Learn:** a teacher must be (1) good enough to write useful examples, (2) licensed for this use, (3) affordable to run
many times, and (4) not trained on your test data. Before generating thousands of examples, run a **pilot** and read
the output.

**Files:** `synth/generate_teacher.py`, `configs/teacher.yaml`, `results/pilot_review/`

**Run:** `python -m synth.generate_teacher --gen main --limit 192 --out pilot_main` then the filter on the pilot.

**Check:** the outputs read like stories; no near-copies of held-out stories; speed is acceptable.

**Result:**
- **Pilot:** 576 candidates; the rule filter kept 93.9%, the judge filter 88.4%. 0 near-duplicates of held-out
  stories. 12.7% share a 13-word sequence with a held-out story, *less* than real training stories do (23.7%), so this
  is ordinary story phrasing ("Once upon a time, there was a little girl named"), not leakage.
- **Review of 100 pilot stories by an AI annotator (Claude, not a human), `results/pilot_review/labels_ai.json`:**
  coherence 0/1/2 = 16/72/12, relevance to the prompt 2 for 90/100, complete 95/100, child-appropriate 99/100.
  The most common issue is a *logic error* (66/100, e.g. a puppy comes out of a locked box), then contradictions
  (16), characters swapping names (13), grammar (11). **The teacher is fluent and on-topic but often illogical.**
- **Speed:** plain Hugging Face generation ran at 378 teacher tokens/s. Generating in chunks and removing finished
  stories from the batch raised it to 555; top-k 50 sampling instead of top-p 0.95 (top-p sorts the whole vocabulary
  every step and was 1.4× slower) and 96 prompts per batch gave **~730 tokens/s**, about one hour per 15,000 stories.

---
## Phase 5 — Prompt templates

**Learn:** a teacher given the same instruction writes the same kind of story. **Diversity has to be put into the
prompts.** Two kinds of prompt are used, plus a deliberately bad one for the ablation.

**Files:** `synth/templates.py`, `prompts/templates.jsonl`

**Result:**

| template | example | prompts |
|---|---|---|
| `open` | the first sentence of a real seed story: "John and his sister had a wonderful day." | 2,500 |
| `frame` | "Once upon a time, in a sunny meadow, there was a curious fox named Mia. She wanted to find a hidden treasure." built from 20 adjectives, 18 animals and people, 40 names, 20 places, 40 goals | 2,500 |
| `plain` (ablation only) | "Once upon a time," for every prompt | 5,000 |

Prompt sampling uses seed 21; each batch's generation seed is 1000 + the index of its first prompt, so any batch can be
regenerated on its own.

---
## Phase 6 — Generate synthetic examples

**Learn:** sampling settings decide the trade-off between quality and variety. **Temperature** divides the logits: a
low temperature (0.6) picks likely words and gives safe, similar stories; temperature 1.0 gives more variety and more
mistakes. **Top-k 50** only samples among the 50 most likely tokens, which cuts the worst mistakes. Several
**candidates** per prompt give more distinct stories per prompt.

**Files:** `synth/generate_teacher.py`, `configs/teacher.yaml`, `scripts/generate_all.sh`

**Run:** `bash scripts/generate_all.sh` (resumable; about 3 hours)

**Check:** every record stores the prompt, template, teacher name and revision, decoding settings, seed, finish reason
and token count (`data/raw/<pool>.jsonl`, `data_manifests/raw_<pool>.json`).

**Result:** three pools of 5,000 prompts × 3 candidates = 15,000 stories each, max 320 new tokens.

| pool | settings | teacher tokens | time | tokens/s | hit the 320-token limit |
|---|---|---|---|---|---|
| main | T 1.0, top-k 50, `open` + `frame` prompts | 2,632,787 | 60.1 min | 731 | 682 |
| lowT | T 0.6, same prompts | 2,550,350 | 58.9 min | 721 | 697 |
| plain | T 1.0, "Once upon a time," only | 2,748,680 | 57.1 min | 803 | 46 |

---
## Phase 7 — Parse and normalise records

**Learn:** model output is messy: empty answers, stories cut off at the token limit, strange characters, the
end-of-text marker written out as text. Parse every record into a fixed **schema** and *flag* problems instead of
silently dropping them, so you can count what went wrong.

**Files:** `synth/parse_outputs.py`, `tests/test_parser.py`

**Run:** `python -m synth.parse_outputs --gen main` (and `lowT`, `plain`) → `data/parsed/<pool>.jsonl`

**Result:** the story is the prompt plus the continuation, cleaned with Project 1's `clean_story` (same normalisation
as the real data). Flags: main — 682 truncated, 75 malformed, 4 empty; lowT — 697 / 1 / 2; plain — 46 / 73 / 0.

---
## Phase 8 — Multi-stage quality filter

**Learn:** a filter is a series of cheap checks, each with a clear reason, so you can see *what* was removed. Each
threshold is set on **real** stories ("reject what is rarer than the worst 1% of real stories"), not tuned until the
student looks good. A model-based **judge** (here: Project 1's small model, which scores how surprising a story is)
can catch what rules miss, but it may also remove good unusual stories.

**Files:** `synth/filter_quality.py`, `synth/judge.py`, `tests/test_filters.py`,
`data_manifests/filter_calibration.json`, `data_manifests/filter_<pool>.json`

**Run:** `python -m synth.filter_quality --calibrate`, then `--gen main` (all variants), `--gen lowT`, `--gen plain`

**Stages:** 1 schema (empty / truncated / malformed: non-ASCII or a literal `<|endoftext|>`) → 2 length 80–450 student
tokens → 3 proper ending (`.`, `!` or `?`, optionally followed by a quote) → 4 repetition (share of repeated 4-grams
above the real 99th percentile 0.064, or one sentence 3+ times) → 5 language (share of words never seen twice in 40k
real stories above the real 99th percentile 0.0325) → 6 contamination (near-duplicate of a val/test story) →
7 deduplication (Phase 9) → 8 copies of a seed story → 9 judge (optional: Project 1 loss above the real 90th
percentile, 2.24). On real seed stories the rules reject 2.0%.

**Variants:** `raw` (only empty and contaminated removed), `filt` (stages 1–8, the main one), `nodedup` (1–6), `judge`
(1–9), `cand1` (`filt` on the first candidate of each prompt only).

**Result:** ([figure 1](results/figures/fig1_filter_funnel.png))

| pool | kept by `filt` | main reasons for removal |
|---|---|---|
| main | 13,966 / 15,000 (93.1%), 2.48M tokens | 682 hit token limit, 96 no ending, 75 unusual words, 71 odd characters, 56 length, 50 repetitive |
| lowT | 13,710 (91.4%) | 697 token limit, **453 repetitive** (low temperature loops) |
| plain | 14,674 (97.8%) | very little |
| main, `judge` | 13,034 (86.9%) | the judge removes 932 more |

Filtering all 15,000 stories takes about 1 s for the rules and 5 s for dedup; the judge takes 165 s.

---
## Phase 9 — Deduplication and contamination

**Learn:** **exact duplicates** have the same text; **near duplicates** share most of their wording. Comparing every
pair is too slow (15,000² pairs), so **MinHash + LSH** gives each story a short fingerprint of its word 5-grams and only
compares stories whose fingerprints collide; the candidates are then checked exactly (Jaccard similarity ≥ 0.5 =
duplicate). **Contamination** means synthetic stories that copy the test set; that would inflate the test score.

**Files:** `synth/deduplicate.py`, `tests/test_dedup.py`, `results/metrics/dedup_check.json`

**Result:**
- **Contamination: 0** synthetic stories are near duplicates of a val/test story (the 99th percentile of the closest
  match is Jaccard 0.049).
- **Deduplication removed 0 stories in all three pools.** To make sure this is real and not a bug, the three
  candidates of each prompt (the most likely duplicates) were compared directly: mean Jaccard 0.049 (main), 0.060
  (lowT), 0.012 (plain); only 4 of 45,000 pairs are above 0.3, none above 0.5. The one pair with Jaccard 1.0 is two
  empty outputs (text = prompt), already removed as "empty". Sampling at temperature 0.6–1.0 from a 33M model simply
  does not repeat itself. Consequence: the `nodedup` mixture is **byte-identical** to `filt` (same sha1), so it was not
  trained, and H9 is true by construction.
- **Copies of seed stories: 0.**

---
## Phase 10 — Dataset quality and diversity

**Learn:** measure the data, not only the model. **distinct-2** = distinct word pairs ÷ all word pairs over 2,000
stories (higher = more varied). **Name concentration** = share of "named X" characters that use the five most common
names (high = stylistic collapse). **Vocabulary coverage** = share of the real vocabulary that appears.

**Files:** `synth/dataset_report.py` → `results/metrics/dataset_report.json`

**Result:** ([figure 2](results/figures/fig2_diversity.png))

| data | stories | median length | distinct-2 | real vocabulary covered | top-5 name share |
|---|---|---|---|---|---|
| real seeds | 5,000 | 178 | 0.240 | 0.502 | 0.581 |
| real (fresh) | 5,000 | 178 | 0.239 | 0.514 | 0.599 |
| main, unfiltered | 14,996 | 176 | 0.218 | 0.424 | 0.259 |
| main, filtered | 13,966 | 173 | 0.218 | 0.413 | 0.262 |
| low temperature | 13,710 | 167 | **0.163** | 0.318 | 0.285 |
| no prompt diversity | 14,674 | 174 | 0.211 | 0.421 | **0.692** |

The synthetic stories are a little less varied than real ones. Lower temperature costs a quarter of the word-pair
variety. Without varied prompts the teacher falls back on the same few names (Lily, Tom, …). The main pool's *low*
name concentration comes from the `frame` template, which draws 40 names uniformly.

---
## Phase 11 — Training mixtures

**Learn:** a **mixture** fixes how many training tokens come from each source. Keeping the *total* fixed (6M tokens)
is what makes the comparison fair: a model that sees more tokens would win anyway. When a source is smaller than its
share, it is repeated (another shuffled *epoch*).

**Files:** `synth/build_mixtures.py`, `configs/mixtures.yaml`, `tests/test_mixtures_losses.py`

**Run:** automatic when a student is trained (`data/mixtures/<mixture>_s<seed>.bin`, manifest in
`data_manifests/mixtures/`)

**Result:** every story is one unit `<BOS> story <EOS>`; units of both sources are shuffled together (seed
1000·seed + 7).

| mixture | real share | synthetic source | real epochs | synthetic epochs |
|---|---|---|---|---|
| real_only | 100% | — | 6.1 | — |
| real_fresh | 100% (other real stories) | — | 0.8 of 40k | — |
| mix25_filt / mix50_filt / mix75_filt | 75 / 50 / 25% | main, filtered | 4.6 / 3.1 / 1.5 | 0.6 / 1.2 / 1.8 |
| syn_only | 0% | main, filtered | — | 2.4 |
| mix50_raw | 50% | main, unfiltered | 3.1 | 1.1 |
| mix50_lowT / mix50_plain / mix50_judge | 50% | lowT / plain / main judge | 3.1 | ~1.2 |
| mix50_cand1 | 50% | main, first candidate only (4,649 stories) | 3.1 | 3.6 |

---
## Phase 12 — Train the student models

**Learn:** the student is trained **from scratch** (random weights) so it only knows what its mixture teaches it.
The **learning rate** matters a lot for small models, so it is chosen in a short **pilot** on validation data (never
test data), averaged over a real-only and a synthetic mixture so the choice does not favour either. When the best
value sits at the edge of the grid, extend the grid.

**Files:** `synth/train_student.py`, `synth/losses.py`, `configs/student.yaml`, `scripts/lr_pilot.sh`,
`scripts/train_all.sh`, `results/metrics/lr_pilot.json`

**Run:** `bash scripts/lr_pilot.sh` (+ the two extension values), then `bash scripts/train_all.sh` (about 5 hours)

**Setup:** Project 1 GPT, 4 layers, 4 heads, width 256, dropout 0.1, **5,322,240 parameters**; 1,464 updates × 16 ×
256 tokens = 6.0M tokens; AdamW (betas 0.9/0.95, weight decay 0.1), 100 warm-up steps, cosine decay to 0.1×, gradient
clip 1.0. 10 evaluations on the 500 real validation stories; the best one is kept.

**Learning-rate pilot (seed 100, mean real-validation loss):**

| peak lr | 5e-4 | 1e-3 | 2e-3 | **4e-3** | 8e-3 |
|---|---|---|---|---|---|
| validation loss | 3.237 | 3.037 | 2.910 | **2.876** | 3.062 |

The first grid (5e-4…2e-3) had its best at the edge, so 4e-3 and 8e-3 were added. **4e-3 was used for every
student.** The bigger `small` student (6 layers, 384 wide, 13.9M parameters) used the same value without its own
pilot (a limitation, see H11).

**Result:** 27 runs (7 main systems × 3 seeds, 4 ablations, 2 size runs), about 9.9 min each for the base student
(10,150 tokens/s, peak memory 2.1 GB) and 25 min for `small`. **In every run the best validation loss is at the last
step:** none overfits within 6M tokens, not even real-only, which repeats its 5,000 stories 6 times
([figure 5](results/figures/fig5_validation_curves.png)). The data is scarce, but the *compute* is too small for
memorisation to set in; this limits how much new stories can help.

---
## Phase 13 — Evaluate model improvement

**Learn:** **test loss** = the average surprise (nats per token) of the student on 2,000 real stories it never saw.
It measures how well the student models *real* stories, the thing we care about. Comparing two systems: per story,
the loss is averaged over each system's seeds; the difference is summed over stories, and **bootstrapping** (redraw
the 2,000 stories with replacement 10,000 times) gives a 95% interval. This interval only covers *which test stories*
were drawn. Because different seeds vary too, a second check was added after training: in how many of the 3 × 3
**seed pairings** is A better than B? A conclusion is called *robust* only when both agree.

**Files:** `synth/evaluate.py`, `synth/sampling.py`, `synth/compare.py`, `scripts/eval_all.sh`,
`results/metrics/summary_table.json`, `results/metrics/comparisons.json`

**Run:** `bash scripts/eval_all.sh`, then `python -m synth.compare`

**Result — test loss (lower is better):** ([figure 3](results/figures/fig3_test_loss.png),
[figure 4](results/figures/fig4_mixture_ratio.png))

| system | test loss (mean of 3 seeds) | seeds | perplexity |
|---|---|---|---|
| real_fresh (6M tokens of *new* real stories, reference) | **2.816** | 2.814 – 2.817 | 16.7 |
| mix50_filt (50% filtered synthetic) | **2.855** | 2.843 – 2.867 | 17.4 |
| mix25_filt | 2.860 | 2.852 – 2.872 | 17.5 |
| mix50_raw (50% unfiltered) | 2.864 | 2.834 – 2.902 | 17.5 |
| mix75_filt | 2.885 | 2.882 – 2.887 | 17.9 |
| real_only (5,000 stories, 6 epochs) | 2.898 | 2.880 – 2.926 | 18.1 |
| syn_only (100% synthetic) | 2.991 | 2.980 – 2.998 | 19.9 |

**Paired comparisons** (A − B, negative = A better; seed pairings where A is better):

| comparison | difference [95% CI] | ASCII-only | seed pairings |
|---|---|---|---|
| mix50_filt − real_only | **−0.0425** [−0.0465, −0.0385] | −0.0430 | 9/9 |
| mix25_filt − real_only | −0.0381 [−0.0408, −0.0354] | −0.0382 | 9/9 |
| mix75_filt − real_only | −0.0130 [−0.0193, −0.0070] | −0.0166 | 5/9 |
| mix50_raw − real_only | −0.0331 [−0.0366, −0.0297] | −0.0342 | 7/9 |
| mix50_filt − mix50_raw | −0.0094 [−0.0114, −0.0075] | −0.0088 | **5/9** |
| mix25_filt − mix50_filt | +0.0045 [+0.0020, +0.0070] | +0.0048 | 4/9 |
| mix75_filt − mix50_filt | +0.0295 [+0.0265, +0.0324] | +0.0264 | 0/9 |
| syn_only − mix50_filt | +0.1356 [+0.1200, +0.1525] | +0.0743 | 0/9 |
| syn_only − real_only | +0.0930 [+0.0757, +0.1115] | +0.0313 | 0/9 |
| real_fresh − mix50_filt | −0.0391 [−0.0434, −0.0347] | −0.0370 | 9/9 |

**Generations** (300 held-out openings, T 0.8, top-k 50; [figure 7](results/figures/fig7_generations.png)):

| system | scorer loss (SmolLM2) | distinct-2 | repetitive | finishes the story | copies a training story |
|---|---|---|---|---|---|
| real continuations | 2.047 | 0.444 | 0.3% | 100% | — |
| real_fresh | 2.782 | 0.258 | 1.3% | 86.7% | 0 |
| real_only | 2.805 | 0.252 | 1.4% | 84.8% | 0 |
| mix50_filt | 2.728 | 0.250 | 1.9% | 84.1% | 0 |
| mix75_filt | 2.742 | 0.261 | 1.4% | 91.4% | 0 |
| syn_only | 2.701 | 0.259 | 1.3% | 91.1% | 0 |
| mix50_raw | 2.787 | 0.254 | 1.9% | 81.4% | 0 |

mix50_filt − real_only scorer loss: −0.078 [−0.099, −0.056].

**What this means:**
- **H1 holds, robustly.** Replacing half of the repeated real tokens with filtered synthetic stories lowers test
  loss by 0.043 nats/token, in all 9 seed pairings and on ASCII-only stories.
- **It is about half of what new real data gives.** 6M tokens of *other* real stories improve by 0.082; synthetic
  data recovers **52%** of that gain (**H4 holds**).
- **Dose curve:** 25% ≈ 50% > 75% > 0% > 100%. A little synthetic data helps; replacing most real data hurts, and
  synthetic alone is the worst system (**H3a holds**). About half of syn_only's penalty is the encoding debris it
  never saw (ASCII-only: +0.074 instead of +0.136); the rest is real: teacher stories differ from real ones.
- **H2 (filtering helps) is only weak.** The story-level interval excludes 0 (−0.009), as the pre-registered rule
  requires, but unfiltered mix50_raw varies so much between seeds (2.834–2.902) that only 5 of 9 seed pairings agree.
  Only 7% of the main pool was removed, so `raw` and `filt` are 93% the same data. Filtering mainly made results more
  *stable*, not clearly better.
- **H5 holds but needs care.** Students trained with synthetic data get a better scorer loss, but the scorer also
  prefers syn_only (worst test loss) and the low-temperature student (most repetitive, Phase 14): it rewards bland,
  teacher-like text. Test loss on real stories stays the main evidence.
- **H6 fails:** no student copies a training story (0 of 300 in every run). At 6 epochs a 5.3M model does not memorise.

---
## Phase 14 — Ablation experiments

**Learn:** an **ablation** changes one ingredient and keeps the rest. With one seed per ablation, an effect only
*counts* if (a) its interval against the main system's seed 0 excludes zero **and** (b) its test loss lies outside the
range of the main system's three seeds (2.843 – 2.867). Rule (b) protects against calling seed noise an effect.

**Files:** `configs/systems.yaml` (ablation pairs), `results/metrics/comparisons.json`

**Result:** ([figure 6](results/figures/fig6_ablations.png))

| ablation (seed 0) | test loss | vs mix50_filt seed 0 [95% CI] | outside seed range? | counts? |
|---|---|---|---|---|
| no prompt diversity (`plain`) | 2.852 | +0.0090 [+0.0048, +0.0131] | no | no → **H3b not confirmed** |
| teacher T 0.6 (`lowT`) | 2.863 | +0.0203 [+0.0168, +0.0238] | no | no → **H7 not confirmed** |
| 1 candidate per prompt (`cand1`) | 2.867 | +0.0243 [+0.0212, +0.0273] | yes | **yes → H8 holds** |
| + judge filter (`judge`) | 2.852 | +0.0086 [+0.0058, +0.0112] | no | no → **H10 not confirmed** |
| no dedup (`nodedup`) | identical data | — | — | **H9 true by construction** |
| bigger student, mix50_filt − real_only | 2.972 vs 3.012 | −0.040 [−0.045, −0.035] | — | **H11 not supported** (base: −0.044) |

- **Number of distinct synthetic stories matters most.** With one candidate per prompt only 4,649 stories exist and
  are repeated 3.6 times; three candidates give 13,966 stories seen 1.2 times. More *different* stories beat the same
  stories repeated.
- **Low temperature** is worse on test loss and its student is the most repetitive (3.3%) and name-collapsed (80% top-5
  names), yet the scorer likes it best (2.53): the clearest example of a fluency score rewarding dull text.
- **The judge filter** removed 932 more stories without a detectable gain.
- **The bigger student** did *worse* than the small one (3.012 vs 2.898 for real-only), very likely because the
  learning rate chosen for the 5.3M model is too high for 13.9M and because 6M tokens are too few for it. Its gain from
  synthetic data (−0.040) is about the same as the base student's, so H11 is not supported, but this test is weak.

---
## Phase 15 — Error analysis and failure modes

**Learn:** averages hide *how* models fail. Look at concrete outputs, count failure types, and compare them with the
same measures on real text. A **blind** review hides which system wrote each text so the reviewer cannot favour one.

**Files:** `synth/errors.py`, `results/metrics/errors_auto.json`, `results/error_review/` (blind sample, key, labels),
`results/metrics/errors_review.json`

**Run:** `python -m synth.errors --stage auto`, `--stage sample`, label `blind.jsonl`, `--stage count`

**Automatic measures (300 continuations per run, seed-averaged):**

| system | "moral of the story" | named character forgotten | top-5 name share | repetitive | unfinished |
|---|---|---|---|---|---|
| real continuations | 3.0% | 0% | 0.71 | 0.3% | 0% |
| real_only | 2.1% | 43% | 0.72 | 1.4% | 15.2% |
| real_fresh | 1.0% | 47% | 0.68 | 1.3% | 13.3% |
| mix50_filt | 2.7% | 42% | 0.70 | 1.9% | 15.9% |
| mix50_raw | 2.8% | 42% | 0.70 | 1.9% | 18.6% |
| syn_only | 3.7% | 39% | 0.72 | 1.3% | 8.9% |
| mix50_lowT | 3.3% | 29% | **0.80** | **3.3%** | 13.7% |

("named character forgotten" = of the 52 prompts that introduce a named character, share of continuations that never
use that name again.) The teacher uses "the moral of the story" slightly more than real stories (5.4% of filtered
synthetic stories vs 3.6% of real seeds), and students trained on more synthetic data copy that habit a little. More
synthetic data → fewer unfinished stories (teacher stories always end). Every student forgets named characters in
about 40% of stories: a limit of a 5.3M model, not of the data.

**Blind review** (AI annotator, Claude, not a human; 20 test prompts × 5 students + the real continuations, 120
items shuffled with the system hidden; same rubric as the teacher pilot; `results/metrics/errors_review.json`):

| system | coherence (0–2) | relevance (0–2) | most frequent issues (of 20) |
|---|---|---|---|
| real continuations | **1.80** | 2.00 | logic error 4, grammar 2 |
| real_only | 0.10 | 0.90 | logic error 11, **repetition 10**, character switch 9 |
| real_fresh | 0.05 | 0.80 | logic error 14, character switch 11 |
| mix50_raw | 0.15 | 0.80 | character switch 14, logic error 12 |
| mix50_filt | 0.10 | 0.80 | logic error 12, character switch 12, repetition 5 |
| syn_only | 0.10 | 0.70 | character switch 15, logic error 10, off prompt 5 |

**All students are at the floor:** 90 of 100 student continuations were judged largely incoherent, against 0 of 20
real ones. A 5.3M-parameter model trained on 6M tokens writes grammatical sentences but loses the thread of the story
(characters swap names, events do not follow). Differences of 0.04 nats/token are real (Phase 13) but far too small to
be seen by a reader at this level. The only hint is that real-only repeats itself twice as often as the mixtures
(10 vs 5 of 20), which fits "it has seen the same 5,000 stories 6 times". 20 items per system is too few for more.

---
## Phase 16 — Efficiency and reproducibility

**Learn:** a result is only useful if someone else can repeat it and judge whether it is worth its cost. Report the
teacher and student costs separately and the gain *per unit of cost*.

**Files:** `synth/benchmark.py` → `results/metrics/benchmark.json`, `README.md`, `.gitignore`, `project9.html`

**Cost:**
- Teacher: 3 pools × ~1 CPU-hour; main pool 2.63M teacher tokens, 731 tokens/s, **13,951 accepted stories per
  generation hour** (2.48M accepted student tokens).
- Filter: about 6 s per pool without the judge, 165 s with it.
- Students: 27 runs, median 9.8 min, **4.9 CPU-hours** in total (+ 1.0 for the 10 pilot runs); evaluation about 25 s
  per run.
- **Value:** the main synthetic pool (1 teacher CPU-hour) buys −0.043 nats/token, i.e. 0.016 per million teacher
  tokens and 52% of the gain of 6M tokens of new real stories. When collecting more real data is impossible, an hour
  of teacher time is a good deal; when it is possible, real data is still better.

**Reproduce everything (CPU, about 10 hours):**
```
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.lock.txt
python scripts/download_scorer.py && python scripts/env_report.py
python -m synth.prepare_seeds
python -m synth.templates
bash scripts/generate_all.sh                          # 3 pools, about 3 h
python -m synth.parse_outputs --gen main              # and lowT, plain
python -m synth.filter_quality --calibrate
python -m synth.filter_quality --gen main             # and --gen lowT, --gen plain
python -m synth.dataset_report
bash scripts/lr_pilot.sh                              # + lr 4e-3 and 8e-3 (see Phase 12), about 1.7 h
bash scripts/train_all.sh                             # 27 students, about 5 h
bash scripts/eval_all.sh                              # about 15 min
python -m synth.compare
python -m synth.errors --stage auto && python -m synth.errors --stage sample
python -m synth.errors --stage count                  # after labelling results/error_review/blind.jsonl
python -m synth.benchmark
python -m synth.plots
pytest                                                # 16 tests
```
Requires Project 1 (`data/tokenizer.json`, `data/clean/`, `checkpoints/small/model_best.pt`) and Project 4's teacher
download (`checkpoints/scorer_tinystories33m`) next to this folder, read-only; hashes in `results/metrics/env.json`.
Every random choice has a fixed seed. Data, mixtures and checkpoints (about 2.6 GB) are not in the repository; the
manifests, metrics, samples and figures are.

## Limitations
- One task (children's stories), one teacher, one student size with a tuned learning rate, one 6M-token budget. With
  more compute the real-only student would start to memorise and synthetic data might help more.
- Ablations use one seed; the seed-pairing check was added after the pre-registration (it is stricter, not looser).
- The pilot review and the blind error review were done by an AI annotator (Claude), not by people.
- The scorer (SmolLM2-135M) rewards bland text; it is a secondary measure only.
