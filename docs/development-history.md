# SWEET.TV: temporal TV recommendation

Rustam Shangareev · Individual LLM4Rec final exam · 2026

**Current best (post-submission experiments):** `submissions/v6_discovery_blend.csv` — Kaggle private MAP@5 **0.45170**, public **0.45571**, among 19 actual submissions. The previous v5 result was **0.45164 private / 0.45786 public**: the latest private increase is negligible and does not establish meaningful improvement. The initial PDF/package below has not been updated. Later adaptive studies and exact configurations are documented at the end of this README; `artifacts/discovery-v6/kaggle-result.json` records the latest score and file hash. The reported best is selected after repeated leaderboard observations, not an untouched test estimate.

This project joins viewing intervals to broadcast airings and ranks five available programs for each of the 1,260 template users. It compares content-based filtering, implicit matrix factorization, pairwise association rules and Personalized PageRank, plus popularity, repeat counts, EASE and a channel/hour predictor. A histogram gradient-boosting fusion model is selected on development data, then evaluated once on a later audit period.

## Measured results

| Method | Development MAP@5 | Later audit MAP@5 |
|---|---:|---:|
| Popularity | 0.26842 | 0.27778 |
| Repeat counts | 0.40077 | 0.36841 |
| Content-based | 0.28829 | 0.26345 |
| Implicit ALS | 0.19692 | 0.19481 |
| Association | 0.26902 | 0.26885 |
| Personalized PageRank | 0.39455 | 0.36410 |
| EASE | 0.38659 | 0.35690 |
| Channel/hour | 0.40759 | 0.36599 |
| 50:50 repeat/channel | 0.42027 | 0.37724 |
| Learned fusion (selected) | **0.43108** | **0.38922** |

Primary cohorts: 775 and 1,078 template users respectively, each with five distinct eligible targets. All models use identical cohorts, availability filtering, repeat policy and cold fallback. `artifacts/audit-results.json` also reports all 1,260 users (empty truth counts as zero), nonempty users, cold/warm users, runtime and paired bootstrap intervals. These local numbers are **not Kaggle scores**.

The local ground-truth tie rule is undocumented by the competition: 785/1,078 audit users have a frequency tie across fifth place. With reverse-ID ties, learned fusion scores 0.33555; three random tie orders give 0.35740–0.36083. It still outperforms the simple hybrid in every tested tie variant. Alternative single-session 80% labels give 0.39004. These sensitivity checks did not change model selection.

## Kaggle submission

The first submission, `submissions/final_hgb.csv`, returned **private MAP@5 0.43957** and public MAP@5 0.45015 on 31 October 2026. It was selected before seeing either score. The description identifies Рустам Шангареев / Rustam Shangareev. A copy downloaded from Kaggle is byte-identical to the local file (SHA-256 `05839015ca5bb3c9c0dcb0c5d914845c8a3ef5b95074af8e950b07ab06d09cc7`). Evidence: `artifacts/kaggle-score.json` and `artifacts/kaggle-private-score.png`. The upload form showed 100 daily submissions remaining before this first upload; the old competition accepts late submissions.

## Data and matching

Download the five original CSVs after accepting the [competition rules](https://www.kaggle.com/competitions/sweettv-tv-program-recommender). Put them in `data/raw/`. The data is not redistributed. Original hashes and sizes are recorded in `artifacts/data-manifest.json`.

- Source: 5,821,987 viewing rows, 4,914 users. User identifiers are strings, never floating-point numbers.
- Join on equal channel and positive overlap of half-open intervals. Union all a user's viewing segments within each airing before summing duration; do not double-count repeated or overlapping logs.
- Watch fraction = unioned overlap seconds / scheduled duration. A target-positive airing requires fraction ≥ 0.8, following the competition's target description. Ground truth is the five programs with the most qualifying airings; ties use numeric ID ascending.
- ID 0 is unlabelled airtime, never a candidate. Its viewing time can still inform channel/hour context. In the first schedule, 10.762% of **rows** have a nonzero ID; this is not a measured share of airtime.
- Source timestamps are interpreted in Europe/Kyiv, then stored in UTC. This is an inference: naïve subtraction disagrees with 117 supplied durations by one hour, all across spring DST; timezone-aware subtraction agrees with all 5,821,987 durations. No source timestamps are ambiguous/nonexistent under this interpretation.
- The log median is 396 seconds, versus a maximum of 342,145. A relative threshold prevents one absolute-seconds cutoff from meaning different things for short and long programs. Long sessions are retained and correctly intersected with each airing.
- Final processing produces 611,755 qualifying user-airing events. A single log, without unioning sessions, would qualify 562,933. The organizer does not fully specify session aggregation; sensitivity is reported.

## Temporal protocol

All boundaries below are local Kyiv midnight, start-inclusive/end-exclusive for target airing starts. Training airings must end by the cutoff; target labels require the whole airing to end within the target period. Boundary-crossing airings are excluded from labels.

| Purpose | Available training views | Target window |
|---|---|---|
| Development | Before 18 May 2020 | 18 May–22 June |
| Untouched later audit | Before 22 June 2020 | 22 June–27 July |
| Final Kaggle inference | Before 27 July 2020 | 27 July–19 October (official schedule, including overnight spill) |

The ranker is fitted only on three earlier pseudo-future windows: 23 March–27 April, 6 April–11 May, and 13 April–18 May. Each window has its own past-only base models. These training windows overlap, so training examples are correlated. Outer development labels select settings; audit labels never fit or tune any model. Final base models refit on all released past viewing, while the supervised ranker retains its original pre-18-May labels and frozen parameters.

Local candidates are every nonzero program ID airing in the target window; final candidates are exactly the distinct IDs in the supplied test schedule, including its final overnight spill: 4,279 development, 4,566 audit, 6,636 final. Previously watched programs remain eligible for every method. Future schedule **channel/time/availability** is allowed; future titles, genres and other attributes never enter features. For display only, examples may look up names from the schedule.

Cold users have no ≥80%-watched training airing and receive the same train-user-popularity ranking in every method. All lists contain exactly five distinct eligible nonzero IDs. Scores sort descending, then training popularity descending, then ID ascending. Negative MF scores remain valid scores; score ties use this common rule. Predicted scores are ranking signals, not calibrated enjoyment or purchase probabilities.

MAP@5 uses binary membership in the target set, with precision accumulated at each hit and divided by min(5, number of targets). The official test cohort has exactly five targets; therefore that cohort definition is primary locally. Empty-target users remain in the separately reported all-user metric.

## Models and selected settings

`src/sweettv/experiment.py` implements the individual methods:

- **Popularity:** number of distinct users with a qualifying training airing.
- **Repeat:** per-user program airing counts; development chose ≥50% watch, no decay. This weaker input threshold does not change the ≥80% label definition.
- **CBF:** one-hot categories/genres/actors/director/year plus title word tokens. Category weight 3, genres 2, other tokens 1. IDF uses historical metadata only, followed by L2 normalization; a weighted history profile scores items by cosine. Half-life 28 days; missing historical metadata produces a zero vector.
- **Implicit ALS:** 32 factors, regularization 1, 25 iterations, confidence 1 + 10 log(1 + qualifying count), seed 20261031. Unobserved entries have preference 0 and confidence 1 under the library's implicit objective; this is not an observed dislike or a zero rating. Four fitting threads, one BLAS thread. Prediction is the user/item dot product.
- **Association:** binary co-user item pairs; directional weight co-count/(antecedent users + 1), diagonal zero, top 100 outgoing neighbors. A log-count history aggregates outgoing weights. This is a pairwise rule recommender, not exhaustive Apriori mining of larger itemsets.
- **PPR:** cosine-normalized binary co-user edges, diagonal zero, top 100 outgoing neighbors, row-normalized transitions. Log-count history seeds the walk; alpha 0.5 follows links, the remaining mass restarts. Dangling mass returns to the seed. L1 residual threshold 1e-6, at most 200 iterations; convergence and mass are recorded.
- **EASE:** log-count input, ridge coefficient 10; invert the regularized Gram matrix, normalize each inverse column by its diagonal with a negative sign, set diagonal to zero. This follows Steck's closed form [3].
- **Channel/hour:** normalized past viewing seconds over channel × local hour, half-life 56 days, scored against known future airing counts in those contexts. Long intervals are split by elapsed hour, resolving local hour separately across DST. Overlapping source sessions can contribute multiple context seconds; only target watch fractions use unioning. The airing start hour approximates exposure, not exact probability of 80% completion.
- **Simple hybrid:** equally weighted per-user max-normalized nonnegative repeat and channel scores.
- **Learned fusion:** union top 100 candidates from repeat/CBF/ALS/EASE/channel/PPR plus top 100 popularity candidates, all intersected with the common candidate pool. No true target is injected into candidates. Candidate features include component scores/normalized scores/reciprocal ranks, counts and shares at thresholds 0.2/0.5/0.8 and half-lives none/14/28/56, history size, popularity, recency and airing supply. User and item IDs are not learned features. HistGradientBoostingClassifier uses 15 leaves, 250 iterations, learning rate .05, L2 10, minimum leaf 50, seed 20261031, and **early stopping disabled**. Sample weights equalize total weight per training user. Its output ranks generated candidates; it is not a calibrated user satisfaction probability.

Selection searches 48 individual configurations, 24 simple fusion settings and six boosting settings. See measured JSON files for the complete grid. Some methods share dimensions, but no private Kaggle score was used for model selection. A future study could use pairwise/listwise ranking losses and longer validation horizons; neither is claimed tested here.

## Reproduction

Tested: macOS arm64, 24 GiB RAM, 12 CPU cores, Python 3.9.6. Packages are pinned in `pyproject.toml` and `requirements.lock.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest==8.4.2
export OPENBLAS_NUM_THREADS=1
python -m sweettv.prepare
python -m sweettv.audit_data
pytest -q
```

To repeat selection and the one-time audit, use a **fresh checkout/output directory**. The committed audit is evidence; the audit command refuses to overwrite an existing audit silently. Preserve the original `artifacts/` in a separate copy before a complete rerun.

```bash
python -m sweettv.experiment
python -m sweettv.rerank
python -m sweettv.select
python -m sweettv.finalize audit
python -m sweettv.finalize final
```

`finalize final` writes `submissions/final_hgb.csv`; its SHA-256 is recorded in `artifacts/final-results.json`. The final ranker binary is intentionally not required to be downloaded: `rerank` recreates it from the original data. Raw data, prepared data and binary model/feature caches are excluded from Git.

For a new run, copy the source tree to a new directory **without the measured artifacts**, then copy `artifacts/data-manifest.json` as input provenance and put the five CSVs in `data/raw/`. The preparation script creates processed files; ensure `data/processed/` and `artifacts/` exist. No network calls occur during training.

## Verification and caveats

Ten unit tests cover AP arithmetic, duplicate predictions, empty cohorts, candidate filtering, interval boundaries/nesting, duplicate-log union, randomized interval matching, DST, PPR mass/dangling nodes, and future-content exclusion. Independently, 200 sampled user-airing unions were checked by a different endpoint-partition algorithm, with zero seconds difference. This is a sample check, not proof for every joined row.

An ALS scoring warning was investigated rather than suppressed: factors and scores were finite, float64 independent dot accumulation differed by at most 5.72e-7, and all 1,260 development Top-5 lists were unchanged. Final scoring uses explicit float64 accumulation; the exact low-level warning cause is not established.

The five-week local horizons are shorter than the 12-week test. Most schedule rows are unlabelled, target-frequency tie-breaking is unknown, metadata can vary across airings sharing an ID, and source devices can remain on without attention. Local scores do not imply user satisfaction, novelty does not prove relevance, and no online experiment or revenue lift is claimed.

## External sources and assistance

[1] Kaggle / ogurbych, SWEET.TV competition Data description (target construction and data), https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data, accessed 2026-10-31.

[2] Implicit, CPU AlternatingLeastSquares API (sparse confidence semantics), https://benfred.github.io/implicit/api/models/cpu/als.html, accessed 2026-10-31. Installed package 0.7.3; served docs identify 0.7.2.

[3] H. Steck, “Embarrassingly Shallow Autoencoders for Sparse Data,” The Web Conference, 2019. https://arxiv.org/html/1905.03375v1, accessed 2026-10-31. Sections 2 and 3.1, including zero diagonal and column-normalized inverse, were consulted.

[4] scikit-learn 1.6.1, HistGradientBoostingClassifier API, https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html, accessed 2026-10-31. Fitting, log loss, sample weights and early-stopping controls were checked.

[5] S. Jin, LLM4Rec Homework Guidelines, official course repository, https://github.com/dryjins/RecSys-LLMs/blob/main/homework_guidelines.pdf, accessed 2026-10-31. The source has conflicting 5/6-page statements; the report uses the stricter five content pages.

OpenAI Codex assisted with task/source reading, design, implementation, debugging, experiments, automated verification and report drafting. External libraries implement ALS and boosting; EASE follows the cited derivation. The initial submission used no external solution notebook, pretrained model or extra dataset. The later improvement phase inspects public winning notebooks and adapts a published neural block; see the attribution below. No external dataset or pretrained weights were added. Automated checks are not claimed as Rustam's personal verification. The separate original AI-session export is not published in this repository.

## Post-submission improvement experiments

The original frozen selection and audit above describe the **first** submission.
A later, explicitly requested improvement phase is stored in `artifacts/improvement-v2/`.
Its original audit window is now an already-open validation set, not an independent
final test. Kaggle public/private aggregate feedback is used adaptively; the
best observed Kaggle score is therefore not an unbiased generalization estimate.
No future user views or hidden labels are loaded. The original `final_hgb.csv`
is preserved unchanged; candidate files have `v2_` / `v3_` prefixes.

Install the optional dependencies with `pip install -e '.[improvement]'`.
On Apple Silicon, LightGBM also needs `brew install libomp`. Sequence experiments
use PyTorch MPS when available. Run from this directory, using the same processed
data as the original experiment:

```sh
PYTHONPATH=src .venv/bin/python -m sweettv.improve train --fold dev
PYTHONPATH=src .venv/bin/python -m sweettv.improve train --fold audit
PYTHONPATH=src .venv/bin/python -m sweettv.rankmap --fold dev
PYTHONPATH=src .venv/bin/python -m sweettv.rankmap --fold audit
PYTHONPATH=src .venv/bin/python scripts/improvement_ablation.py
PYTHONPATH=src .venv/bin/python -m sweettv.improve_final
PYTHONPATH=src .venv/bin/python scripts/improvement_xgb_final.py
PYTHONPATH=src .venv/bin/python -m sweettv.sequence --fold dev
PYTHONPATH=src .venv/bin/python -m sweettv.sequence --fold audit
PYTHONPATH=src .venv/bin/python -m sweettv.sequence --fold final
PYTHONPATH=src .venv/bin/python -m sweettv.blend
PYTHONPATH=src .venv/bin/python -m sweettv.allusers
.venv/bin/python scripts/check_improved_submissions.py
```

The fixed v2 feature schema has 121 columns: original six retrieval signals,
BM25 user neighbors, shrunk cosine item neighbors, recent interaction counts,
completion fractions, airing-normalized rates, weekly future availability and
channel affinity. The expanded retrieval pool is shared by every new reranker.
LightGBM binary classification and LambdaRank, CatBoost classification, and
XGBoost MAP-oriented LambdaMART are compared. Supervised labels always end before
the prediction cutoff. Final training adds the two completed validation windows;
this is allowed refitting, not a claim that those windows remain unseen.

`allusers.py` additionally learns from non-template users in the supplied history;
the evaluated cohort and candidate policy remain unchanged. Early training windows
overlap, so their observations are correlated. No user identity is a tree feature.

Sources and transfer decisions are recorded in `protocol-and-sources.json`.
The H&M first-place write-up informed recent/cumulative interaction features and
boosted ranking. OTTO third-place notebooks were inspected for candidate coverage,
feature aggregation and fusion; their RAPIDS pipeline was not copied or executed.
The small sequence model adapts LiGR blocks from the **eSASRec (RecSys 2025)**
authors' implementation, with its Apache-2.0 license bundled under `third_party/`.
It is a task-specific adaptation, not a reproduction of the paper's scores or a
claim of state-of-the-art accuracy on this dataset. Its next-event training target
differs from this competition's multi-week Top-5 frequency target.

Measured comparisons, negative results, feature ablations, error cohorts,
submission hashes and actual returned scores are retained in the improvement
artifact folder. The older PDF has deliberately not been updated during this
model-development phase.

Additional diagnostics use `python -m sweettv.horizon` (ten-week validation and
five-week-equivalent exposure scaling), `python -m sweettv.countmodel` (Poisson
counts and log-count RMSE), and `scripts/improvement_neural_blend.py` (fixed 90/10
CatBoost/eSASRec rank fusion). Prefix commands with `PYTHONPATH=src` as above.
The long diagnostic overlaps the existing validation windows and is **not** an
independent holdout. `scripts/summarize_improvements.py` compiles actual stored
results; `kaggle-submissions.json` records observed scores and exact file hashes.

## Local training study (v3)

`artifacts/deep-training-v3/` contains a subsequent training study, separate from
the submission search. `protocol.json` records temporal boundaries, model grids,
adaptive extensions and source attribution. Each run retains its configuration,
training curve, measured scores and model checkpoint locally. Large model/data
files are excluded from Git. This phase does not refresh the PDF.

The neural reranker learns on 3,798,323 candidate pairs grouped into 11,017
historical user-window queries. It compares listwise softmax, balanced binary
classification and LambdaRank-style gradients weighted by exact AP@5 swap gains.
The latter uses all retrieved positives and the top 20 hard negatives; a
brute-force test checks the swap weights. Every loss masks padding and weights
queries equally. Training-only normalization, AdamW, a cosine learning-rate
schedule, gradient clipping, early stopping and checkpoint/RNG saving are used.

The multi-week sequence model predicts future Top-5 programs from histories of
100 events, using full-vocabulary softmax and rolling five-week targets. It
compares 64/128/256-dimensional representations, repeat-count priors and dropout.
The sequence input uses fully viewed events; the tabular reranker also uses
partial-view/channel features, so this is a comparison of complete pipelines,
not an architecture-only experiment. The old next-event model additionally had
a shorter context and different size: improvement over it cannot be attributed
solely to the new loss. The LiGR attribution/license above still applies.

Tree experiments include MAP-oriented YetiRank and LightGBM's `rank_xendcg`
(XE_NDCG_MART). Checkpoints are selected with the full-ground-truth AP denominator
of five. Library ranking metrics computed only over retrieved positive candidates
can be optimistic and are not used as the final reported metric.

```sh
PYTHONPATH=src .venv/bin/python -m sweettv.neural_rank --config list128
PYTHONPATH=src .venv/bin/python -m sweettv.neural_rank --config list256
PYTHONPATH=src .venv/bin/python -m sweettv.neural_rank --config drop128
PYTHONPATH=src .venv/bin/python -m sweettv.neural_rank --config bce128
PYTHONPATH=src .venv/bin/python -m sweettv.neural_rank --config map128
PYTHONPATH=src .venv/bin/python scripts/confirm_neural_rank.py
PYTHONPATH=src .venv/bin/python -m sweettv.temporal_sequence --config seq128
PYTHONPATH=src .venv/bin/python -m sweettv.temporal_sequence --config seq128prior
PYTHONPATH=src .venv/bin/python -m sweettv.temporal_sequence --config seq256prior
PYTHONPATH=src .venv/bin/python -m sweettv.temporal_sequence --config seq64prior
PYTHONPATH=src .venv/bin/python -m sweettv.temporal_sequence --config seq128reg
PYTHONPATH=src .venv/bin/python -m sweettv.yeti_rank
PYTHONPATH=src .venv/bin/python -m sweettv.xendcg
PYTHONPATH=src .venv/bin/python -m sweettv.cat_replicate
PYTHONPATH=src .venv/bin/python scripts/summarize_deep_training.py
PYTHONPATH=src .venv/bin/python scripts/analyze_deep_errors.py
```

The neural confirmation script selects architecture, epoch budget and blending
weight on the earlier window, then transfers those settings to the later window
and repeats three seeds. The later window was already inspected in prior work;
it must not be described as an untouched test. All supervised training labels
still end before the corresponding prediction cutoff. Overlapping training
windows are correlated. Runtime logs can overlap other local training jobs and
are not a dedicated hardware benchmark. Negative results are retained.

The completed study contains **25 training runs and 390 neural epochs**. The
selected XE_NDCG_MART/CatBoost rank fusion improved early/later validation MAP@5
from 0.43836/0.39852 to 0.44154/0.40135. Paired bootstrap intervals include zero;
these reused validation windows do not establish a significant improvement.
The final XE_NDCG_MART refit used 5,881,656 pairs in 17,188 historical queries.
Only one new candidate was submitted to Kaggle: `trained_xendcg1200.csv` scored
**0.44735 private / 0.46151 public**. It did not beat the existing best,
`v2_cat200_horizon.csv` (**0.45024 private / 0.45844 public**), which is retained.
See `completion.json`, `training-runs.csv`, `xendcg-confirmation.json`,
`submission-verification.json` and `kaggle-result.png` in the study directory.

For the selected candidate, `scripts/confirm_xendcg.py select` freezes the early
validation choice; `scripts/confirm_xendcg.py evaluate` measures later and longer
windows. After the final refit, `scripts/finalize_trained_xendcg.py` writes the
CSV and `scripts/check_trained_submission.py` independently reloads both models
and checks byte-identical predictions and official candidate eligibility. Run
these with the same `PYTHONPATH=src .venv/bin/python` prefix as above. Model
checkpoints and cached datasets must be present (or regenerated) for these steps.

## Error-driven cycles (v4)

`artifacts/error-cycles-v4/experiments.csv` records the next local study. Starting
from the best CatBoost model, it tests completion/duration calibration, clock
and weekday context, channel-conditioned repeat forecasts, longer historical
targets, categorical program identity, feature removal, training recency,
hard-negative reranking, recent full-view neighbors and separate category
affinity. Failed hypotheses are retained. `baseline-errors.json` and
`selected-error-changes.json` distinguish known, partially watched and unseen
targets. Diagnostic ranks include the production cold-user fallback and ties.

Run these entry points with `PYTHONPATH=src .venv/bin/python`: the module
`-m sweettv.error_cycles --stage N`, and scripts `error_cycle_horizon.py`,
`error_cycle_identity.py`, `error_cycle_regularize.py`, `error_cycle_hardneg.py`,
`error_cycle_calibrate.py`, and `error_cycle_ensemble.py`. Stages 1/2/3/4/5/6
identify feature groups; stages 13–16 isolate one feature group while preserving
the original five-week feature scale. Stage 0 is the exposure-normalization
control. Nondefault configurations and validation windows are recorded with
each result; `--fold audit` and `--fold final` refresh training only with labels
that have already ended at the corresponding prediction time.

The selected candidates combine the incumbent with 400-tree CatBoost using
categorical program IDs, at weights 75% and 25% for the new component. Both
weights were frozen in `submission-selection.json` before Kaggle evaluation.
`scripts/finalize_error_cycles.py export` reloads both fitted models, checks
their scores against stored predictions and validates each CSV against the
raw official test schedule. All local windows have been reused for selection;
the small validation differences are not statistically conclusive.

Kaggle returned **0.44825 private / 0.46116 public** for the 75% blend and
**0.44902 private / 0.45955 public** for the 25% blend. Neither improved the
incumbent private score of **0.45024**; `v2_cat200_horizon.csv` remains the best
submission. `kaggle-results.json` records hashes and actual scores, with UI
evidence in `kaggle-ui-observation.json` and `kaggle-result-detail.png`. This
study produced 21 retained fitted checkpoints, 42 evaluation records and two
Kaggle submissions. The complete local test suite passes 23 tests.

## Simple alternatives and schedule replay (v5)

The next study in `artifacts/approaches-v5/` compares 92 standalone configurations:
32 repeat-count/duration/completion variants, 20 channel projections, 12 recent
neighbor configurations, 16 clock-session replay variants and 12 adaptations of
a public competition formula. Including 276 nontrivial blends with the incumbent,
the development grid contains 368 ranking evaluations. These are parameter
configurations, not 92 independently trained models. A separate 400-tree CatBoost
fit uses 133 features and 1,164,190 historical pairs; its 200/400-tree checkpoints
fail to improve the early window and are not submitted.

Channel projection weights each channel by the user's share of viewing minutes
in the previous 28 days, then multiplies by future program airing counts. Replay
projects historical session intervals onto the known future channel schedule in
Kyiv civil time, with daily or weekly repetition. Only past sessions are used;
future schedule metadata is allowed. Replay uses raw session templates without
unioning and caps intervals at the replay period. Its scores are heuristics, not
calibrated probabilities. Midnight wrapping and DST are covered by tests.

The public reference is [Gornyaki_final_solution, version 10](https://www.kaggle.com/code/kinahem/gornyaki-final-solution)
by Myhaylo Turkin and Alex Tsepa (Apache-2.0). Its prepared input datasets are
private, so this is an adaptation of its completion/recency/airing-count formula,
not an exact reproduction or a verified winning solution. The notebook was read,
not executed, and its prediction outputs were not reused. Its custom positional
metric is not used: every experiment here uses the existing tested standard AP@5.
Source attribution and the downloaded notebook's hash are in `public-source.json`.

| Frozen candidate | Early MAP@5 | Later MAP@5 | Ten-week MAP@5 | Kaggle private | Kaggle public |
|---|---:|---:|---:|---:|---:|
| Previous incumbent | 0.438357 | 0.398519 | 0.440968 | 0.45024 | 0.45844 |
| 10% channel projection | 0.439088 | 0.398942 | 0.441020 | **0.45164** | 0.45786 |
| 25% session replay | 0.440065 | 0.398828 | 0.440577 | 0.44967 | 0.45822 |

Both weights were frozen before either Kaggle upload. Channel projection improves
all three local windows slightly; replay improves the two shorter windows but
declines on the longer window and on Kaggle. Other families fail confirmation.
All windows have been reused for development, and the long window overlaps the
shorter windows. Paired bootstrap intervals include zero. The private gain of
0.00140 is an observed score increase, not evidence of statistical significance.

The later-window channel blend improves AP for 28 users, worsens it for 22 and
leaves 1,028 unchanged. Hits on partially watched targets rise from 79 to 86;
hits on completely unseen targets rise only from 25 to 26 out of 1,185. Discovery
of unseen programs remains the main weakness. `selected-error-changes.json`
contains all three windows and bootstrap intervals; `experiments.csv` retains
all development configurations and four trained-checkpoint evaluation records.

Reproduction (after generating the v2 score caches and installing the optional
`replay` dependency):

```sh
PYTHONPATH=src .venv/bin/python -m sweettv.simple_search select
PYTHONPATH=src .venv/bin/python -m sweettv.simple_search confirm
PYTHONPATH=src .venv/bin/python -m sweettv.session_replay select
PYTHONPATH=src .venv/bin/python -m sweettv.session_replay confirm
PYTHONPATH=src .venv/bin/python -m sweettv.public_formula select
PYTHONPATH=src .venv/bin/python -m sweettv.public_formula confirm
PYTHONPATH=src .venv/bin/python -m sweettv.replay_ranker --fold dev
PYTHONPATH=src .venv/bin/python scripts/finalize_approaches.py
PYTHONPATH=src .venv/bin/python scripts/check_approaches.py
PYTHONPATH=src .venv/bin/python scripts/summarize_approaches.py
```

`finalize_approaches.py` writes a fresh selection timestamp; preserve the original
`submission-selection.json` when reproducing historical decisions. The independent
check recomputes new signals from raw history and confirms byte-identical CSVs,
the exact 1,260 users and five distinct valid IDs per user. The local suite passes
26 tests. Actual Kaggle observations, hashes and the best-score screenshot are
stored in this study directory; `best-submission.json` identifies the new best.

## Investigating new-to-user errors (v6)

The later validation window contains 1,185 Top-5 targets with no earlier matched
overlap for that user. The incumbent retrieves 830 into its candidate union and
puts only 26 in Top-5. Their median rank is 84. Of these targets, 561 have one
qualifying future view, 295 are films, and 128 have no full-view history across
any training user. These are different limitations: candidate coverage, ranking,
thin evidence and catalog cold start. They are not interchangeable causes.

`artifacts/discovery-v6/` records five interventions:

1. A separate unseen-item classifier using 3,310,631 historical pairs, including
   other supplied users, plus direct unseen-score calibration.
2. Expanded retrieval and 167 features: separate genre/actor/title-family
   similarities, long/recent raw-channel context, audience growth and historical
   ID-zero title-family evidence. The context join yields 104,425 overlapping
   user-airing records, 12,443 with at least 80% coverage. These remain features;
   no ID-zero airing becomes a labelled item. Family mappings are restricted to
   metadata observed before each prediction cutoff. No unseen validation target
   had a full ID-zero family match, so this particular recovery hypothesis failed.
3. A separate unseen-item classifier with the expanded features.
4. Fractional training labels for equal-frequency ties at the fifth position.
   Validation labels and the evaluation metric remain unchanged.
5. Directed associations from past histories to later new-to-user programs,
   learned only from completed historical windows.

The study fits six boosting models, including the audit and final refits, and
retains 190 evaluation records. A standalone discovery model did not yield a
large gain. The selected configuration uses 45% new 600-tree CatBoost ranking,
45% previous CatBoost ranking and 10% recent-channel ranking, through reciprocal
ranks with offset 5. It was selected on early validation and transferred unchanged.
Preserve the exact float32 arithmetic order in `finalize_discovery.py`, since
algebraically regrouping the fusion can change exact rank ties.

| Window | Incumbent MAP@5 | Selected MAP@5 | Unseen hits before | Unseen hits after |
|---|---:|---:|---:|---:|
| Early | 0.439088 | 0.440422 | 14 / 755 | 13 / 755 |
| Later | 0.398942 | 0.399079 | 26 / 1,185 | 28 / 1,185 |
| Ten-week | 0.441020 | 0.441594 | 51 / 1,231 | 54 / 1,231 |

All paired bootstrap intervals include zero. These are reused, overlapping
validation windows. On the early window, removing all previously watched items
still lets the unseen-item specialist retrieve only 149 of 755 new targets in
its five-item lists, versus 144 for the original scorer. Therefore, low final
exposure alone does not explain the failure: discrimination among new items is
also weak. This is a diagnostic of these models, not a proven quality ceiling.

Only the selected blend was submitted. Its final refit uses 2,584,719 pairs.
Kaggle returned **0.45170 private / 0.45571 public**, compared with the incumbent's
**0.45164 / 0.45786**. The hoped-for jump to 0.50 was not achieved. The minimal
private increase should not be described as solving the discovery problem.

Run the following with `PYTHONPATH=src .venv/bin/python`: modules
`sweettv.novel_specialist --allusers`, `sweettv.discovery_ranker`,
`sweettv.discovery_ranker --mode novel`, `sweettv.tie_training`, and
`sweettv.temporal_discovery`. Then run `scripts/discovery_select.py`.
The selected joint model supports `--fold audit` and `--fold final`; only labels
ending before the corresponding cutoff are used for each refit. The final model
has no local test metric because Kaggle labels are hidden.

`scripts/finalize_discovery.py` reloads both models before exporting and checks
exact predictions against stored scores. `scripts/check_discovery.py` recomputes
all 11 new signals, compares 415,412 shared feature pairs with the independently
stored original feature cache, and verifies all 1,260 prediction rows and official
candidate IDs. `scripts/summarize_discovery.py` recreates the error comparisons
and experiment ledger. The suite passes 29 tests, including temporal exclusion
for ID-zero context and correct probability allocation for tied training targets.
