# SWEET.TV: temporal TV recommendation

Rustam Shangareev · Individual LLM4Rec final exam · 2026

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

OpenAI Codex assisted with task/source reading, design, implementation, debugging, experiments, automated verification and report drafting. External libraries implement ALS and boosting; EASE follows the cited derivation. No external solution notebook, pretrained model or extra dataset was used. Automated checks are not claimed as Rustam's personal verification. The separate original AI-session export is not published in this repository.
