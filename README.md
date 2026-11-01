# SWEET.TV: temporal TV recommendation

Rustam Shangareev · Individual LLM4Rec final exam · 1 November 2026

**Designated final submission:** [`submissions/v6_discovery_blend.csv`](submissions/v6_discovery_blend.csv)

- Kaggle private MAP@5: **0.45170**; public: **0.45571**.
- 1,260 template users; five distinct official program IDs per user.
- SHA-256: `9a295e20a024ef074e3c388b00b64e2a7ecc83b62f2cc65278ce8b7d03061f89`.
- Actual returned score and screenshot: `artifacts/discovery-v6/kaggle-result.json` and `kaggle-result-detail.png`.
- The private difference from v5 (0.45164) is negligible. There were 19 actual submissions; validation and leaderboard feedback were used adaptively. This is the best observed submission, not an unbiased test estimate.

## Approach

Channel/interval matching recovers user-program events. Overlapping log segments are unioned per airing; at least 80% of its duration constitutes a full watch. Top-5 future full-watch frequency defines local targets, with ascending-ID ties. ID 0 is excluded from labels/candidates. Past channel context remains usable. Timezone-aware conversion through Europe/Kyiv fixes DST discrepancies.

Four course methods are evaluated individually: **content-based filtering, implicit ALS, pairwise association and Personalized PageRank**. Popularity, repeat counts, EASE and channel/hour models provide references. Every model shares temporal folds, scheduled candidates, repeat allowance and cold-user fallback. Future viewing behavior and future titles/genres are excluded from features. Future schedules supply availability/channel/time only.

| Method | Development MAP@5 | Later validation MAP@5 |
|---|---:|---:|
| Popularity | .26842 | .27778 |
| CBF | .28829 | .26345 |
| Implicit ALS | .19692 | .19481 |
| Association | .26902 | .26885 |
| PPR | .39455 | .36410 |
| Repeat/channel | .42027 | .37724 |
| Initial HGB | .43108 | .38922 |
| v2 CatBoost200 | .43836 | .39852 |
| v5 channel blend | .43909 | .39894 |
| v6 designated blend | .44042 | .39908 |

The final blend is `0.9 * (0.5 * RRF(v2) + 0.5 * RRF(new600)) + 0.1 * RRF(channel28)` with `RRF=1/(5+zero_based_rank)`. Preserve this float32 grouping for exact reproduction. v2 has 121 features; the newer CatBoost component has 167. Both use depth 6, learning rate .05, L2 10 and seed 20261031. User/item IDs are not tree features. v2 uses four fitting threads; new600 uses five.

Development predicts May 18–June 22, 2020; later validation June 22–July 27. Earlier supervised target windows end by the prediction cutoff. Final fitting adds completed development/later-validation labels, never test views. All local windows were eventually reused for tuning; the ten-week diagnostic overlaps them. Detailed chronological notes are preserved in [`docs/development-history.md`](docs/development-history.md), whose initial frozen-audit claims refer only to the first phase.

## Files

- `src/sweettv/`: preparation, individual algorithms, ranking and experimental variants.
- `tests/`: metric, interval, leakage, graph and feature checks.
- `scripts/`: experiments, verification and report generation.
- `artifacts/`: measured JSON/CSV results and score screenshots. Model/feature caches and source data are excluded.
- `submissions/`: designated final CSV and preserved experimental predictions.
- `third_party/`: external-code attribution and licenses.

## Reproduction

Tested on macOS ARM64, 24 GiB RAM, Python 3.9.6. Install the pinned dependencies (LightGBM needs `libomp` on macOS):

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[improvement,replay]'
pip install pytest==8.4.2
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
```

Accept the competition rules and download the original five CSVs into `data/raw/`: `dataset11-30.csv`, `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`, `export_arh_31-42-final.csv`, `submission.csv`. They are not redistributed here.

```sh
python -m sweettv.prepare
python -m sweettv.audit_data
pytest -q
```

The original course-method selection can be repeated in a separate working copy using `python -m sweettv.experiment`, `python -m sweettv.rerank`, `python -m sweettv.select`, then `python -m sweettv.finalize audit` and `python -m sweettv.finalize final`. Preserve committed measured artifacts first: the original audit script intentionally refuses silent overwrite. To rebuild only the fixed selected configuration from prepared data, without historical model binaries:

```sh
python scripts/reproduction/build_selected_from_processed.py
```

This generates missing feature caches, refits the exact 200/600-tree models and writes all three finalist CSVs under `output/session/from_processed/trained/`. It does not redo selection and does not depend on an audit-confirmation-generation step. The final-fold uncached preparation was separately checked against the original arrays; the complete six-fold uncached command has not been rerun in the final review. Both models were freshly trained from all five original historical feature caches, reproducing every final prediction exactly.

For that cached-data refit alone:

```sh
python scripts/reproduction/retrain_selected.py
```

The original 34-MB development chat and the separate new reproduction-session export are retained privately. They are not part of this public repository.

If the existing model/feature caches are available, these commands independently rebuild features or score saved models and verify exact submitted CSV bytes; **they do not retrain the final CatBoost rankers** (feature rebuilding can refit ALS):

```sh
python scripts/check_improvement_reproduction.py
python scripts/check_approaches.py
python scripts/check_discovery.py
python scripts/finalize_discovery.py
```

These checks passed, including 415,412 shared feature-pair comparisons, 11 recomputed discovery signals and all 1,260 final prediction rows. Tests: 29 passed, with 13 dependency deprecation warnings. The separate final review also executes selected-model training; its evidence is in `artifacts/final-review/`, separate from historical experiment records.

## Limitations and attribution

Unseen-program ranking remains weak: the v5 baseline retrieved 830/1,185 later-validation unseen targets but placed only 26 in Top-5; v6 places 28. New-to-user output does not establish discovery, satisfaction or never having watched a program. Only 10.762% of first-period schedule **rows** carry a program ID; this is not an airtime statistic. Unknown target tie-breaking and test-horizon mismatch materially affect local MAP.

Sources, actual transfer decisions and licenses are documented in `artifacts/source-checks.json`, `artifacts/improvement-v2/protocol-and-sources.json`, `artifacts/approaches-v5/public-source.json` and `third_party/`. Public H&M/OTTO/SWEET.TV solutions informed statistical features and candidate fusion. An eSASRec-author LiGR block was adapted for a small sequence experiment; it was not a benchmark reproduction. No external dataset or pretrained weights were used.

OpenAI Codex assisted with design, code, experiments, verification and report drafting. Rustam's personal check is limited to the supplied AP@5 calculation and temporal-leakage explanation (`artifacts/personal-verification.json`). Agent-executed tests and training are not attributed to him as manual verification. The separate unmodified reproduction-session log is not published to GitHub.

Optional PDF build dependencies: `pip install matplotlib==3.9.4 reportlab==4.4.9 pypdf==6.6.2`. The report builder uses the macOS Arial fonts; adapt font paths on another OS.
