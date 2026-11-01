"""Build a measured five-content-page report, plus a references page."""

from pathlib import Path
import json, html
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
OUT = ROOT / "output/pdf"
OUT.mkdir(parents=True, exist_ok=True)
font = Path("/System/Library/Fonts/Supplemental")
pdfmetrics.registerFont(TTFont("Arial", str(font / "Arial.ttf")))
pdfmetrics.registerFont(TTFont("ArialB", str(font / "Arial Bold.ttf")))
pdfmetrics.registerFontFamily(
    "Arial", normal="Arial", bold="ArialB", italic="Arial", boldItalic="ArialB"
)
body = ParagraphStyle(
    "body",
    fontName="Arial",
    fontSize=9.2,
    leading=11.8,
    spaceAfter=6,
    textColor=HexColor("#18222b"),
)
small = ParagraphStyle("small", parent=body, fontSize=8.1, leading=10.2, spaceAfter=4)
heading = ParagraphStyle(
    "heading",
    parent=body,
    fontName="ArialB",
    fontSize=12.5,
    leading=15,
    spaceBefore=8,
    spaceAfter=6,
    textColor=HexColor("#194f64"),
)
title = ParagraphStyle(
    "title", parent=heading, fontSize=20, leading=23, spaceBefore=0, spaceAfter=9
)
caption = ParagraphStyle("caption", parent=small, textColor=HexColor("#445360"))
J = lambda name: json.loads((A / name).read_text())
audit = J("audit-results.json")
dev = J("development-best.json")
frozen = J("frozen-selection.json")
final = J("final-results.json")
examples = J("final-examples.json")
rerank = J("reranker/selection.json")
publication = J("publication.json") if (A / "publication.json").exists() else {}
scoring = J("kaggle-score.json") if (A / "kaggle-score.json").exists() else {}
personal = (
    J("personal-verification.json")
    if (A / "personal-verification.json").exists()
    else {}
)
ready = bool(
    publication.get("repository_url")
    and scoring.get("private_score") is not None
    and personal
)
filename = "Rustam_Shangareev_exam_report" + ("" if ready else "_draft") + ".pdf"
story = []


def P(s, style=body):
    return Paragraph(s, style)


def add(s, style=body):
    story.append(P(s, style))


def H(s):
    add(s, heading)


def table(rows, widths, style=small):
    cells = [[P(str(v), style) for v in row] for row in rows]
    t = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e7f0f3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, HexColor("#45788a")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, HexColor("#d6dfe4")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 7))


labels = {
    "popularity": "Popularity",
    "repeat": "Repeat counts",
    "content": "Content-based",
    "als": "Implicit ALS",
    "association": "Association",
    "ppr": "PPR",
    "ease": "EASE",
    "channel": "Channel/hour",
    "simple_hybrid": "50:50 hybrid",
    "hgb": "Learned fusion",
}
order = list(labels)

# Page 1
add("Learning TV Habits from Sparse<br/>Broadcast Viewing Logs", title)
add("Rustam Shangareev · Individual · LLM4Rec final exam · 31 October 2026", small)
if publication.get("repository_url"):
    url = publication["repository_url"]
    add(
        'Code and designated submission: <link href="'
        + url
        + '" color="#146888">'
        + url
        + "</link>",
        small,
    )
else:
    add(
        "Implementation and submission are prepared locally; GitHub publication is pending.",
        small,
    )
add(
    "<b>Abstract.</b> Predicting television preferences requires recovering program interactions from channel-viewing intervals and an incompletely labelled schedule. I compare four course methods and four practical baselines, then select a learned fusion model using an earlier temporal development period. On a later untouched five-week audit, fusion achieves MAP@5 of <b>0.38922</b>, versus 0.27778 for popularity and 0.37724 for equal-weight repeat/channel fusion. The result supports combining recurring viewing habits with schedule availability, while target ties, incomplete labels and a longer test horizon limit the conclusions."
)
H("1. Introduction and related work")
add(
    "The task is to predict five available programs for each of 1,260 SWEET.TV users [1]. A news viewer may repeatedly watch the same bulletin; excluding all previously watched programs would erase useful candidates. The contribution is a reproducible interval join and a fair temporal comparison, followed by a ranker trained on earlier pseudo-future windows. Popularity and repeat counts test whether more complex models add value."
)
add(
    "The implicit ALS API [2] establishes confidence-weighted training semantics; missing interactions are not observed zero ratings. Steck [3, Sec. 3.1] supplies EASE’s constrained closed form as an additional alternative. The boosting API [4] supplies fitting and early-stopping controls; random internal early stopping is disabled to preserve the explicit temporal protocol. These sources justify mechanisms, not performance on this dataset."
)
H("2. Data and method")
add(
    "<b>Matching.</b> The log contains 5,821,987 rows from 4,914 users. Match equal channels and positive half-open interval overlaps; union a user’s overlapping segments within each airing, then divide seconds by scheduled duration. Target positives require at least 80% coverage, following [1]. Count qualifying airings per program and retain five highest counts; resolve unspecified frequency ties by ascending ID. The 396-second median and 342,145-second maximum make raw viewing duration an unsuitable universal completion threshold."
)
add(
    "<b>Coverage and time.</b> ID 0 never becomes an item, but its channel/time can inform habits. Only 10.762% of the first schedule’s <i>rows</i> carry an ID; this is not a measured airtime fraction. Kyiv timezone interpretation resolves 117 one-hour DST discrepancies and matches every supplied viewing duration. Final preparation yields 611,755 qualifying user-airing events. Titles/genres from future schedule rows never enter features."
)
add(
    "<b>Individual models.</b> CBF uses weighted category/genre/person/year/title tokens, past-only IDF, L2-normalized profiles and cosine (28-day decay). ALS uses 32 factors, 25 iterations, regularization 1 and observed confidence <i>1 + 10 log(1 + count)</i>. Association aggregates directional co-user weights with denominator shrinkage 1 and 100 neighbors. PPR uses cosine co-view edges, 100 neighbors, link probability 0.5, log-count seeds and seed-directed dangling redistribution. EASE uses log counts and ridge 10. Repeat counts use a development-selected 50% input threshold; labels stay at 80%. Channel/hour scores combine decayed past seconds (56-day half-life) with known future airing counts."
)
add(
    "<b>Fusion.</b> Candidate generation unions each of six models’ top 100 and 100 popularity items. A 49-feature binary ranker combines component scores/ranks, viewing counts/shares, recency, popularity and airing supply. Selected settings: 15 leaves, 250 boosting iterations, learning rate 0.05, L2 10, minimum leaf 50. It uses equal total training weight per user; user/item IDs are not features. All model scores are ranking signals, not satisfaction probabilities.",
    small,
)
story.append(PageBreak())

# Page 2
H("3. Experiments and verification")
add(
    "<b>Temporal design.</b> Development uses history before 18 May and targets 18 May–22 June 2020; the later audit uses history before 22 June and targets 22 June–27 July. The ranker’s three training target windows end by 18 May (23 March–27 April, 6 April–11 May, 13 April–18 May), each with past-only base fitting. These windows overlap. Development selects 48 individual configurations, 24 simple hybrids and six ranker settings. Choices and example users were frozen before the audit; no audit-driven retuning was performed."
)
add(
    "<b>Common evaluation.</b> Candidates are scheduled nonzero IDs: 4,279 development, 4,566 audit and exactly 6,636 in the final test file, including its overnight spill. Repeats are allowed. Every model uses descending score, then training user-popularity, then ascending ID. Users without an 80%-watched training airing receive identical popularity fallback. Boundary-crossing label airings are excluded. AP@5 sums precision at relevant ranks and divides by min(5, target count); MAP averages user AP. Primary cohorts match the official five-target condition. Secondary cohorts are retained rather than silently discarded."
)
rows = [
    [
        "Method",
        "Dev. MAP@5<br/>(775 users)",
        "Audit MAP@5<br/>(1,078 users)",
        "Audit fit +<br/>score, s",
    ]
]
for m in order:
    d = (
        dev[m]["map5_five_targets"]
        if m in dev
        else (
            frozen["simple_hybrid_development_map5"]
            if m == "simple_hybrid"
            else frozen["development_map5"]
        )
    )
    t = audit["timing_seconds"].get(m)
    ts = (
        f"{t:.2f}"
        if t is not None
        else (
            "component costs"
            if m == "simple_hybrid"
            else f"+{audit['timing_seconds']['hgb_features_and_inference']:.2f}*"
        )
    )
    rows.append(
        [labels[m], f"{d:.5f}", f"{audit['metrics'][m]['map5_five_targets']:.5f}", ts]
    )
table(rows, [160, 110, 120, 115])
add(
    "*Fusion cost is feature construction/inference after six component fits, not full end-to-end training. Selected boosting fit plus development inference took 53.53 s; pseudo-window feature preparation is additional. Hardware: arm64 Mac, 24 GiB RAM, 12 cores; Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, pandas 2.3.3, scikit-learn 1.6.1, implicit 0.7.3. Seed 20261031; one BLAS thread, four ALS threads. Full grids and timings are in the repository.",
    caption,
)
add(
    "<b>Evidence.</b> Fusion improves audit MAP by 0.11144 over popularity and 0.01197 over the simple hybrid. Paired user bootstrap (2,000 resamples) gives 95% percentile intervals [0.09861, 0.12409] and [0.00613, 0.01798], respectively; these describe this audit cohort, not causal benefit. Fusion MAP is 0.35941 on all 1,260 users and 0.36877 on 1,228 nonempty users. Six cold users in the five-target cohort score 0.18889 under the shared fallback."
)
add(
    "<b>Checks performed by the agent.</b> Ten unit tests cover metric arithmetic, candidate constraints, interval edge cases, duplicate union, DST, PPR mass/dangling behavior and future-content exclusion. A different endpoint-partition algorithm agrees on 200 sampled user-airing durations. Float64 ALS dot-product checks change scores by at most 5.72×10<super>-7</super> and preserve all 1,260 development Top-5 lists; final scoring uses that accumulation. All 1,260 final rows independently pass original-template order and official-test-candidate checks. A fresh base-model refit reproduces the same final CSV hash.",
    small,
)
add(
    "<b>Label sensitivity.</b> 785/1,078 audit users tie across fifth place. Fusion MAP becomes 0.33555 with reverse-ID ties and 0.35740–0.36083 across three random tie orders, remaining above the simple hybrid. Single-session 80% labels give 0.39004. Absolute MAP therefore depends materially on an organizer detail that is not specified.",
    small,
)
story.append(PageBreak())

# Pages 3-4: matrix and local title dictionary, no hidden file lookup needed.
catmap = {
    "Инфо": "News",
    "Развлечения": "Entertainment",
    "Познавательное": "Educational",
    "Фильмы": "Film",
    "Сериалы": "Series",
    "Другое": "Other",
    "Детям": "Children",
    "Спорт": "Sport",
    "Музыка": "Music",
}
for idx, u in enumerate(examples):
    H(f"4.{idx + 1}. Final recommendations: user " + u["user_id"])
    add(
        ("Light-history example" if idx == 0 else "Heavy-history example")
        + f": selected at the {'25th' if idx == 0 else '90th'} percentile of positive development-training counts, before inspecting audit outcomes. In the final training history this user has {u['qualifying_airings']} qualifying airings across {u['distinct_programs']} program IDs. The same two users are used for every method."
    )
    hist = "; ".join(
        f"{html.escape(p['title'])} [{p['id']}] ×{p['count']}"
        for p in u["training_top10"][:4]
    )
    add("<b>Most frequent past programs:</b> " + hist + ".", small)
    rows = [["Method", "Rank 1", "Rank 2", "Rank 3", "Rank 4", "Rank 5"]]
    for m in order:
        rows.append([labels[m]] + [str(p["id"]) for p in u["recommendations"][m]])
    table(rows, [115, 78, 78, 78, 78, 78])
    catalog = {p["id"]: p for ps in u["recommendations"].values() for p in ps}
    add(
        "Every ranked ID above is resolved below. Titles are representative schedule labels; a shared ID can carry different episode text across airings. Categories are translated for readability. The count is this user’s ≥80%-watched training airings.",
        caption,
    )
    rows = [["Program ID", "Schedule title", "Category", "Past count"]]
    for i, p in sorted(catalog.items()):
        rows.append(
            [
                i,
                html.escape(p["title"]),
                catmap.get(p["category"], html.escape(p["category"])),
                p["train_watch_count"],
            ]
        )
    table(rows, [73, 287, 100, 45])
    if idx == 0:
        add(
            "<b>Interpretation.</b> The selected list keeps “Факты. День.”, “Сегодня.”, “ТСН.” and “Вещдок”, then adds the already observed “Утро с Интером”. CBF instead extends a news profile toward “Деньги” and “Подробности недели”, with no qualifying history for this user. PPR retains the watched film “Воин”; association and popularity favor broadly co-viewed bulletins. These are different mechanisms, not evidence that either unfamiliar bulletin is a successful discovery.",
            small,
        )
    else:
        add(
            "<b>Interpretation.</b> The selected list preserves the strong “ТСН.”/“Завтрак с 1+1” habit and includes “Секретные материалы”. CBF concentrates on news and omits the second-most-watched breakfast program; broad categories lose that channel/routine distinction. PPR includes the watched series “Великолепный век”; ALS instead gives a high rank to “Утро с Интером”, while channel/hour introduces “Завтрак. Выходной” without a qualifying watch. Co-view methods can reflect cross-category habits that content profiles miss, but this user’s plausible ALS list does not erase ALS’s weak aggregate score.",
            small,
        )
    story.append(PageBreak())

# Page 5
H("5. Discussion and final strategy")
add(
    "I select learned fusion for the primary habitual-viewing recommendation list: it wins both temporal comparisons and uses identifiable repeat and channel/hour signals. On this scale, final fitting/scoring of its six components plus feature inference is measured in seconds, after the separate ranker-training stage. The simple hybrid is a reasonable operational fallback: its audit MAP is only 0.01197 lower and it avoids the supervised ranker and four auxiliary models. A production latency or retention advantage has not been measured."
)
add(
    "Both example users receive familiar programs under the chosen model. Across the complete final cohort, only 8.63% of its recommendations have no prior qualifying watch, compared with 34.51% for CBF. This supports a habitual-viewing interpretation, not a discovery claim. Popularity performs well because common bulletins recur; association also emphasizes co-watched common programs. PPR’s restart preserves personal seeds, whereas ALS compresses confidence-weighted preferences into low-rank vectors. ALS underperformed within the tested grid; this does not establish that matrix factorization is generally unsuitable."
)
add(
    "<b>Failure and fix.</b> Naïve timestamp subtraction initially disagreed with 117 log durations by exactly one hour. They cross spring DST; interpreting source times in Kyiv and converting to UTC removed all discrepancies, including local-hour assignment for channel features. A final contract check also exposed calendar-window versus broadcast-file spill: final candidates were corrected to the exact official test schedule. Neither correction used Kaggle labels. ALS emitted matrix-multiplication warnings despite finite factors; independent accumulation verified the output, without inventing a low-level root cause."
)
add(
    "<b>Limits and next step.</b> Sparse schedule labels hide most content; continuous playback is not proof of attention. Test recommendations span 12 weeks versus five locally, and target ties substantially affect MAP. Labels for overlapping early ranker-training windows are correlated. Future work should evaluate a genuinely longer untouched horizon and compare pairwise/listwise ranking losses, keeping a frozen audit. No live A/B test, satisfaction gain or revenue impact is claimed.",
    small,
)
H("6. AI usage and personal verification")
add(
    "OpenAI Codex assisted with source reading, design, code, debugging, experiments, automated checks and this draft. ALS and boosting use the cited libraries; no external solution notebook, pretrained model or extra dataset was used. The installed Codex model/build identifier is not independently established here. Agent-executed tests and source checks are explicitly distinguished from my personal verification. The original unmodified AI-session export must accompany the submitted report.",
    small,
)
if personal:
    add("<b>My check:</b> " + html.escape(personal["statement"]), small)
else:
    add(
        "<b>Personal verification status:</b> no completed student check is claimed in this draft. The requested AP@5 calculation and explanation of temporal leakage still need my own review before submission. I have not independently re-derived ALS, EASE or boosting.",
        small,
    )
H("7. Designated Kaggle submission")
if scoring:
    add(
        f"Private MAP@5: <b>{scoring['private_score']}</b>; public: {scoring.get('public_score', 'not recorded')}. File: final_hgb.csv (screenshot below). The frozen leaderboard position is not used.",
        small,
    )
    shot = A / scoring.get("screenshot", "kaggle-private-score.png")
    if shot.exists():
        from PIL import Image as PILImage

        w, h = PILImage.open(shot).size
        scale = min(350 / w, 220 / h)
        story.append(Image(str(shot), width=w * scale, height=h * scale))
else:
    add(
        "The validated file is ready; Kaggle upload is awaiting authorization. No returned score or screenshot is claimed yet.",
        small,
    )
add(
    "File: submissions/final_hgb.csv · 1,260 rows · five distinct official test IDs per row.<br/>SHA-256: "
    + final["submission"]["sha256"],
    caption,
)
story.append(PageBreak())
H("References")
refs = [
    '[1] Kaggle / ogurbych, “sweet.tv - TV program recommender,” Data description, 2020. <link href="https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data">https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data</link>. Accessed 2026-10-31. Target construction and user-wise private split checked.',
    '[2] Implicit, “AlternatingLeastSquares,” CPU API documentation. <link href="https://benfred.github.io/implicit/api/models/cpu/als.html">https://benfred.github.io/implicit/api/models/cpu/als.html</link>. Accessed 2026-10-31. Served documentation identifies 0.7.2; implementation uses installed 0.7.3. Sparse preference/confidence semantics checked.',
    '[3] H. Steck, “Embarrassingly Shallow Autoencoders for Sparse Data,” The Web Conference, 2019, arXiv:1905.03375. <link href="https://arxiv.org/html/1905.03375v1">https://arxiv.org/html/1905.03375v1</link>. Accessed 2026-10-31. Sections 2 and 3.1, including Eqs. (2)–(8), checked.',
    '[4] scikit-learn, “HistGradientBoostingClassifier,” version 1.6.1 API documentation. <link href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html">https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html</link>. Accessed 2026-10-31. Loss, sample weights and early-stopping controls checked.',
    '[5] S. Jin, “LLM4Rec - Homework Guidelines,” HSE University, official course repository. <link href="https://github.com/dryjins/RecSys-LLMs/blob/main/homework_guidelines.pdf">https://github.com/dryjins/RecSys-LLMs/blob/main/homework_guidelines.pdf</link>. Accessed 2026-10-31. Repository commit 4b9598b4e670417387aabe3facb7868b6b6143eb. Conflicting 5/6-page statements were resolved by following the stricter five-content-page limit.',
]
for r in refs:
    add(r)


def footer(c, d):
    c.setFont("Arial", 8)
    c.setFillColor(HexColor("#647780"))
    c.drawString(45, 25, "Rustam Shangareev · SWEET.TV recommendation")
    c.drawRightString(550, 25, str(d.page))


doc = SimpleDocTemplate(
    str(OUT / filename),
    pagesize=(595.276, 841.89),
    leftMargin=45,
    rightMargin=45,
    topMargin=38,
    bottomMargin=40,
    title="Learning TV Habits from Sparse Broadcast Viewing Logs",
    author="Rustam Shangareev",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT / filename)
