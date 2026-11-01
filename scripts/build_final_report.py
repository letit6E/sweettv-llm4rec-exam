"""Final measured report: five content pages and separate references."""
from pathlib import Path
import json, html, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1]; A=ROOT/'artifacts'; OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
for name, fn in [('Arial','Arial.ttf'),('ArialB','Arial Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,'/System/Library/Fonts/Supplemental/'+fn))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialB',italic='Arial',boldItalic='ArialB')
body=ParagraphStyle('body',fontName='Arial',fontSize=9.6,leading=12.3,spaceAfter=5,textColor=HexColor('#172937'))
small=ParagraphStyle('small',parent=body,fontSize=8.5,leading=10.7,spaceAfter=4)
cap=ParagraphStyle('cap',parent=small,fontSize=7.3,leading=9,textColor=HexColor('#4c606b'))
head=ParagraphStyle('head',parent=body,fontName='ArialB',fontSize=12,leading=14,spaceBefore=7,spaceAfter=5,textColor=HexColor('#16566b'))
title=ParagraphStyle('title',parent=head,fontSize=19,leading=22,spaceBefore=0,spaceAfter=8)
J=lambda p:json.loads((A/p).read_text()); story=[]; markdown=[]
def add(s,style=body):story.append(Paragraph(s,style));markdown.append(re.sub('<[^>]+>','',html.unescape(s)))
def H(s):add(s,head)
def table(rows,widths,style=small):
 t=Table([[Paragraph(str(v),style) for v in r] for r in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e8f1f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,HexColor('#42788b')),('LINEBELOW',(0,1),(-1,-1),.2,HexColor('#d7e1e5')),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]));story.append(t);story.append(Spacer(1,5))
 markdown.extend([' | '.join(re.sub('<[^>]+>','',str(v)) for v in r) for r in rows])
def page():story.append(PageBreak())
def mathline(expr,name,width):
 path=ROOT/'tmp/pdfs'/name
 fig=plt.figure(figsize=(7,.45));fig.text(.5,.5,'$'+expr+'$',ha='center',va='center',fontsize=15);fig.savefig(path,dpi=200,bbox_inches='tight',pad_inches=.03,transparent=True);plt.close(fig)
 from PIL import Image as PI
 w,h=PI.open(path).size;story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,3))

labels={'popularity':'Popularity','repeat':'Repeat','content':'CBF','als':'Implicit ALS','association':'Association','ppr':'PPR','ease':'EASE','channel':'Channel/hour','simple_hybrid':'Repeat/channel','hgb':'Initial HGB','v2':'v2 Cat200','v5':'v5 channel blend','v6':'v6 final blend'}
publication=J('publication.json') if (A/'publication.json').exists() else {}
url=publication.get('repository_url','https://github.com/letit6E/sweettv-llm4rec-exam')
add('Learning Recurring TV Habits<br/>from Incomplete Viewing Logs',title)
add('Rustam Shangareev · Individual · LLM4Rec final exam · 1 November 2026',small)
add(f'Code and final CSV: <link href="{url}" color="#12627b">{url}</link>',small)
add('<b>Abstract.</b> Television recommendation requires recovering program interactions from viewing intervals and a schedule with many missing program IDs. I compare four course methods with popularity and habitual-viewing baselines, then combine temporally trained CatBoost rankers with a channel-affinity score. The designated submission achieves private MAP@5 of <b>0.45170</b>, while its later local validation score is 0.39908 versus 0.27778 for popularity. Recurring habits explain most of the gain; unseen-program ranking remains weak, and repeated validation and leaderboard selection limit generalization claims.')
H('1. Problem, references and alternatives')
add('The task is to rank five scheduled programs for each of 1,260 SWEET.TV users [1]. Recommending a familiar news bulletin is useful for this frequency-based target, so I allow repeated programs. My contributions are an explicit interval-to-program matching rule, a shared temporal comparison, and error-driven refinement of a repeat-sensitive ranker. The objective is MAP@5, not viewing enjoyment or revenue.')
add('The implicit ALS API [2] defines confidence-weighted feedback, and Steck [3, Sec. 3.1] provides the EASE closed form. CatBoost [4] supplies the final binary classifiers. Public H&amp;M and OTTO solutions [5] motivated multiple retrieval sources and recent interaction features; a SWEET.TV notebook [6] motivated channel-frequency and session-replay alternatives. Their reported scores are not reproduced or treated as our results. A small adapted sequence model [7] was also trained; no pretrained weights or external dataset were used.',small)
H('2. Method and design decisions')
add('<b>Data and matching.</b> The supplied log has 5,821,987 rows and 4,914 users. Match the channel and positive half-open interval overlap; union a user’s overlapping segments within each airing and divide by scheduled duration. At least 80% coverage defines a full watch [1]. Count full airings per program and take the five largest counts, breaking unspecified frequency ties by ascending ID. Preparation yields 611,755 full user-airing events. Only 10.762% of the first schedule’s <i>rows</i> have nonzero IDs; this is not an airtime estimate.')
add('<b>Missing values and leakage.</b> ID 0 is never a target/candidate, but its observed channel/time can describe habits. Past zero-ID title families were tested as context; they did not recover unseen target history in the examined folds. Kyiv timezone conversion resolves DST. Every fold uses behavior strictly before its cutoff; supervised labels must finish before that cutoff. Future schedules provide availability, channel and time only. Future titles/genres are excluded from features; titles may be looked up later to label the displayed recommendations.')
add('<b>Individually evaluated methods.</b> CBF uses weighted metadata/title tokens, historical IDF, cosine and 28-day decay. ALS uses 32 factors, 25 iterations, regularization 1 and confidence 1 + 10 log(1 + count); missing feedback is not an observed dislike. Pairwise association uses co-user counts with denominator shrinkage 1 and 100 outgoing neighbors. PPR uses cosine co-view edges, 100 neighbors, link probability 0.5, log-count seeds and seed-directed dangling redistribution. EASE uses log counts and ridge 10. Repeat input uses 50% coverage while labels remain 80%; channel/hour uses past seconds with 56-day decay.',small)
add('<b>Final ranking.</b> v2 learns 121 retrieval/count/recency/exposure features with CatBoost (200 trees). v6 adds contextual/content signals for 167 features (600 trees). Both use depth 6, learning rate 0.05, L2 10 and seed 20261031; IDs are not tree features. v2 retrieval combines eight signals plus popularity; v6 adds 11 context/content signals. Both remain restricted to the common scheduled pool. Final rank fusion is 90% of the equally weighted v2/v6 rank scores plus 10% channel score. Each rank score is 1/(5 + r), with zero-based rank r; preserve the implementation’s float32 grouping when reproducing ties. Channel scores combine normalized past-28-day minutes with future airing counts. These are ranking scores, not probabilities of enjoyment.',small)
page()
H('3. Experiments: temporal comparison')
add('<b>Splits.</b> Development predicts 18 May–22 June 2020; later validation predicts 22 June–27 July. Earlier supervised target windows are 23 March–27 April, 6 April–11 May and 13 April–18 May. Later fitting adds completed development labels; final fitting adds completed later-validation labels, using only pre-27-July behavior. A ten-week diagnostic predicts 18 May–27 July and overlaps the shorter windows. All these windows were eventually reused during adaptive tuning: none is now an untouched holdout.',small)
add('<b>Shared protocol.</b> Candidates are scheduled nonzero IDs: 4,279 development, 4,566 later validation and 6,636 official test IDs, including file spill past midnight. All methods allow repeats, use the same 1,260 evaluation users, and fall back to training user-popularity for users with no full training watch (4 in the final cohort). Score ties use training popularity, then ascending ID. Primary MAP@5 uses users with five eligible targets: 775 and 1,078 locally; all-user secondary metrics remain in the artifacts. AP sums precision at relevant ranks and divides by min(5, target count).',small)
d=J('development-best.json');a=J('audit-results.json');f=J('frozen-selection.json')
rows=[['Method','Dev. MAP@5','Later MAP@5','Role']]
for m in list(labels)[:10]:
 dv=d[m]['map5_five_targets'] if m in d else (f['simple_hybrid_development_map5'] if m=='simple_hybrid' else f['development_map5'])
 rows.append([labels[m],f'{dv:.5f}',f"{a['metrics'][m]['map5_five_targets']:.5f}",'Course method' if m in ['content','als','association','ppr'] else 'Reference / fusion'])
for n,v,w,r in [('v2',.438357,.398519,'121-feature ranker'),('v5',.439088,.398942,'v2 + channel'),('v6',.440422,.399079,'Designated blend')]:rows.append([labels[n],f'{v:.5f}',f'{w:.5f}',r])
rows.append(['New Cat600 alone','0.43701','0.39136','167-feature component'])
table(rows,[160,95,95,155])
add('<b>What the comparison establishes.</b> Final fusion exceeds popularity by 0.12130 on later validation. The new 600-tree component alone is worse than v2, so its value is complementary rank fusion, not superior standalone accuracy. v5 and v6 differ by only 0.00014 locally; paired bootstrap intervals for their differences include zero on all three temporal windows. The ten-week v2/v5/v6 scores are 0.44097/0.44102/0.44159. No statistically established improvement is claimed for v6.',small)
table([['Designated variant','Kaggle private','Kaggle public'],['v2 Cat200','0.45024','0.45844'],['v5 channel blend','0.45164','0.45786'],['v6 final blend','0.45170','0.45571']],[215,145,145])
add('<b>Compute and reproducibility.</b> ARM64 Mac, 24 GiB RAM, 12 CPU cores; Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, pandas 2.3.3, scikit-learn 1.6.1, implicit 0.7.3, CatBoost 1.2.10. Seed 20261031, one BLAS thread, four ALS / four v2 CatBoost / five v6 CatBoost threads. Initial individual final-fit/scoring costs ranged from 0.03 s (popularity) to 2.78 s (channel/hour). The new final CatBoost fit plus checkpoint evaluation took about 192 s for 2,584,719 candidate pairs and 24,413 positives; feature preparation and older-model fitting are additional, not included in that timing.',small)
add('<b>Verification.</b> The agent ran 29 passing tests covering AP, duplicate predictions, temporal bounds, overlap union, DST, PPR, future-content exclusion and new context signals. A distinct endpoint-partition algorithm matched 200 sampled airing unions. Independent feature rebuilding/reloaded scoring reproduces the submitted CSVs, checks 415,412 shared feature pairs, and validates all 1,260 output rows against the original template and test schedule. A separate session also freshly refit both selected CatBoost models (38.32 s and 169.24 s) from historical feature caches: all three finalist CSVs were byte-identical, with zero changed positions. This is model retraining, not a complete raw-data rebuild. Agent checks are distinguished from my hand calculation on page 5.',small)
page()
catmap={'Инфо':'News','Развлечения':'Entertainment','Познавательное':'Educational','Фильмы':'Film','Сериалы':'Series','Другое':'Other','Детям':'Children','Спорт':'Sport','Музыка':'Music'}
examples=J('final-examples-updated.json')
for idx,u in enumerate(examples):
 H(f'4.{idx+1}. Same-user comparison: {u["user_id"]}')
 add(f"{'Lighter' if idx==0 else 'Heavier'} history: selected at the {'25th' if idx==0 else '90th'} percentile of development-training activity before examining later-validation outcomes. Final history contains {u['qualifying_airings']} full airings and {u['distinct_programs']} programs. This same user is used across all principal methods and three finalists.",small)
 add('<b>Past favorites:</b> '+'; '.join(f"{html.escape(p['title'])} [{p['id']}] ×{p['count']}" for p in u['training_top10'][:4])+'.',small)
 rows=[['Method','Rank 1','Rank 2','Rank 3','Rank 4','Rank 5']]
 for m in labels:rows.append([labels[m]]+[str(p['id']) for p in u['recommendations'][m]])
 table(rows,[105,80,80,80,80,80],small)
 catalog={p['id']:p for m in labels for p in u['recommendations'][m]}
 rows=[['Program ID','Title (representative schedule label)','Category','Past full watches']]
 for i,p in sorted(catalog.items()):rows.append([i,html.escape(p['title']),catmap.get(p['category'],p['category']),p['train_watch_count']])
 table(rows,[65,291,89,60],small)
 add('Ranked IDs resolve directly to titles/categories above; episode labels can differ across airings sharing an ID. The last column counts this user’s full training watches. No held-out test outcomes are available for these illustrative lists.',cap)
 if idx==0:
  add('<b>Interpretation.</b> The final list follows the established news/investigation routine: “Факты. День.”, “ТСН.”, “Вещдок” and “Сегодня.” remain recognizable favorites. CBF expands toward “Деньги” and “Подробности недели” without a full watch in this history; PPR can preserve the watched film “Воин”. The final fifth choice, “Завтрак с 1+1”, has no full watch in this user’s history. Its channel context may be plausible, but relevance is unverified. This illustrates content similarity versus co-view habits, not proven discovery.',small)
 else:
  add('<b>Interpretation.</b> The strongest history is “ТСН.” and “Завтрак с 1+1”. The final ranking preserves this routine, while CBF concentrates on broad news similarity and misses the distinction between a bulletin and a familiar breakfast format. PPR also retains the watched series “Великолепный век”. ALS suggests a plausible breakfast program here, but one plausible example cannot override its weak aggregate MAP.',small)
 page()
H('5. Discussion: failures, refinement and strategy')
add('<b>Build and question.</b> Popularity was the required reference; repeat/channel signals were stronger than the four course methods. I therefore selected learned fusion for the main habitual-viewing list, asking whether extra modeling could recover programs absent from a user’s history. A simpler repeat/channel blend remains a practical fallback, but production latency, retention and revenue effects have not been measured.',small)
add('<b>Diagnose and improve.</b> On later validation, the v5 baseline retrieved 830 of 1,185 unseen targets but placed only 26 in Top-5; their median candidate rank was 84. Of these targets, 561 had one future full watch and 126 had no prior global overlap. This points to both sparse evidence and weak reranking. Expanded content/channel features, separate unseen classifiers, soft tie targets and historical transitions were tested. The selected blend raised unseen hits only to 28; it did not solve discovery. Across the final cohort, the measured new-to-user share is '+f"{J('final-novelty-updated.json')['v6']*100:.2f}%"+' for v6 versus 34.51% for CBF. New to the recorded full-watch history does not mean never seen, enjoyed, or useful.',small)
add('<b>Concrete failure and fix.</b> Naïve timestamp subtraction disagreed with 117 supplied durations by exactly one hour. Kyiv-aware conversion through UTC removed the DST discrepancies. A contract check also exposed a calendar-window/test-file spill mismatch; the final candidate set now matches the official file. Rebuilding features and checking prediction hashes verified these fixes. Separately, 785/1,078 later-validation users tie at the fifth target: alternative tie rules materially change absolute MAP, so our ascending-ID convention is a limitation.',small)
add('<b>Negative results and scope.</b> Neural training, MAP-oriented ranking and 12 later error-analysis cycles failed to beat the best simple fusion convincingly. There were 19 actual Kaggle submissions. The final private gain over v5 is just 0.00006 (public score decreases); aggregate leaderboard feedback was used adaptively, so the selected score is not an unbiased test estimate. Missing schedule IDs, correlated overlapping windows, unknown official tie rules and the longer test horizon remain limitations. A useful next step is an untouched longer-period evaluation with an agreed target definition. I make no online-benefit or score-ceiling claim.',small)
H('6. AI usage and personal verification')
add('OpenAI Codex (GPT-6-Astra; native session metadata retained) assisted with source reading, design, coding, debugging, experiments, automated checks and drafting. A separate new agent session reproduces and reviews the existing v2/v5/v6 work; it is not presented as the original multi-day development history. Its unchanged native export records that scope. Earlier experimental configurations/results and external-code attributions remain in the repository. I have not personally re-derived every model or rerun all automated checks.',small)
add('<b>My hand check.</b> For targets A, B, C, D, E and predictions A, X, B, Y, C, hits are at ranks 1, 3 and 5: The calculation is shown below. Randomly mixing past and future views would train on behavior unavailable at recommendation time, so I use earlier views to predict later views. This calculation and explanation were supplied in the working chat.',small)
mathline(r'\mathrm{AP}@5=\frac{1+\frac{2}{3}+\frac{3}{5}}{5}=\frac{34}{75}\approx0.4533','ap5.png',245)
H('7. Designated Kaggle result')
add('<b>v6_discovery_blend.csv:</b> private MAP@5 <b>0.45170</b>; public <b>0.45571</b>. 1,260 users, five distinct official program IDs each. The screenshot is the returned result, not the frozen leaderboard position. “After deadline” refers to the original 2020 competition.',small)
shot=A/'discovery-v6/kaggle-result-detail.png';story.append(Image(str(shot),width=340,height=340*440/600))
add('SHA-256: 9a295e20a024ef074e3c388b00b64e2a7ecc83b62f2cc65278ce8b7d03061f89',cap)
page();H('References and external-code attribution')
refs=[
'[1] Kaggle / ogurbych, “sweet.tv - TV program recommender,” Data description, 2020. https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data. Accessed 2026-10-31. Target construction and user-wise private split were consulted.',
'[2] Implicit, “AlternatingLeastSquares,” CPU API documentation. https://benfred.github.io/implicit/api/models/cpu/als.html. Accessed 2026-10-31. Sparse confidence semantics checked; installed version 0.7.3.',
'[3] H. Steck, “Embarrassingly Shallow Autoencoders for Sparse Data,” The Web Conference, 2019, arXiv:1905.03375. https://arxiv.org/html/1905.03375v1. Accessed 2026-10-31. Sections 2 and 3.1, including the zero diagonal and column-normalized inverse, were consulted.',
'[4] CatBoost, “CatBoostClassifier,” Python reference. https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier. Accessed 2026-11-01. Implementation uses 1.2.10; depth, iterations, learning rate, L2 regularization, seed and thread settings are recorded in source.',
'[5] H&amp;M first-place solution discussion, https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/324070; OTTO third-place solution repository, https://github.com/TheoViel/kaggle_otto_rs (commit b1507a8728cfed4edb7c36a48417c8603a697fcc). Accessed 2026-10-31. Retrieved features/candidate fusion ideas were adapted; their pipelines and benchmark scores were not reproduced.',
'[6] M. Turkin and A. Tsepa, “Gornyaki_final_solution,” public Kaggle notebook, version 10, Apache-2.0. https://www.kaggle.com/code/kinahem/gornyaki-final-solution. Accessed 2026-11-01. Statistical formulas were reimplemented against our own interval pipeline; private prejoined datasets and prediction outputs were not used. Its positional metric was not adopted.',
'[7] eSASRec / transformer_benchmark, author implementation and paper (2025). https://github.com/blondered/transformer_benchmark; https://arxiv.org/abs/2508.06450. Accessed 2026-10-31. LiGR gated attention/SwiGLU blocks were adapted for a small next-event model, not a paper-score reproduction. Apache-2.0 license is retained under third_party/. No pretrained weights were added.'
]
for r in refs:
 r=re.sub(r'(https://[^ ;<]+)',lambda m:f'<link href="{m[0].rstrip(chr(46))}" color="#12627b">{m[0]}</link>',r)
 add(r,body)
add('Course-format source: Seungmin Jin, LLM4Rec Homework Guidelines, official course repository, https://github.com/dryjins/RecSys-LLMs/blob/main/homework_guidelines.pdf. Read in full; conflicting five/six-page statements are resolved using the stricter five content pages. The new session is disclosed as a reproduction log, not a replacement claim about the original development chronology.',small)
def footer(c,d):
 c.setFont('Arial',7.5);c.setFillColor(HexColor('#5a6e79'));c.drawString(43,24,'Rustam Shangareev · SWEET.TV recommendation');c.drawRightString(552,24,str(d.page))
doc=SimpleDocTemplate(str(OUT/'Rustam_Shangareev_exam_report.pdf'),pagesize=(595.276,841.89),leftMargin=43,rightMargin=43,topMargin=32,bottomMargin=38,title='Learning Recurring TV Habits from Incomplete Viewing Logs',author='Rustam Shangareev')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'Rustam_Shangareev_exam_report.md').write_text('\n\n'.join(markdown))
print(OUT/'Rustam_Shangareev_exam_report.pdf')
