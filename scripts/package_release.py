"""Create a public source/results archive, excluding data, caches and chat logs."""
from pathlib import Path
import zipfile,json,hashlib,re
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'output/release';OUT.mkdir(parents=True,exist_ok=True)
files=[]
for base in ['src','scripts','tests','configs','third_party','docs','artifacts','submissions']:
 for p in (ROOT/base).rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts:continue
  rel=p.relative_to(ROOT).as_posix()
  if p.suffix in ['.npy','.npz','.pkl','.pt','.cbm','.pyc','.log'] or '/cache/' in rel or '/predictions/' in rel:continue
  if p.suffix not in ['.py','.json','.csv','.png','.md','.txt']:continue
  if p.name in ['model.txt','kaggle-data-description.txt','kaggle-terms.png','kaggle-submit-form.png','development-results-before-dst-hour-fix.json','experiment-source-v1.py'] or 'pre-submission-boundary-fix' in rel:continue
  if p.stat().st_size>15_000_000:raise RuntimeError(('Unexpected large public file',rel))
  files.append(p)
files += [ROOT/n for n in ['README.md','pyproject.toml','requirements.lock.txt','.gitignore']]
patterns=[r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'\bgh[pousr]_[A-Za-z0-9_]{24,}',r'\bsk-[A-Za-z0-9_-]{24,}',r'(?i)(?:api_key|password|access_token|secret_key)\s*[:=]\s*[\x22\x27][^\x22\x27\s]{12,}']
issues=[]
for p in files:
 if p.suffix=='.png':continue
 s=p.read_text(errors='replace')
 if any(re.search(pat,s) for pat in patterns):issues.append(str(p.relative_to(ROOT)))
if issues:raise RuntimeError(('Possible secrets; review before publication',issues))
with zipfile.ZipFile(OUT/'solution-source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(set(files)):z.write(p,p.relative_to(ROOT))
manifest=[{'file':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(files))]
(OUT/'public-manifest.json').write_text(json.dumps(manifest,indent=2))
(OUT/'README.md').write_text('# SWEET.TV final exam — Rustam Shangareev\n\nFull implementation, tests, experimental results, source attribution and reproduction instructions are in **[solution-source.zip](solution-source.zip)**. Unzip it before running commands from the enclosed README. Raw competition data, learned-model caches and AI-session logs are not published.\n\nDesignated predictions: **[v6_discovery_blend.csv](v6_discovery_blend.csv)** — private MAP@5 **0.45170**, public **0.45571**. SHA-256: `9a295e20a024ef074e3c388b00b64e2a7ecc83b62f2cc65278ce8b7d03061f89`.\n\nThe result is the best observed among 19 submissions; repeated validation/leaderboard selection limits generalization claims. The previous v5 private score was 0.45164, so the latest increase is negligible.\n\nThe archive includes individually evaluated content-based filtering, implicit ALS, association rules, Personalized PageRank, popularity and other baselines, plus the final CatBoost/channel rank fusion.\n\n[Public file manifest](public-manifest.json) lists exact archive contents and hashes.\n')
import shutil
shutil.copy2(ROOT/'submissions/v6_discovery_blend.csv',OUT/'v6_discovery_blend.csv')
print({'files':len(files),'zip_bytes':(OUT/'solution-source.zip').stat().st_size,'checks':'no raw data, credentials or chat files; text scanned'})
