"""CPU-only MR05 fixed-input and timing contracts."""
import csv,json,math,time,hashlib,os
from collections import Counter
from pathlib import Path
STAGES=['image_read_sec','preprocess_sec','transfer_sec','generate_only_latency_sec','decode_sec']
def write_json(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)
def read_json(path):return json.loads(Path(path).read_text())
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def append_jsonl(path,row):
 with Path(path).open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
def validate_samples(rows):
 if len(rows)!=200 or [r['run_order'] for r in rows]!=list(range(1,201)):raise ValueError('Expected ordered 200 records')
 if len({r['record_id'] for r in rows})!=200:raise ValueError('Duplicate record IDs')
 if len({r['image_name'] for r in rows})!=195:raise ValueError('Expected 195 distinct images')
 if sorted(Counter(r['image_name'] for r in rows).values())!=[1]*190+[2]*5:raise ValueError('Expected exactly five duplicate image pairs')
 if any(sum(r['source_group']==g for r in rows)!=100 for g in ('safe','hazard')):raise ValueError('Expected 100 source-safe and100 source-hazard')
 for r in rows:
  if Path(r['image_name']).name!=r['image_name']:raise ValueError('Unsafe image filename')
 return rows
def load_samples(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 for r in rows:
  for k in ('run_order','dataset_index','source_sample_index'):r[k]=int(r[k])
 return validate_samples(rows)
def schedule(rows,rounds):return [(n,r) for n in range(1,rounds+1) for r in rows]
def warmup_samples(rows):
 a=[r for r in rows if r['source_group']=='safe'][:5];b=[r for r in rows if r['source_group']=='hazard'][:5]
 if len(a)!=5 or len(b)!=5:raise ValueError('Need five warmups from each source group')
 return [v for pair in zip(a,b) for v in pair]
def timing_fields(ts):
 if len(ts)!=6 or any(not math.isfinite(t) for t in ts) or any(b<a for a,b in zip(ts,ts[1:])):raise ValueError('Invalid timing boundaries')
 return {**dict(zip(STAGES,[b-a for a,b in zip(ts,ts[1:])])),'end_to_end_latency_sec':ts[-1]-ts[0]}
def failure_result(exc):
 return {**{k:None for k in STAGES+['end_to_end_latency_sec']},'generated_tokens':None,'generated_token_ids':None,'response_text':None,'predicted_branch':None,'last_token_id':None,'eos_token_ids':None,'ended_with_eos':None,'hit_token_limit':None,'quality_flags':[],'status':'error','error':f'{type(exc).__name__}: {exc}'}
def await_release(run_root,deadline):
 root=Path(run_root)
 while True:
  if (root/'ABORT.json').exists():raise RuntimeError('Session aborted before barrier release')
  if time.time()>=deadline:raise TimeoutError('Barrier deadline exceeded')
  p=root/'config/barrier_release.json'
  if p.exists():return read_json(p)
  time.sleep(.2)
