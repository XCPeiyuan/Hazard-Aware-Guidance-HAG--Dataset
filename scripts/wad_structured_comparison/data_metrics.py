"""Object-level structured scoring with the word-boundary name matcher."""
from __future__ import annotations
import ast,hashlib,json,re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any,Iterable

CATEGORIES={"GROUND","PIT","OVERHEAD"}; STATUSES={"ok","review_required","failed"}
ID_RE=re.compile(r"^\S+\.(?:png|jpe?g)$",re.I); SAFE_RE=re.compile(r"^\s*<SAFE\s*/>\s*$",re.I)
class InputValidationError(ValueError): pass

def _hash_bytes(x:bytes)->str:return hashlib.sha256(x).hexdigest()
def _hash_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def _json(path:Path,value:Any)->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");tmp.replace(path)
def _jsonl(path:Path,values:Iterable[dict])->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
 with tmp.open("w",encoding="utf-8") as f:
  for x in values:f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n")
 tmp.replace(path)

def verify_manifest(root:Path|str,manifest_path:Path|str)->dict:
 root=Path(root).resolve();mp=Path(manifest_path).resolve();data=json.loads(mp.read_text(encoding="utf-8"));entries=data.get("files",[]) if isinstance(data,dict) else data
 if not isinstance(entries,list):raise InputValidationError(f"invalid manifest: {mp}")
 checked=[]
 for e in entries:
  rel=e.get("path",e.get("destination"));p=(root/rel).resolve()
  try:p.relative_to(root)
  except ValueError as exc:raise InputValidationError(f"manifest path escapes root: {rel}") from exc
  if not p.is_file() or p.stat().st_size!=e.get("bytes") or _hash_file(p)!=e.get("sha256"):raise InputValidationError(f"manifest mismatch: {rel}")
  checked.append(rel)
 return {"manifest":str(mp),"verified_files":len(checked),"paths":checked}

def parse_record_blocks(path:Path|str)->list[dict]:
 p=Path(path);lines=p.read_text(encoding="utf-8-sig").splitlines();starts=[i for i,x in enumerate(lines) if ID_RE.fullmatch(x.strip())]
 if not starts:raise InputValidationError(f"no image IDs in {p}")
 if any(x.strip() for x in lines[:starts[0]]):raise InputValidationError(f"content before first image ID in {p}")
 out=[]
 for n,start in enumerate(starts):
  end=starts[n+1] if n+1<len(starts) else len(lines);body=lines[start+1:end]
  while body and body[-1]=="":body.pop()
  out.append({"sample_id":lines[start].strip(),"raw_text":"\n".join(body),"source_record_index":n})
 return out

def select_target_records(records:list[dict],target_ids:Iterable[str])->tuple[list[dict],dict]:
 grouped=defaultdict(list)
 for x in records:grouped[str(x["sample_id"])].append(x)
 selected=[];provenance={};seen=set()
 for sample_id in target_ids:
  if sample_id in seen:raise InputValidationError(f"duplicate target ID: {sample_id}")
  seen.add(sample_id);xs=grouped.get(sample_id,[])
  if not xs:raise InputValidationError(f"target ID missing: {sample_id}")
  indices=[int(x["source_record_index"]) for x in xs]
  if len({x.get("raw_text","") for x in xs})!=1:raise InputValidationError(f"conflicting target duplicate {sample_id} at indices {indices}")
  x=deepcopy(xs[0]);x["source_record_indices"]=indices;selected.append(x);provenance[sample_id]=indices
 return selected,provenance

# Exact P0-05 synonym table and compatibility algorithm, copied without importing legacy side effects.
SYN={
"traffic cone":{"traffic cone","cone","cones","red cone","red cones","red traffic cone","red traffic cones","orange cone","orange cones","construction cone","construction cones"},
"barrier":{"barrier","barriers","construction barrier","construction barriers","barrier tape","tape","striped barrier","striped barriers","green-and-white striped barrier","green-and-white striped barriers","fence","fences","low fence","low fences","hoop barrier","hoop barriers","iron barrier","iron barriers","concrete barrier","concrete barriers","construction fence","construction fences","green construction barrier","green construction barriers","red-and-white barrier","red-and-white barriers","red-and-white barrier post","red-and-white barrier posts","low concrete barrier","low concrete barriers","bicycle rack","bicycle racks","gate","gates","metal gate","metal gate with vertical bars","barrier rope","rope barrier"},
"construction area":{"construction area","construction site","construction","construction zone"},"construction workers":{"construction worker","construction workers","worker","workers"},"construction machinery":{"construction machinery","machinery","machine","machines"},
"open manhole":{"open manhole","manhole","missing manhole cover","pothole","potholes","large hole in the pavement","tree pit","hole in the pavement"},"manhole cover":{"manhole cover","open manhole cover","manhole cover missing"},"construction pit":{"construction pit","pit","construction pits","pits"},"hole":{"hole","holes","ditch","ditches","trench","trenches"},
"car":{"car","cars","vehicle","vehicles","parked car","parked cars","small vehicle","small vehicles","electric vehicle","electric vehicles","minivan","white minivan","small truck","truck","trucks","pickup truck","pickup trucks"},
"bicycle":{"bicycle","bicycles","bike","bikes","bicyclist","bicycle rider","electric bicycle","electric bicycles","e-bike","e-bikes","motorcycle","motorcycles","Motorcycle","Motorcycles","scooter","scooters","electric scooter","electric scooters","parked scooter","parked scooters","electric tricycle","electric tricycles","tricycle","tricycles","two-wheeler","two-wheelers"},
"pole":{"pole","poles","utility pole","utility poles","antenna pole","antenna poles","post","posts","blue pole","lamppost","lampposts","signpost","signposts","streetlight pole","streetlight poles","column","columns","raking column","concrete column","support pillar","pillar","pillars","overhanging pillar","overhanging structure","overhanging support pillar"},
"pile of debris":{"pile of debris","debris pile","debris","pile of bricks","pile of soil","piles of soil","piles of soil and bricks","pile of bricks and soil","debris on the ground","wooden debris","wooden boards","broken pavement with debris","cracked pavement with debris","construction debris","pile of construction debris","large pile of debris","rusty cart with debris","brick","bricks","construction materials","pavement debris","broken pavement","cracked pavement","pile of debris and bricks"},
"caution tape":set(),"suspended chain":{"suspended chain","chain","hanging chain","yellow caution tape","caution tape","warning tape","barrier rope","rope barrier"},"outdoor unit":{"outdoor unit","air conditioner unit","air conditioner","ac unit","low-hanging air conditioner unit","low-hanging ac unit"},"trash can":{"trash can","trash cans","garbage can","garbage cans","bin","bins","waste bin","waste bins","dustbin","dustbins","trash bag","trash bags","garbage bag","garbage bags"},"tree":{"tree","trees","tree trunk","tree trunks","trunk","trunks"},"sign":{"sign","signs","signboard","signboards","signage","sandwich board","sandwich boards","signpost","signposts"},"pedestrian":{"pedestrian","pedestrians","person","people","persons"},"traffic cone and barrier":{"traffic cone and barrier","traffic cones and barriers","cones and barriers","cone and barrier"}}
def normalize_name(name:str)->str:
 words=re.sub(r"\s+"," ",re.sub(r"[^\w\s]"," ",name.strip().lower())).strip().split();return " ".join(x[:-1] if x.endswith("s") and len(x)>2 and not x.endswith("ss") else x for x in words)
NORM={normalize_name(x):c for c,xs in SYN.items() for x in xs}
def names_compatible(a, b):
    if not a or not b:
        return False, "empty_name"
    if a == "not_mentioned" or b == "not_mentioned":
        return False, "not_mentioned"
    an, bn = normalize_name(a), normalize_name(b)
    if an == bn:
        return True, "exact_match"
    ac, bc = NORM.get(an), NORM.get(bn)
    if ac is not None and bc is not None and ac == bc:
        return True, f"synonym({ac})"
    for s, c in NORM.items():
        if len(s) < 3:
            continue
        # Names are already normalized into space-separated tokens. Only this
        # containment boundary differs from the frozen implementation.
        if ac is not None and ac == c and f" {s} " in f" {bn} ":
            return True, f"contain_syn({s})"
        if bc is not None and bc == c and f" {s} " in f" {an} ":
            return True, f"contain_syn({s})"
    at, bt = set(an.split()), set(bn.split())
    if not at or not bt:
        return False, "no_tokens"
    intersection = at & bt
    j = len(intersection) / len(at | bt)
    if j >= .4:
        return True, f"jaccard({j:.2f})"
    if intersection == at or intersection == bt:
        return True, f"subset({j:.2f})"
    return False, f"no_match(jac={j:.2f})"

def _cats(x:dict)->set[str]:
 if x.get("category_status","resolved")!="resolved":return set()
 v=x.get("categories",[])
 if not isinstance(v,list):raise InputValidationError("categories must be a list")
 out=set(map(str,v));bad=out-CATEGORIES
 if bad:raise InputValidationError(f"invalid categories: {sorted(bad)}")
 return out
def _cost(g:dict,p:dict)->float|None:
 if not names_compatible(str(g.get("name","")),str(p.get("name","")))[0]:return None
 a,b=_cats(g),_cats(p);return 2*(1-len(a&b)/len(a|b)) if a and b else 0.
EXPECTED_SCIPY_VERSION="1.17.1"

def _require_hungarian_backend():
 try:
  import numpy as np
  import scipy
  from scipy.optimize import linear_sum_assignment
 except ImportError as exc:
  raise ImportError("MR-01 scoring requires SciPy 1.17.1 for its frozen Hungarian matching backend; install the pinned offline runtime dependency and rerun scoring.") from exc
 actual=str(scipy.__version__)
 if actual!=EXPECTED_SCIPY_VERSION:
  raise RuntimeError(f"MR-01 scoring requires SciPy {EXPECTED_SCIPY_VERSION} for reproducible equal-cost assignment; found {actual}. Use the pinned offline runtime.")
 return np,linear_sum_assignment,actual

def _match(gs:list[dict],ps:list[dict],backend=None)->dict:
 np,linear_sum_assignment,_version=backend or _require_hungarian_backend();pairs=[]
 if gs and ps:
  matrix=np.full((len(gs),len(ps)),1e6)
  for i,g in enumerate(gs):
   for j,p in enumerate(ps):
    c=_cost(g,p)
    if c is not None:matrix[i,j]=c+(j+1)*1e-9/((len(ps)+1)**(i+1))
  rows,cols=linear_sum_assignment(matrix);pairs=[(int(i),int(j)) for i,j in zip(rows,cols) if matrix[i,j]<1e6]
 mg={i for i,_ in pairs};mp={j for _,j in pairs};detail=[];candidates=[]
 for i,g in enumerate(gs):
  for j,p in enumerate(ps):
   cost=_cost(g,p)
   if cost is not None:candidates.append({"gt_idx":i,"pred_idx":j,"gt_object_id":g.get("object_id",str(i)),"pred_object_id":p.get("object_id",str(j)),"cost":cost,"name_reason":names_compatible(str(g.get("name","")),str(p.get("name","")))[1]})
 for i,j in sorted(pairs):detail.append({"gt_idx":i,"pred_idx":j,"gt_object_id":gs[i].get("object_id",str(i)),"pred_object_id":ps[j].get("object_id",str(j)),"cost":_cost(gs[i],ps[j]),"name_reason":names_compatible(str(gs[i].get("name","")),str(ps[j].get("name","")))[1]})
 competing=False
 for field in ("gt_idx","pred_idx"):
  grouped=defaultdict(list)
  for edge in candidates:grouped[edge[field]].append(edge["cost"])
  competing|=any(sum(abs(value-min(values))<1e-12 for value in values)>1 for values in grouped.values())
 return {"matched_pairs":detail,"unmatched_gt":[i for i in range(len(gs)) if i not in mg],"unmatched_pred":[j for j in range(len(ps)) if j not in mp],"candidate_edges":candidates,"competing_candidates":competing}
def _metric(tp,fp,fn):
 d=2*tp+fp+fn;return {"tp":tp,"fp":fp,"fn":fn,"f1":2*tp/d if d else 0.,"defined":bool(d)}
def _index(xs,label):
 out={}
 for x in xs:
  key=(str(x.get("dataset_id","")),str(x.get("sample_id","")))
  if not all(key) or key in out:raise InputValidationError(f"{label} invalid/duplicate record: {key}")
  if x.get("status","ok") not in STATUSES or not isinstance(x.get("objects",[]),list):raise InputValidationError(f"{label} malformed record: {key}")
  out[key]=x
 return out
def _eligible(r,metric):
 explicit=r.get(metric+"_eligible")
 if explicit is not None:return bool(explicit)
 if r.get("status","ok")!="ok" or r.get("scope_uncertain",False):return False
 return True if metric=="hazard" else all(x.get("category_status","resolved")=="resolved" and bool(_cats(x)) for x in r.get("objects",[]))
def score_dataset(refs:list[dict],preds_a:list[dict],preds_b:list[dict])->dict:
 backend=_require_hungarian_backend()
 ref=_index(refs,"reference");models={"a":_index(preds_a,"model_a"),"b":_index(preds_b,"model_b")}
 for label,m in models.items():
  missing=[x for x in ref if x not in m];extra=[x for x in m if x not in ref]
  if missing:raise InputValidationError(f"model_{label} missing prediction records: {','.join(x[1] for x in missing[:10])}")
  if extra:raise InputValidationError(f"model_{label} unknown records: {extra[:10]}")
 datasets={x[0] for x in ref};coverage={};cohorts={}
 for metric in ("hazard","category"):
  included=[];excluded=[];reasons={};keys=set()
  for key,r in ref.items():
   why=[]
   if not _eligible(r,metric):why.append("reference_ineligible")
   for label,m in models.items():
    p=m[key]
    if p.get("status","ok")!="ok":why.append(f"model_{label}:{p.get('status')}")
    elif p.get("scope_uncertain",False):why.append(f"model_{label}:scope_uncertain")
   display=key[1] if len(datasets)==1 else "/".join(key)
   if why:excluded.append(display);reasons[display]=why
   else:included.append(display);keys.add(key)
  cohorts[metric]=keys;included.sort();excluded.sort();coverage[metric]={"target_count":len(ref),"included_count":len(included),"excluded_count":len(excluded),"coverage":len(included)/len(ref) if ref else 0.,"included_ids":included,"excluded_ids":excluded,"exclusion_reasons":reasons}
 result={}
 for label,m in models.items():
  totals={"hazard":[0,0,0],"category":[0,0,0]};samples=defaultdict(dict)
  for metric in totals:
   for key in sorted(cohorts[metric]):
    gs,ps=ref[key].get("objects",[]),m[key].get("objects",[]);match=_match(gs,ps,backend)
    if metric=="hazard":counts=[len(match["matched_pairs"]),len(match["unmatched_pred"]),len(match["unmatched_gt"])]
    else:
     tp=fp=fn=0
     for pair in match["matched_pairs"]:
      g,p=_cats(gs[pair["gt_idx"]]),_cats(ps[pair["pred_idx"]]);tp+=len(g&p);fp+=len(p-g);fn+=len(g-p)
     fn+=sum(len(_cats(gs[i])) for i in match["unmatched_gt"]);fp+=sum(len(_cats(ps[j])) for j in match["unmatched_pred"]);counts=[tp,fp,fn]
    for i,x in enumerate(counts):totals[metric][i]+=x
    samples[key[1]][metric]={**match,"counts":{"tp":counts[0],"fp":counts[1],"fn":counts[2]}}
  result[label]={"hazard":_metric(*totals["hazard"]),"category":_metric(*totals["category"]),"samples":dict(samples)}
 return {"schema_version":"mr01-score-v1","matching_backend":{"implementation":"scipy.optimize.linear_sum_assignment","scipy_version":backend[2],"ordered_input_tie_policy":f"SciPy {backend[2]} result for input object order"},"coverage":coverage,"models":result}
