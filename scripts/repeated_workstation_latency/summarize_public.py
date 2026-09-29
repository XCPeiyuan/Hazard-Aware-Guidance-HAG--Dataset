"""Recompute pooled statistics from the published numeric timing records."""
import csv,json,argparse,math
from pathlib import Path
from mr05_stats import percentile,STAGES
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.input.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=1600 or any(r['status']!='ok' for r in rows):raise ValueError('Expected 1600 successful observations')
    out=[]
    for model in ('3b','7b'):
        for branch in ('safe','hazard'):
            group=[r for r in rows if r['model']==model and r['predicted_branch']==branch]
            d={'model':model,'predicted_branch':branch,'count':len(group)}
            for field in (*STAGES,'end_to_end_latency_sec','generated_tokens'):
                vals=[float(r[field]) for r in group]
                d['mean_'+field]=math.fsum(vals)/len(vals)
                if field!='generated_tokens':d['p95_'+field]=percentile(vals,.95)
            out.append(d)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
if __name__=='__main__':main()
