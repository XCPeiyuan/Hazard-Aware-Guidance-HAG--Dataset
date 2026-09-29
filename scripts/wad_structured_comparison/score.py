"""Score supplied structured references and two prediction sets."""
import argparse,json
from pathlib import Path
from data_metrics import score_dataset
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--model-a',type=Path,required=True)
    p.add_argument('--model-b',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=score_dataset(*[json.loads(x.read_text(encoding='utf-8-sig')) for x in (a.reference,a.model_a,a.model_b)])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
if __name__=='__main__':main()
