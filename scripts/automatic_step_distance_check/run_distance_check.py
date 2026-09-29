"""Independent depth-to-step rerun; original artifacts remain read-only."""
import os,sys,json,hashlib,shutil,time,math,argparse
from pathlib import Path
from collections import Counter
from dataclasses import replace
CODE_ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--annotations',type=Path,required=True)
parser.add_argument('--images',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--limit',type=int,default=77)
args=parser.parse_args()
ROOT=args.output.resolve();ROOT.mkdir(parents=True,exist_ok=False)
GT=args.annotations.resolve();IMAGES=args.images.resolve()
os.environ['HF_HOME']=str(ROOT/'model_cache')
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
sys.dont_write_bytecode=True
VENDOR=CODE_ROOT/'implementation'
source_hashes={p.relative_to(VENDOR).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in VENDOR.rglob('*.py')}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sys.path.insert(0,str(VENDOR))
import numpy as np
import torch,transformers
from PIL import Image
from config import build_config
from real_world_ssi.model_registry import ModelRegistry
from real_world_ssi.depth_estimation import DepthAnythingV2Service,representative_depth,depth_to_steps
from real_world_ssi.mask_extraction import GroundedSam2MaskService,MaskExtractionError
from real_world_ssi.schemas import YoloObject

torch.manual_seed(0);torch.set_num_threads(4)
cfg=replace(build_config(),dataset_root=IMAGES,output_root=ROOT/'results',model_cache_dir=ROOT/'model_cache',device='cuda',auto_download_models=True)
cfg.validate()
assert torch.cuda.is_available(),'CUDA unavailable'
reg=ModelRegistry(cfg,'cuda');depth_service=DepthAnythingV2Service(reg);mask_service=GroundedSam2MaskService(reg,cfg)
ann=json.loads(GT.read_text(encoding='utf-8'))['annotations']
assert len(ann)==77
out=ROOT/'results';out.mkdir(exist_ok=True)
save(ROOT/'run_manifest.json',{'gt':str(GT),'gt_sha256':digest(GT),'source_code_sha256':source_hashes,'device':torch.cuda.get_device_name(),'torch':torch.__version__,'transformers':transformers.__version__,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(cfg).items()},'scope':'Diagnostic rerun using reconstruction defaults; not original historical calibration residuals. No refitting. No reference-distance-dependent selection. All non-ignore roles reported, required separately. No depth-consistency image filtering, so the complete evaluation subset is attempted.'})
print('Loading depth, Grounding DINO, and SAM2 models',flush=True)
for field in ['depth_processor','depth_model','grounding_processor','grounding_model','sam2_processor','sam2_model']:
    getattr(reg,field); print('Loaded '+field,flush=True)
revisions={k:getattr(v.config,'_commit_hash',None) for k,v in reg._models.items() if hasattr(v,'config')}
save(ROOT/'model_revisions.json',revisions)
allrows=[]
for number,(name,record) in enumerate(ann.items()):
    if number>=args.limit:break
    folder=out/Path(name).stem;folder.mkdir(exist_ok=True)
    resultfile=folder/'objects.json'
    if resultfile.exists():
        allrows.extend(json.loads(resultfile.read_text(encoding='utf-8')));print('Resume '+name,flush=True);continue
    start=time.monotonic()
    image=Image.open(IMAGES/name).convert('RGB')
    assert image.size==(record['image_width'],record['image_height'])
    depth=depth_service.predict(image)
    np.save(folder/'depth_raw.npy',depth.raw)
    Image.fromarray(depth.depth_8bit).save(folder/'depth_8bit.png')
    rows=[]
    for index,h in enumerate(record['hazards']):
        row={'image':name,'hazard_id':h['id'],'object_name':h['object_name'],'category':h['hazard_category'],'role':h['evaluation_role'],'reference_steps':h['distance_steps'],'status':'pending'}
        box=h['bbox'];w,hei=image.size
        xyxy=(box['x']*w,box['y']*hei,(box['x']+box['width'])*w,(box['y']+box['height'])*hei)
        row['bbox_xyxy']=xyxy
        obj=YoloObject(index,0,h['object_name'],xyxy)
        try:
            mask=mask_service.extract(image,(obj,))[0]
            Image.fromarray(mask.mask.astype(np.uint8)*255).save(folder/f'mask_{index:03d}.png')
            d=representative_depth(depth.depth_8bit,mask.mask)
            pred=depth_to_steps(d)
            row.update(status='ok',representative_depth=d,predicted_steps=pred,signed_error=pred-h['distance_steps'],absolute_error=abs(pred-h['distance_steps']),mask_source=mask.source,mask_diagnostics=mask.diagnostics)
        except (MaskExtractionError,ValueError) as exc:
            row.update(status='failed',error=str(exc))
        rows.append(row)
    save(resultfile,rows);allrows.extend(rows)
    print(f'{number+1}/{args.limit} {name}: {sum(r["status"]=="ok" for r in rows)}/{len(rows)} objects; {time.monotonic()-start:.1f}s',flush=True)
def metrics(rows):
    ok=[r for r in rows if r['status']=='ok']
    errors=np.array([r['signed_error'] for r in ok],dtype=float)
    return {'eligible_objects':len(rows),'successful_objects':len(ok),'failed_objects':len(rows)-len(ok),'images_eligible':len({r['image'] for r in rows}),'images_successful':len({r['image'] for r in ok}),'mae':float(np.abs(errors).mean()) if len(ok) else None,'rmse':float(np.sqrt((errors**2).mean())) if len(ok) else None,'bias':float(errors.mean()) if len(ok) else None,'median_absolute_error':float(np.median(np.abs(errors))) if len(ok) else None,'within_one_step':float((np.abs(errors)<=1).mean()) if len(ok) else None}
save(out/'all_objects.json',allrows)
summary={'images_attempted':min(args.limit,len(ann)),'all_non_ignore':metrics([r for r in allrows if r['role']!='ignore']),'required':metrics([r for r in allrows if r['role']=='required']),'optional':metrics([r for r in allrows if r['role']=='optional']),'ignored_not_in_primary':metrics([r for r in allrows if r['role']=='ignore']),'by_category':{c:metrics([r for r in allrows if r['role']!='ignore' and r['category']==c]) for c in ['GROUND','PIT','OVERHEAD']}}
save(out/'summary.json',summary)
assert all(digest(VENDOR/n)==h for n,h in source_hashes.items())
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
