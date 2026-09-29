"""Single-card MR05 worker; retain the original inference path and timing boundary."""
import argparse,os,sys,time,traceback,platform
from pathlib import Path
from mr05_common import *
def infer_once(model,processor,image_path,prompt,max_new_tokens,*,torch_module=None,clock=time.perf_counter):
 if torch_module is None:
  import torch as torch_module
 from PIL import Image
 torch=torch_module
 started=time.time();t0=clock()
 image=Image.open(image_path).convert('RGB');t1=clock()
 messages=[{'role':'user','content':[{'type':'text','text':prompt},{'type':'image','image':image}]}]
 inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt');t2=clock()
 inputs={k:v.to(model.device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
 torch.cuda.synchronize();t3=clock()
 with torch.no_grad():generated_ids=model.generate(**inputs,max_new_tokens=max_new_tokens)
 torch.cuda.synchronize();t4=clock()
 trimmed=[out_ids[len(in_ids):] for in_ids,out_ids in zip(inputs['input_ids'],generated_ids)]
 texts=processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False);t5=clock()
 ended=time.time()
 token_ids=trimmed[0].tolist() if trimmed else [];text=texts[0].strip() if texts else ''
 eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos or [])
 image.close()
 return {**timing_fields([t0,t1,t2,t3,t4,t5]),'started_at_epoch':started,'ended_at_epoch':ended,'timing_boundaries_perf_counter':[t0,t1,t2,t3,t4,t5],'generated_tokens':len(token_ids),'generated_token_ids':token_ids,'response_text':text,'last_token_id':token_ids[-1] if token_ids else None,'eos_token_ids':eos,'ended_with_eos':bool(token_ids and token_ids[-1] in eos),'hit_token_limit':len(token_ids)>=max_new_tokens,'status':'ok','error':None}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-root',required=True);ap.add_argument('--replica',required=True);args=ap.parse_args()
 root=Path(args.run_root);cfg=read_json(root/'config/run_config.json');spec=next(s for s in cfg['workers'] if s['replica']==args.replica);out=root/'workers'/args.replica;out.mkdir(parents=True,exist_ok=True)
 meta={'run_id':cfg['run_id'],'model':spec['model'],'replica':spec['replica'],'physical_gpu':spec['physical_gpu'],'started_at_epoch':time.time(),'status':'loading','model_load_count':0,'warmup_completed':0,'rounds':[],'peak_reset_scope':'per_round','visible_cuda_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'seed':cfg['seed'],'seed_scope':'Python, NumPy, torch CPU/CUDA; one seed set at process start; no reset at round boundary','warmup_finished_at_epoch':None,'barrier_released_at_epoch':None}
 try:
  if os.environ.get('CUDA_VISIBLE_DEVICES')!=spec['physical_gpu']['uuid']:raise RuntimeError('Unexpected visible physical GPU')
  import torch,transformers,PIL,random,numpy as np
  from transformers import AutoModelForImageTextToText,AutoProcessor
  from mr05_stats import classify_response,quality_flags
  random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
  meta.update(visible_device_count=torch.cuda.device_count(),python=sys.version,pytorch=torch.__version__,transformers=transformers.__version__,pillow=PIL.__version__,cuda=torch.version.cuda,torch_num_threads=torch.get_num_threads(),torch_num_interop_threads=torch.get_num_interop_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)),platform=platform.platform())
  if meta['visible_device_count']!=1:raise RuntimeError('Worker must see exactly one GPU')
  t=time.perf_counter();model=AutoModelForImageTextToText.from_pretrained(spec['model_path'],torch_dtype='auto',device_map='auto',local_files_only=True);meta['model_load_count']=1
  processor=AutoProcessor.from_pretrained(spec['model_path'],local_files_only=True);model.eval();meta['load_seconds']=time.perf_counter()-t
  mapping={str(k):str(v) if not isinstance(v,int) else v for k,v in getattr(model,'hf_device_map',{}).items()}
  devices=sorted({str(p.device) for p in model.parameters()});meta.update(actual_device_map=mapping,parameter_devices=devices,actual_dtypes=sorted({str(p.dtype) for p in model.parameters()}),model_device=str(model.device),generation_parameters={**model.generation_config.to_dict(),'max_new_tokens':cfg['max_new_tokens']},processor_class=type(processor).__name__,processor_config=processor.to_dict() if hasattr(processor,'to_dict') else {},status='warming_up')
  if not mapping or any(str(v) not in ('0','cuda:0') for v in mapping.values()) or devices!=['cuda:0']:raise RuntimeError('CPU/disk offload or non-single-GPU placement')
  props=torch.cuda.get_device_properties(0)
  actual_uuid=str(getattr(props,'uuid',''))
  if actual_uuid and actual_uuid.removeprefix('GPU-').lower()!=spec['physical_gpu']['uuid'].removeprefix('GPU-').lower():raise RuntimeError('Actual GPU UUID mismatch')
  meta['cuda_device_properties']={'name':torch.cuda.get_device_name(0),'total_memory':props.total_memory,'uuid':actual_uuid or None,'uuid_selection_method':'CUDA_VISIBLE_DEVICES exact physical UUID'}
  meta['effective_image_processor']=processor.image_processor.to_dict()
  if getattr(processor,'video_processor',None) is not None:meta['effective_video_processor']=processor.video_processor.to_dict()
  meta['effective_chat_template']=processor.chat_template
  meta['effective_tokenizer']={'class':type(processor.tokenizer).__name__,'vocab_size':processor.tokenizer.vocab_size,'padding_side':processor.tokenizer.padding_side,'truncation_side':processor.tokenizer.truncation_side,'model_max_length':processor.tokenizer.model_max_length}
  write_json(out/'metadata.json',meta);write_json(out/'model_config.json',model.config.to_dict());write_json(out/'generation_config.json',meta['generation_parameters'])
  samples=load_samples(root/'sample_manifest.csv');prompt=(root/'config/prompt.txt').read_text();deadline=cfg['deadline_epoch']
  def one(sample,phase,round_id,sequence):
   if (root/'ABORT.json').exists() or time.time()>=deadline:raise TimeoutError('Session stop/deadline')
   base={k:sample[k] for k in ('record_id','run_order','dataset_index','image_name','source_group','source_sample_index')}
   base.update(run_id=cfg['run_id'],model=spec['model'],replica=args.replica,gpu_id=spec['physical_gpu']['index'],gpu_uuid=spec['physical_gpu']['uuid'],round=round_id,phase=phase,image_path=str(Path(cfg['local_images_root'])/sample['image_name']),warmup_index=sequence if phase=='warmup' else None)
   started=time.time();fatal=None
   try:
    value=infer_once(model,processor,Path(base['image_path']),prompt,cfg['max_new_tokens']);value['predicted_branch']=classify_response(value['response_text']);value['quality_flags']=quality_flags(value['response_text'])
   except Exception as exc:
    value={**failure_result(exc),'started_at_epoch':started,'ended_at_epoch':time.time()}
    if phase=='warmup' or isinstance(exc,torch.cuda.OutOfMemoryError) or 'CUDA' in str(exc):fatal=exc
   row={**base,**value};append_jsonl(out/('warmup.jsonl' if phase=='warmup' else f'round_{round_id}.jsonl'),row)
   print(f"{phase} round={round_id} record={sample['record_id']} status={row['status']} e2e={row['end_to_end_latency_sec']}",flush=True)
   if fatal:raise RuntimeError('Fatal inference failure') from fatal
   return row
  for i,s in enumerate(warmup_samples(samples),1):
   row=one(s,'warmup',0,i)
   if row['status']!='ok':raise RuntimeError('Warmup failed')
   meta['warmup_completed']+=1
  meta.update(warmup_finished_at_epoch=time.time(),status='ready',barrier_wait_started_at_epoch=time.time());write_json(out/'metadata.json',meta);write_json(out/'ready.json',{'replica':args.replica,'ready_at_epoch':time.time(),'warmup_completed':10})
  release=await_release(root,deadline);meta.update(barrier_released_at_epoch=release['released_at_epoch'],barrier_wait_ended_at_epoch=time.time(),status='running');write_json(out/'metadata.json',meta)
  for n in (1,2):
   torch.cuda.reset_peak_memory_stats(0);info={'round':n,'peak_reset_at_epoch':time.time(),'start_at_epoch':time.time()}
   for s in samples:one(s,'formal',n,None)
   info.update(end_at_epoch=time.time(),peak_memory_allocated_gib=torch.cuda.max_memory_allocated(0)/(1024**3),peak_memory_reserved_gib=torch.cuda.max_memory_reserved(0)/(1024**3));meta['rounds'].append(info);write_json(out/'metadata.json',meta)
  meta.update(status='complete',ended_at_epoch=time.time(),peak_memory_allocated_gib=max(r['peak_memory_allocated_gib'] for r in meta['rounds']),peak_memory_reserved_gib=max(r['peak_memory_reserved_gib'] for r in meta['rounds']));write_json(out/'metadata.json',meta)
 except BaseException as exc:
  meta.update(status='failed',ended_at_epoch=time.time(),error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());write_json(out/'metadata.json',meta);write_json(out/'failure.json',meta);raise
if __name__=='__main__':main()
