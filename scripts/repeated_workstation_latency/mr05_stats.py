"""CPU-only MR05 observation statistics and fail-closed acceptance checks."""
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

STAGES = ('image_read_sec', 'preprocess_sec', 'transfer_sec', 'generate_only_latency_sec', 'decode_sec')
TIMINGS = STAGES + ('end_to_end_latency_sec',)
WORKERS = ('3b_replica_1','3b_replica_2','7b_replica_1','7b_replica_2')


def percentile(values, q):
    if not 0 <= q <= 1:
        raise ValueError('q must be in [0, 1]')
    values = sorted(values)
    if not values:
        return None
    pos = (len(values)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)


def classify_response(text):
    return 'safe' if text.strip().lower() == '<safe/>' else 'hazard'


def quality_flags(text):
    text = text.strip()
    if not text:
        return ['empty_output', 'malformed_output']
    if classify_response(text) == 'safe':
        return []
    return [] if re.fullmatch(r'<ALERT>.+?</ALERT>\s*<GUIDE>.+?</GUIDE>', text, re.DOTALL) else ['malformed_output']


def _mean(values):
    return math.fsum(values)/len(values) if values else None


def summarize_rows(rows):
    rows = list(rows)
    good = [r for r in rows if r.get('status') == 'ok']
    result = dict(attempts=len(rows), successes=len(good), failures=len(rows)-len(good), branches={}, quality=dict(Counter(flag for r in good for flag in quality_flags(r['response_text']))), hit_token_limit_count=sum(bool(r.get('hit_token_limit')) for r in good), errors=[r.get('error') for r in rows if r.get('status') != 'ok'])
    for branch in ('safe','hazard'):
        group = [r for r in good if classify_response(r['response_text']) == branch]
        summary = {'count':len(group)}
        for field in TIMINGS + ('generated_tokens',):
            values = [r[field] for r in group if r.get(field) is not None]
            summary['mean_'+field] = _mean(values)
            if field in TIMINGS:
                summary['p95_'+field] = percentile(values, .95)
        result['branches'][branch] = summary
    return result


def _jsonl(path, diagnostics=None):
    rows = []
    # Strict by default; explicit diagnostics opt into preserving readable rows.
    with Path(path).open('rb') as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('JSONL row must be an object')
                rows.append(row)
            except (ValueError, UnicodeError) as exc:
                if diagnostics is None:
                    raise
                diagnostics.append(f'{path}: line {number}: {exc}')
    return rows


def legacy_regression(bundle_root):
    comparisons = []
    for model in ('3b','7b'):
        base = Path(bundle_root)/'materials'/'original_results'/model
        rows = _jsonl(base/'per_run.jsonl')
        for row in rows:
            row['status'] = 'error' if row.get('error') else 'ok'
        actual = summarize_rows(rows)['branches']
        expected = json.loads((base/'summary.json').read_text())['groups']
        for branch in ('safe','hazard'):
            for field, old in (('count','count'),('mean_end_to_end_latency_sec','avg_end_to_end_latency_sec'),('p95_end_to_end_latency_sec','p95_end_to_end_latency_sec'),('mean_generated_tokens','avg_generated_tokens')):
                a,e = actual[branch][field],expected[branch][old]
                comparisons.append(dict(model=model,branch=branch,field=field,expected=e,actual=a,absolute_error=abs(a-e),passed=abs(a-e)<=1e-9))
    return dict(passed=all(c['passed'] for c in comparisons),tolerance=1e-9,comparisons=comparisons)


def summarize_run(run_root):
    root=Path(run_root)
    pooled={m:[] for m in ('3b','7b')}
    by_round=[]
    quality={}
    for worker in WORKERS:
        all_rows=[]
        parse_errors=[]
        for round_no in (1,2):
            path=root/'workers'/worker/f'round_{round_no}.jsonl'
            rows=_jsonl(path, parse_errors) if path.exists() else []
            all_rows.extend(rows)
            result=summarize_rows(rows)
            for branch,stats in result['branches'].items():
                by_round.append(dict(model=worker[:2],replica=worker,round=round_no,predicted_branch=branch,**stats))
        pooled[worker[:2]].extend(all_rows)
        quality[worker]={k:v for k,v in summarize_rows(all_rows).items() if k!='branches'}
        quality[worker]['parse_errors']=parse_errors
    pooled_table=[]
    for model,rows in pooled.items():
        for branch,stats in summarize_rows(rows)['branches'].items():
            pooled_table.append(dict(model=model,predicted_branch=branch,**stats))
    return dict(pooled_by_model_branch=pooled_table,by_replica_round=by_round,detailed_timing=by_round,quality_and_errors=quality)


def validate_run(run_root):
    root=Path(run_root)
    errors=[]
    parse_errors=[]
    def check(condition,message):
        if not condition: errors.append(message)
    def read_json(path):
        try: return json.loads(path.read_text())
        except (OSError,ValueError) as exc:
            errors.append(f'{path.relative_to(root)}: {exc}')
            return {}
    def read_rows(path):
        try:
            diagnostics=[]
            rows=_jsonl(path,diagnostics)
            errors.extend(diagnostics)
            parse_errors.extend(diagnostics)
            return rows
        except (OSError,ValueError) as exc:
            errors.append(f'{path.relative_to(root)}: {exc}')
            return []
    config=read_json(root/'config'/'run_config.json')
    barrier=read_json(root/'config'/'barrier_release.json').get('released_at_epoch')
    check(config.get('max_new_tokens')==1024,'max_new_tokens must be 1024')
    try:
        with (root/'sample_manifest.csv').open() as f: manifest=list(csv.DictReader(f))
    except (OSError,ValueError) as exc:
        errors.append(f'sample_manifest: {exc}'); manifest=[]
    check(len(manifest)==200,'manifest must contain 200 records')
    check([r.get('run_order') for r in manifest]==[str(i) for i in range(1,201)],'manifest run_order must be 1..200')
    check(len({r.get('record_id') for r in manifest})==200,'manifest record IDs must be unique')
    images=Counter(r.get('image_name') for r in manifest)
    check(len(images)==195 and Counter(images.values())=={1:190,2:5},'manifest must contain 195 images and five duplicates')
    check(Counter(r.get('source_group') for r in manifest)=={'safe':100,'hazard':100},'manifest source groups must be 100 safe / 100 hazard')
    formal=[]; warmups=[]; metas=[]; seen=set(); physical=[]
    finite=lambda x: isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x)
    for worker in WORKERS:
        base=root/'workers'/worker
        meta=read_json(base/'metadata.json'); metas.append(meta)
        check(meta.get('model_load_count')==1,f'{worker}: model_load_count != 1')
        check(meta.get('warmup_completed')==10,f'{worker}: warmup_completed != 10')
        check(meta.get('peak_reset_scope')=='per_round',f'{worker}: peak reset scope must be per_round')
        check(meta.get('generation_parameters',{}).get('max_new_tokens')==1024,f'{worker}: generation token limit')
        check(bool(meta.get('actual_dtypes')),f'{worker}: missing actual dtypes')
        mapping=meta.get('actual_device_map',{})
        check(bool(mapping) and all(v in (0,'0','cuda:0') for v in mapping.values()),f'{worker}: actual device map is not one local GPU')
        gpu=meta.get('physical_gpu',{})
        physical.append(gpu.get('uuid'))
        check(gpu.get('uuid') and 'A6000' in gpu.get('name',''),f'{worker}: missing A6000 identity')
        check(meta.get('visible_device_count')==1 and meta.get('visible_cuda_devices') in (str(gpu.get('index')),gpu.get('uuid')),f'{worker}: visible device mapping')
        check(meta.get('barrier_released_at_epoch')==barrier and finite(barrier),f'{worker}: barrier release mismatch')
        warm=read_rows(base/'warmup.jsonl'); warmups.extend(warm)
        safe=[r for r in manifest if r.get('source_group')=='safe'][:5]
        hazard=[r for r in manifest if r.get('source_group')=='hazard'][:5]
        expected_warm=[r['record_id'] for pair in zip(safe,hazard) for r in pair]
        check([r.get('record_id') for r in warm]==expected_warm,f'{worker}: warmup sample ordering')
        check([r.get('warmup_index') for r in warm]==list(range(1,11)) and all(r.get('phase')=='warmup' and r.get('round')==0 for r in warm),f'{worker}: warmup sequence identity')
        check(len(warm)==10 and all(r.get('status')=='ok' for r in warm),f'{worker}: requires ten successful warmups')
        warm_finished=meta.get('warmup_finished_at_epoch')
        check(finite(warm_finished) and finite(barrier) and warm_finished<=barrier,f'{worker}: warmup finished metadata ordering')
        if warm and finite(warm_finished):
            check(all(finite(r.get('ended_at_epoch')) and r['ended_at_epoch']<=warm_finished for r in warm),f'{worker}: warmup finished before last warmup')
        rounds=meta.get('rounds',[])
        check([r.get('round') for r in rounds]==[1,2],f'{worker}: missing two round metadata entries')
        if len(rounds)==2 and finite(rounds[0].get('end_at_epoch')) and finite(rounds[1].get('start_at_epoch')):
            check(rounds[0]['end_at_epoch']<=rounds[1]['start_at_epoch'],f'{worker}: outer round ordering')
        for field in ('peak_memory_allocated_gib','peak_memory_reserved_gib'):
            peaks=[r.get(field) for r in rounds]
            check(bool(peaks) and all(finite(p) for p in peaks) and meta.get(field)==max(peaks),f'{worker}: process {field} must be maximum of rounds')
        for rd in rounds:
            reset=rd.get('peak_reset_at_epoch')
            check(finite(reset) and finite(warm_finished) and finite(rd.get('start_at_epoch')) and warm_finished<=reset<=rd['start_at_epoch'],f'{worker}: peak reset timestamp invalid')
            for field in ('peak_memory_allocated_gib','peak_memory_reserved_gib'):
                check(finite(rd.get(field)) and rd[field]>0,f'{worker}: invalid {field}')
            check(finite(rd.get('start_at_epoch')) and finite(rd.get('end_at_epoch')) and rd['end_at_epoch']>=rd['start_at_epoch'],f'{worker}: invalid round boundaries')
        for round_no in (1,2):
            rows=read_rows(base/f'round_{round_no}.jsonl'); formal.extend(rows)
            check(len(rows)==200,f'{worker}/round_{round_no}: expected 200 attempts, got {len(rows)}')
            check([r.get('record_id') for r in rows]==[r.get('record_id') for r in manifest],f'{worker}/round_{round_no}: record ID order mismatch')
            rd=next((r for r in rounds if r.get('round')==round_no),{})
            if rows and finite(rd.get('start_at_epoch')) and finite(rd.get('end_at_epoch')):
                check(all(finite(r.get('started_at_epoch')) and finite(r.get('ended_at_epoch')) and rd['start_at_epoch']<=r['started_at_epoch']<=r['ended_at_epoch']<=rd['end_at_epoch'] for r in rows),f'{worker}/round_{round_no}: rows outside round boundaries')
            for pos,row in enumerate(rows):
                label=f'{worker}/round_{round_no}/row_{pos+1}'
                check(row.get('model')==worker[:2] and row.get('replica')==worker and row.get('round')==round_no and row.get('phase')=='formal',f'{label}: identity mismatch')
                check(row.get('gpu_uuid')==gpu.get('uuid') and str(row.get('gpu_id'))==str(gpu.get('index')),f'{label}: physical GPU mismatch')
                if pos<len(manifest):
                    for field in ('run_order','dataset_index','image_name','source_group'):
                        check(str(row.get(field))==str(manifest[pos].get(field)),f'{label}: {field} mismatch')
                key=tuple(row.get(k) for k in ('run_id','model','replica','round','record_id'))
                check(key not in seen,f'{label}: duplicate result key'); seen.add(key)
        for pos,row in enumerate(warm+ [r for r in formal if r.get('replica')==worker]):
            label=f'{worker}/attempt_{pos+1}'
            start,end=row.get('started_at_epoch'),row.get('ended_at_epoch')
            check(finite(start) and finite(end) and end>=start,f'{label}: invalid timestamps')
            if row.get('status')=='ok':
                check(all(finite(row.get(k)) and row[k]>=0 for k in TIMINGS),f'{label}: invalid successful timing')
                if all(finite(row.get(k)) for k in TIMINGS):
                    check(math.isclose(math.fsum(row[k] for k in STAGES),row['end_to_end_latency_sec'],rel_tol=1e-9,abs_tol=1e-9),f'{label}: timing closure')
                ids=row.get('generated_token_ids')
                check(isinstance(ids,list) and row.get('generated_tokens')==len(ids),f'{label}: token count mismatch')
                if isinstance(ids,list):
                    check(all(isinstance(t,int) for t in ids) and len(ids)<=1024,f'{label}: invalid generated IDs')
                    check(row.get('last_token_id')==(ids[-1] if ids else None),f'{label}: last token mismatch')
                    check(row.get('hit_token_limit')==(len(ids)>=1024),f'{label}: token limit flag mismatch')
                    check(row.get('ended_with_eos')==bool(ids and ids[-1] in (row.get('eos_token_ids') or [])),f'{label}: EOS flag mismatch')
                text=row.get('response_text')
                check(isinstance(text,str),f'{label}: missing response text')
                if isinstance(text,str):
                    check(row.get('predicted_branch')==classify_response(text),f'{label}: prediction classification mismatch')
                    check(row.get('quality_flags')==quality_flags(text),f'{label}: quality flags mismatch')
                check(row.get('error') is None,f'{label}: success contains error')
            else:
                check(row.get('status')=='error' and bool(row.get('error')),f'{label}: invalid failure status/error')
                check(all(row.get(k) is None for k in TIMINGS) and row.get('predicted_branch') is None,f'{label}: failures must have null timings and branch')
    check(len(set(physical))==4 and None not in physical,'four distinct physical GPU UUIDs required')
    check(len({r.get('run_id') for r in formal})==1 and bool(formal and formal[0].get('run_id')),'one nonempty run_id required')
    warm_ends=[r.get('ended_at_epoch') for r in warmups if finite(r.get('ended_at_epoch'))]
    starts=[r.get('started_at_epoch') for r in formal if finite(r.get('started_at_epoch'))]
    ends=[r.get('ended_at_epoch') for r in formal if finite(r.get('ended_at_epoch'))]
    if warm_ends and starts and finite(barrier):
        check(max(warm_ends)<=barrier<=min(starts),'global warmup/barrier/formal ordering violated')
    samples=read_rows(root/'monitoring'/'resource.jsonl')
    task_sizes=[r.get('local_storage',{}).get('task_bytes') for r in samples]
    check(any(finite(n) for n in task_sizes),'no recorded local task storage measurement')
    check(all(not finite(n) or 0<=n<=21474836480 for n in task_sizes),'recorded disk budget exceeded or invalid')
    timestamps=[r.get('sampled_at_epoch') for r in samples if finite(r.get('sampled_at_epoch'))]
    interval=config.get('monitoring_interval_sec',1)
    tolerance=max(2.5*interval,2.5) if finite(interval) and interval>0 else 2.5
    check(bool(timestamps),'missing monitoring timestamps')
    if timestamps and starts and ends:
        check(timestamps==sorted(timestamps),'monitoring timestamps out of order')
        check(min(timestamps)<=min(starts)+tolerance and max(timestamps)>=max(ends)-tolerance,'monitoring does not cover formal interval')
        relevant=[t for t in timestamps if min(starts)-tolerance<=t<=max(ends)+tolerance]
        check(all(b-a<=tolerance for a,b in zip(relevant,relevant[1:])),'monitoring sampling gap')
    monitoring=dict(error_sample_count=sum(bool(r.get('errors')) for r in samples),errors=[dict(sampled_at_epoch=r.get('sampled_at_epoch'),errors=r['errors']) for r in samples if r.get('errors')],missing_gpu_sample_counts={},series_gaps={})
    formal_start=min(starts) if starts else None
    formal_end=max(ends) if ends else None
    formal_samples=[r for r in samples if finite(r.get('sampled_at_epoch')) and formal_start is not None and formal_end is not None and formal_start-tolerance<=r['sampled_at_epoch']<=formal_end+tolerance]
    def check_series(label, series):
        stamps=[r['sampled_at_epoch'] for r in series]
        gaps=[b-a for a,b in zip(stamps,stamps[1:]) if b-a>tolerance]
        monitoring['series_gaps'][label]=gaps
        covered=bool(stamps) and bool(starts) and bool(ends) and min(stamps)<=min(starts)+tolerance and max(stamps)>=max(ends)-tolerance and not gaps
        check(covered,f'monitoring {label} series does not cover formal interval')
    for uuid in physical:
        available=[r for r in formal_samples if any(g.get('uuid')==uuid for g in (r.get('gpus') or []) if isinstance(g,dict))]
        monitoring['missing_gpu_sample_counts'][str(uuid)]=len(formal_samples)-len(available)
        check_series(f'GPU {uuid}',available)
    for name in ('cpu','mem','disk'):
        check_series(name,[r for r in formal_samples if isinstance(r.get(name),dict) and bool(r[name])])
    check(config.get('wall_budget_sec')==14400 and config.get('disk_budget_bytes')==21474836480,'missing or changed approved budgets')
    meta_starts=[m.get('started_at_epoch') for m in metas if finite(m.get('started_at_epoch'))]
    meta_ends=[m.get('ended_at_epoch') for m in metas if finite(m.get('ended_at_epoch'))]
    check(len(meta_starts)==4 and len(meta_ends)==4,'missing worker start/end timestamps')
    if meta_starts and meta_ends: check(max(meta_ends)-min(meta_starts)<=14400,'wall budget exceeded')
    complete=not parse_errors and len(formal)==1600 and len(warmups)==40 and all(m.get('status') in ('complete','completed','complete_with_errors') for m in metas)
    check(complete,'run incomplete')
    return dict(passed=not errors,complete=complete,status=('complete_with_errors' if any(r.get('status')=='error' for r in formal) else 'complete') if not errors else 'incomplete_or_invalid',errors=errors,counts=dict(formal_attempts=len(formal),formal_successes=sum(r.get('status')=='ok' for r in formal),formal_failures=sum(r.get('status')=='error' for r in formal),warmup_attempts=len(warmups),warmup_successes=sum(r.get('status')=='ok' for r in warmups)),monitoring_tolerance_sec=tolerance,monitoring=monitoring,parse_errors=parse_errors)
