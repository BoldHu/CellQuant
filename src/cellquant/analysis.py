"""Verified raw-result analysis, paired seed/document CIs and incomplete gates.

No published numbers, imputations or synthetic demonstration rows are accepted.
Unmeasured audits/ablations remain missing G4 fields instead of defaulting true.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
import json
import math
import numpy as np
from .runtime import atomic_json,read_json,sha256_file
from .gates import evaluate_file
from .metrics import ppl_from_nll


def verified_results(paths):
    rows=[];rejected=[];seen=set()
    for path in paths:
        r=read_json(path)
        if r.get('source_type')!='real_model' or r.get('status')!='COMPLETE' or r.get('evidence_kind')!='REAL_MODEL_EXPERIMENT':
            rejected.append({'path':str(path),'reason':'Not a completed real-model measurement'});continue
        if r['run_id'] in seen:continue  # canonical result and job index may reference the same run
        seen.add(r['run_id'])
        for name,ref in r.get('raw',{}).items():
            if not Path(ref['raw_path']).is_file() or sha256_file(ref['raw_path'])!=ref['raw_sha256']:raise ValueError('Raw result checksum mismatch: '+str(path))
        r=dict(r,source_path=str(Path(path).resolve()),source_sha256=sha256_file(path));rows.append(r)
    return rows,rejected


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def make_paired_cell(pairs,dataset,*,tasks=False):
    """pairs: ordered (baseline_run, candidate_run), one entry per shared seed."""
    clusters=None;delta=[];counts=[];base_means=[];new_means=[]
    for base,new in pairs:
        if (base['model_id'],base['seed'],base['protocol_sha256'],base['dtype'])!=(new['model_id'],new['seed'],new['protocol_sha256'],new['dtype']):
            raise ValueError('Unmatched model/seed/protocol/dtype')
        key='tasks' if tasks else dataset
        b=read_jsonl(base['raw'][key]['raw_path']);n=read_jsonl(new['raw'][key]['raw_path'])
        if tasks:
            b=[r for r in b if r['task_id']==dataset];n=[r for r in n if r['task_id']==dataset]
        id_key='example_id' if tasks else 'block_id'
        bm={r[id_key]:r for r in b};nm={r[id_key]:r for r in n}
        if len(bm)!=len(b) or len(nm)!=len(n) or bm.keys()!=nm.keys():raise ValueError('Missing/duplicated paired evaluation units')
        groups={}
        for ident,br in bm.items():
            nr=nm[ident]
            if tasks:
                if br['metric_key']!=nr['metric_key'] or br['harness_commit']!=nr['harness_commit'] or br['prompt_sha256']!=nr['prompt_sha256'] or br.get('task_dataset_revision')!=nr.get('task_dataset_revision'):
                    raise ValueError('Unmatched task metric/harness/prompt')
                cluster=ident;bval=br['metric_contribution'];nval=nr['metric_contribution'];count=1
            else:
                if br['token_ids_sha256']!=nr['token_ids_sha256'] or br['valid_target_tokens']!=nr['valid_target_tokens'] or br['document_id']!=nr['document_id']:
                    raise ValueError('Unmatched tokenization/counts/document')
                cluster=br['document_id'];bval=br['nll_sum'];nval=nr['nll_sum'];count=br['valid_target_tokens']
            if not all(math.isfinite(v) for v in [bval,nval,count]) or count<=0:raise ValueError('Invalid raw metric')
            group=groups.setdefault(cluster,[0.,0.,0.]);group[0]+=bval;group[1]+=nval;group[2]+=count
        if clusters is None:clusters=sorted(groups)
        if sorted(groups)!=clusters:raise ValueError('Evaluation documents must be fixed across calibration seeds')
        arr=np.array([groups[k] for k in clusters]);delta.append(arr[:,1]-arr[:,0]);counts.append(arr[:,2])
        base_means.append(float(arr[:,0].sum()/arr[:,2].sum()));new_means.append(float(arr[:,1].sum()/arr[:,2].sum()))
    if len(clusters or [])<2:raise ValueError('At least two independent paired clusters required')
    return {'cluster_key':dataset,'ids':clusters,'delta_sums':np.array(delta),'counts':np.array(counts),
            'base_means':base_means,'new_means':new_means,'seeds':[b['seed'] for b,n in pairs]}


def paired_cluster_bootstrap(cells,*,repetitions=10000,seed=20261115):
    """Shared seed draws and shared document/item draws across model cells.

Each model x dataset/task contributes equally. Tokens weight NLL inside a cell
only. All cells sharing a dataset share draws of the union of document IDs.
"""
    if not cells or repetitions<100:raise ValueError('Nonempty cells and >=100 resamples required')
    seeds=cells[0]['seeds']
    if any(c['seeds']!=seeds for c in cells):raise ValueError('Identical ordered calibration seeds required across cells')
    unions={key:sorted(set().union(*(set(c['ids']) for c in cells if c['cluster_key']==key))) for key in {c['cluster_key'] for c in cells}}
    expanded=[]
    for cell in cells:
        union=unions[cell['cluster_key']];d=np.zeros((len(seeds),len(union)));n=np.zeros_like(d)
        index={k:i for i,k in enumerate(union)}
        for j,ident in enumerate(cell['ids']):d[:,index[ident]]=cell['delta_sums'][:,j];n[:,index[ident]]=cell['counts'][:,j]
        expanded.append((cell['cluster_key'],d,n))
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(repetitions):
        si=rng.integers(len(seeds),size=len(seeds))
        sampled={k:rng.integers(len(v),size=len(v)) for k,v in unions.items()}
        values=[]
        for key,d,n in expanded:
            ix=np.ix_(si,sampled[key]);den=n[ix].sum(1)
            if np.any(den<=0):raise ValueError('Bootstrap sampled no tokens for a cell; evaluation cluster design is insufficient')
            values.append((d[ix].sum(1)/den).mean())
        draws.append(np.mean(values))
    effects=[float((c['delta_sums'].sum(1)/c['counts'].sum(1)).mean()) for c in cells]
    low,high=np.quantile(draws,[.025,.975])
    return {'delta':float(np.mean(effects)),'ci95_low':float(low),'ci95_high':float(high),
            'cell_deltas':effects,'seeds':seeds,'repetitions':repetitions,'unit':'seed_and_document_or_item_cluster'}


def analyze(paths,output,*,baseline='spinquant_full',candidate='cellquant',selection_manifest=None,repetitions=10000):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'main_table.csv').exists():raise ValueError('Analysis directory already contains real-result tables; use a new output directory to preserve provenance')
    rows,rejected=verified_results(paths);metrics={};comparisons=[];table=[]
    by={}
    for r in rows:
        key=(r['model_id'],r['method'],r['seed'])
        if key in by:raise ValueError('Multiple completed protocols/runs for same model/method/seed; select one frozen cohort')
        by[key]=r
        for dataset,ref in r['raw'].items():
            if dataset=='tasks':continue
            raw=read_jsonl(ref['raw_path']);measured=ppl_from_nll([v['nll_sum'] for v in raw],[v['valid_target_tokens'] for v in raw])
            table.append({'model_id':r['model_id'],'method':r['method'],'seed':r['seed'],'dataset':dataset,
                          'ppl':measured['ppl'],'nll_per_token':measured['nll_per_token'],'calibration_seconds':r['total_calibration_seconds'],
                          'peak_gpu_gib':r['peak_gpu_allocated_bytes']/2**30,'source_path':r['source_path']})
    if table:
        with (output/'main_table.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for model in sorted({r['model_id'] for r in rows}):
            fig,ax=plt.subplots(figsize=(7,4))
            for method in sorted({r['method'] for r in rows}):
                data=[r for r in table if r['model_id']==model and r['method']==method and r['dataset']=='wikitext2_test']
                if data:ax.scatter([r['calibration_seconds'] for r in data],[r['ppl'] for r in data],label=method)
            ax.set(xlabel='Measured total calibration seconds',ylabel='WikiText-2 PPL',title=model)
            if ax.collections:ax.legend(fontsize=7)
            fig.tight_layout();fig.savefig(output/(model.split('/')[-1]+'_cost_quality.png'),dpi=180);plt.close(fig)
    cells=[];task_cells=[];models=sorted({r['model_id'] for r in rows});seed_sets=[]
    for model in models:
        if len({r['protocol_sha256'] for r in rows if r['model_id']==model})>1:
            raise ValueError('Different protocols for the same model; prepare one shared cohort before pooling seeds/methods')
        base_seeds={s for m,b,s in by if m==model and b==baseline};new_seeds={s for m,b,s in by if m==model and b==candidate}
        if not base_seeds or base_seeds!=new_seeds:continue
        seeds=sorted(base_seeds);seed_sets.append(seeds);pairs=[(by[model,baseline,s],by[model,candidate,s]) for s in seeds]
        for dataset in ['wikitext2_test','c4_test']:
            if all(dataset in b['raw'] and dataset in n['raw'] for b,n in pairs):
                cell=make_paired_cell(pairs,dataset);cells.append(cell);comparisons.append({'model':model,'dataset':dataset,'seeds':seeds})
        from .evaluation import TASK_METRICS
        if all('tasks' in b['raw'] and 'tasks' in n['raw'] for b,n in pairs):
            for task in TASK_METRICS:task_cells.append(make_paired_cell(pairs,task,tasks=True))
    if cells and all(c['seeds']==cells[0]['seeds'] for c in cells):
        ci=paired_cluster_bootstrap(cells,repetitions=repetitions);atomic_json(output/'paired_nll.json',{'comparisons':comparisons,**ci})
        model_deltas={m:[delta for meta,delta in zip(comparisons,ci['cell_deltas']) if meta['model']==m] for m in {p['model'] for p in comparisons}}
        metrics.update(pooled_geomean_ppl_gain_pct=100*(1-math.exp(ci['delta'])),pooled_nll_delta_ci95_high=ci['ci95_high'],
            min_model_mean_ppl_gain_pct=min(100*(1-math.exp(float(np.mean(v)))) for v in model_deltas.values()),
            max_ppl_cell_regression_pct=max(100*math.expm1(v) for v in ci['cell_deltas']),
            min_calibration_seed_count=len(ci['seeds']),ppl_model_dataset_cell_count=len(cells),paired_ci_method_valid=True)
    if task_cells and all(c['seeds']==task_cells[0]['seeds'] for c in task_cells):
        ci=paired_cluster_bootstrap(task_cells,repetitions=repetitions);atomic_json(output/'paired_tasks.json',ci)
        metrics.update(mean_task_accuracy_gain_pp=100*ci['delta'],mean_task_accuracy_delta_ci95_low_pp=100*ci['ci95_low'],
                       zero_shot_task_count=len({c['cluster_key'] for c in task_cells}))
    if rows:
        metrics.update(main_model_count=len(models),main_model_ids_correct=set(models)=={'meta-llama/Llama-2-7b-hf','meta-llama/Meta-Llama-3-8B'},
            paper_results_separated_from_own_runs=True,extra_online_r1_ops=max(r['extra_online_r1_ops'] for r in rows),
            performance_claims_match_backend=all(r['backend']=='dequantized_float_simulation' for r in rows))
    if selection_manifest:
        selection=read_json(selection_manifest)
        # A frozen selection must bind exact protocols, not choose a weaker
        # baseline after reading test results. It does not prove recent-method audits.
        metrics['strongest_baseline_selection_frozen']=(selection.get('status')=='FROZEN' and selection.get('baseline_method')==baseline
            and set(r['protocol_sha256'] for r in rows)<=set(selection.get('protocol_hashes',[])))
    evidence={'schema_version':'1.0','gate_id':'G4','source_type':'real_model' if rows else 'not_run','metrics':metrics,
        'provenance':{'baseline':baseline,'candidate':candidate,'raw_hashes_verified':len(rows)} if rows else {},
        'artifact_refs':[{'path':r['source_path'],'sha256':r['source_sha256']} for r in rows]}
    atomic_json(output/'G4_evidence.json',evidence);gate=evaluate_file('G4',output/'G4_evidence.json');atomic_json(output/'G4.json',gate)
    result={'status':gate['status'],'accepted_runs':len(rows),'rejected':rejected,'missing':gate.get('missing_fields',[]),
            'claims':[{'claim':'Measured paired PPL effect','supported':bool(cells)},
                      {'claim':'Packed INT4 deployment acceleration','supported':False,'reason':'Float simulation backend'},
                      {'claim':'Novel method/competitive paper','supported':False,'reason':'Requires G4 plus current primary-source novelty audit'}]}
    atomic_json(output/'analysis.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',default='runs');p.add_argument('--output',default='runs/paper_analysis')
    p.add_argument('--baseline',default='spinquant_full');p.add_argument('--selection-manifest');p.add_argument('--repetitions',type=int,default=10000)
    a=p.parse_args();print(json.dumps(analyze(sorted(Path(a.runs).rglob('result.json')),a.output,baseline=a.baseline,
        selection_manifest=a.selection_manifest,repetitions=a.repetitions),ensure_ascii=False,indent=2))
