"""Prepare all methods/seeds together so matched runs share one frozen protocol.

First prepare without --include-tests. After measured G2, prepare that frozen
matrix into a new directory with --include-tests, then bind a new measured G3.
Metadata resolution/tokenization is real; this script never creates gate PASS.
"""
from pathlib import Path
import argparse
import copy
from prepare_experiment import prepare
from cellquant.runtime import atomic_json,read_json,sha256_file


def prepare_matrix(matrix_path,output,*,include_tests=False):
    matrix=read_json(matrix_path);output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh immutable matrix preparation directory')
    output.mkdir(parents=True,exist_ok=True)
    jobs=matrix['jobs'];models=sorted({j['model'] for j in jobs});prepared={}
    for model in models:
        selected=[j for j in jobs if j['model']==model]
        job=next(j for j in selected if j['method']=='cellquant')
        cfg=read_json(job['config_path']);cfg['seeds']=sorted({j['seed'] for j in selected})
        cfg['solver'].pop('schedule_seed',None)
        cfg['solver'].update(matrix.get('solver', {}))
        # Preserve upstream full R1+R2 CE budget with a dedicated train pool.
        training=[read_json(j['config_path']).get('spinquant_training') for j in selected if j['method']=='spinquant_full']
        if not training or not all(training):raise ValueError('Explicit full SpinQuant training protocol is required')
        if any(t!=training[0] for t in training):raise ValueError('Inconsistent full SpinQuant training protocol')
        cfg['spinquant_training']=training[0]
        slug=model.replace('/','__');draft=atomic_json(output/(slug+'_input.json'),cfg)
        prepared[model]=read_json(prepare(draft,output/slug,include_tests=include_tests))
    all_data={}
    for model,cfg in prepared.items():
        all_data[model]={s:{'path':p,'sha256':sha256_file(p)} for s,p in cfg['data']['token_manifests'].items()}
    bundle=atomic_json(output/'data_bundle.json',{'schema_version':'1.0','kind':'token_manifest_bundle','models':all_data})
    protocol=atomic_json(output/'protocol.json',{'schema_version':'1.0','status':'FROZEN',
        'models':{m:{'path':c['protocol_manifest_path'],'sha256':c['protocol_sha256']} for m,c in prepared.items()},
        'jobs':[{k:j[k] for k in ['job_id','model','method','seed']} for j in jobs],
        'data_bundle_sha256':sha256_file(bundle),'test_tokens_prepared':include_tests,
        'note':'Fixed cohort; test token preparation is distinct from scoring. G3 must bind this protocol.'})
    dtypes={c['runtime']['dtype'] for c in prepared.values()}
    if len(dtypes)!=1:raise ValueError('Cross-model matrix requires the same declared dtype')
    provenance=atomic_json(output/'execution_provenance.json',{
        'models':{m:{k:c[k] for k in ['model_revision','tokenizer_revision']} for m,c in prepared.items()},
        'data_manifest_path':str(bundle),'data_manifest_sha256':sha256_file(bundle),
        'protocol_manifest_path':str(protocol),'protocol_sha256':sha256_file(protocol),'dtype':next(iter(dtypes))})
    new_jobs=[]
    for j in jobs:
        cfg=copy.deepcopy(prepared[j['model']]);cfg.update(job_id=j['job_id'],method=j['method'],seed=j['seed'])
        cfg.pop('seeds',None);cfg['data']['token_manifest_path']=cfg['data']['token_manifests'][str(j['seed'])]
        cfg.update(protocol_manifest_path=str(protocol),protocol_sha256=sha256_file(protocol),execution_provenance_path=str(provenance))
        cfg['result_path']=str(output/'results'/j['job_id']/'result.json')
        if include_tests:cfg['evaluation']['formal_test_access']=True
        path=atomic_json(output/'jobs'/(j['job_id']+'.json'),cfg)
        new_jobs.append({**j,'config_path':str(path),'result_path':cfg['result_path']})
    matrix.update(jobs=new_jobs,execution_provenance_path=str(provenance),protocol_status='FROZEN',
                  protocol_manifest_path=str(protocol),protocol_sha256=sha256_file(protocol))
    path=atomic_json(output/'matrix.json',matrix);print(f'Prepared matched matrix: {path}');return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--include-tests',action='store_true')
    a=p.parse_args();prepare_matrix(a.config,a.output,include_tests=a.include_tests)
