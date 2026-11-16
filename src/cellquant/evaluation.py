"""Token-weighted NLL and pinned lm-evaluation-harness with raw samples."""
from __future__ import annotations
from pathlib import Path
import importlib
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
from .metrics import ppl_from_nll
from .runtime import atomic_json,sha256_file,config_hash,_json_default
from .data import immutable_revision


TASK_METRICS={'arc_challenge':'acc_norm,none','arc_easy':'acc_norm,none','hellaswag':'acc_norm,none',
              'lambada_openai':'acc,none','piqa':'acc_norm,none','winogrande':'acc,none'}
HARNESS_COMMIT='d6de81643928d653435c431bae19945d41d32520'
TASK_SOURCES={
    'arc_challenge':('arc/arc_challenge.yaml','allenai/ai2_arc'),
    'arc_easy':('arc/arc_easy.yaml','allenai/ai2_arc'),
    'hellaswag':('hellaswag/hellaswag.yaml','Rowan/hellaswag'),
    'lambada_openai':('lambada/lambada_openai.yaml','EleutherAI/lambada_openai'),
    'piqa':('piqa/piqa.yaml','baber/piqa'),
    'winogrande':('winogrande/default.yaml','allenai/winogrande')}


def atomic_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.raw-',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=_json_default)+'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def evaluate_nll(model,blocks,path,*,metadata,device='cuda'):
    import torch
    if not blocks:raise ValueError('Empty NLL evaluation')
    model.eval();model.to(device);rows=[]
    with torch.inference_mode():
        for block in blocks:
            ids=torch.tensor(block['tokens'][None,:],device=device)
            if ids.shape[-1]<2:raise ValueError('No predicted target tokens')
            logits=model(ids,use_cache=False).logits[:,:-1].float()
            # Sum in FP64 after unreduced float32 CE; no averaging block PPL.
            nll=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none')
            rows.append({**metadata,**{k:block[k] for k in ['document_id','block_id','start_token','token_ids_sha256']},
                         'nll_sum':float(nll.double().sum()),'valid_target_tokens':int(nll.numel())})
    atomic_jsonl(path,rows)
    return {**ppl_from_nll([r['nll_sum'] for r in rows],[r['valid_target_tokens'] for r in rows]),
            'raw_path':str(Path(path).resolve()),'raw_sha256':sha256_file(path),'document_count':len({r['document_id'] for r in rows})}


def verify_harness(path,commit):
    path=Path(path).resolve()
    if not immutable_revision(commit):raise ValueError('A fixed harness commit is required')
    base=['git','-c',f'safe.directory={path.as_posix()}','-C',str(path)]
    actual=subprocess.check_output(base+['rev-parse','HEAD'],text=True).strip()
    if actual!=commit:raise ValueError('Harness commit mismatch')
    if subprocess.check_output(base+['status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Harness has tracked modifications')
    if str(path) not in sys.path:sys.path.insert(0,str(path))
    module=importlib.import_module('lm_eval')
    if not Path(module.__file__).resolve().is_relative_to(path):raise ValueError('Wrong harness imported; use fresh process')
    return module


def frozen_task_configs(harness_path,task_revisions):
    from lm_eval.tasks._yaml_loader import load_yaml
    if set(task_revisions)!=set(TASK_METRICS):raise ValueError('Every task needs its immutable dataset revision')
    configs=[]
    for task,(filename,repository) in TASK_SOURCES.items():
        revision=task_revisions[task]
        if not immutable_revision(revision):raise ValueError(f'Missing immutable task dataset revision: {task}')
        cfg=load_yaml(Path(harness_path)/'lm_eval/tasks'/filename)
        if cfg['task']!=task or cfg['dataset_path']!=repository:raise ValueError('Task source changed; re-audit required')
        cfg['dataset_kwargs']={**cfg.get('dataset_kwargs',{}),'revision':revision}
        configs.append(cfg)
    return configs


def evaluate_tasks(model,tokenizer,output,*,harness_path,harness_commit,task_metric_keys,task_revisions,metadata,device='cuda'):
    if task_metric_keys!=TASK_METRICS:raise ValueError('The six metric keys are frozen; do not switch after test access')
    if harness_commit!=HARNESS_COMMIT:raise ValueError('Harness API must be re-audited for a different commit')
    module=verify_harness(harness_path,harness_commit)
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    # Validate callable signature instead of silently adapting an unknown API.
    import inspect
    for key in ['model','tasks','num_fewshot','log_samples']:
        if key not in inspect.signature(module.simple_evaluate).parameters:raise RuntimeError('Unsupported harness API; re-audit before execution')
    lm=HFLM(pretrained=model,tokenizer=tokenizer,batch_size=1,device=device)
    configs=frozen_task_configs(harness_path,task_revisions)
    manager=TaskManager(include_defaults=False);tasks=manager.load(configs)['tasks']
    result=module.simple_evaluate(model=lm,tasks=list(tasks.values()),num_fewshot=0,log_samples=True,
                                  task_manager=manager,bootstrap_iters=0,
                                  random_seed=0,numpy_random_seed=0,torch_random_seed=0,fewshot_random_seed=0)
    rows=[];metrics={}
    for task,key in task_metric_keys.items():
        if key not in result['results'][task]:raise ValueError(f'Frozen metric {key} missing from {task}')
        metrics[task]=float(result['results'][task][key]);sample_key=key.split(',')[0]
        samples=result.get('samples',{}).get(task)
        if not samples:raise ValueError(f'Missing per-question predictions: {task}')
        for sample in samples:
            contribution=sample.get(sample_key)
            if not isinstance(contribution,(int,float)) or isinstance(contribution,bool):
                # Bool is a valid per-item correctness value, not a gate metric.
                if isinstance(contribution,bool):contribution=int(contribution)
                else:raise ValueError(f'Missing per-item {sample_key}: {task}')
            obj=tasks[task];doc=sample['doc'];filtered=sample['filtered_resps']
            gold=sample.get('target')
            if obj.OUTPUT_TYPE=='multiple_choice':
                choices=obj.doc_to_choice(doc);scores=np.array([r[0] for r in filtered])
                if sample_key=='acc_norm':scores=scores/np.array([len(choice) for choice in choices])
                pred=int(np.argmax(scores));gold=obj.doc_to_text(doc) if obj.multiple_input else obj.doc_to_target(doc)
                if isinstance(gold,str):gold=choices.index(gold)
                expected=float(pred in gold if isinstance(gold,list) else pred==gold)
                if expected!=float(contribution):raise ValueError('Stored prediction disagrees with frozen harness metric')
                prediction={'choice_index':pred,'choice_text':choices[pred]}
            else:
                prediction={'target_loglikelihood':float(filtered[0][0]),'target_is_greedy_exact_match':bool(filtered[0][1]),
                            'kind':'Teacher-forced LAMBADA target scoring; no generated completion recorded'}
            rows.append({**metadata,'task_id':task,'harness_commit':harness_commit,'metric_key':key,
                'example_id':str(sample['doc_id']),'metric_contribution':float(contribution),'task_dataset_revision':task_revisions[task],
                'gold':gold,'prediction':prediction,'scores':sample.get('resps'),
                'prompt_sha256':config_hash(sample.get('arguments',[])),'raw_sample':sample})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'task_predictions.jsonl';atomic_jsonl(path,rows)
    from lm_eval.utils import handle_non_serializable
    exported=json.loads(json.dumps({k:v for k,v in result.items() if k!='samples'},default=handle_non_serializable,allow_nan=False))
    exported['verified_harness_commit']=harness_commit
    atomic_json(output/'harness_results.json',exported)
    return {'task_accuracy':metrics,'mean_task_accuracy':float(np.mean(list(metrics.values()))),
            'raw_path':str(path.resolve()),'raw_sha256':sha256_file(path),'harness_commit':harness_commit,'task_dataset_revisions':task_revisions}
