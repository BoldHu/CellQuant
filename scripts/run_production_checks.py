"""Measured synthetic G0 and optional full-dimensional engineering smoke."""
import argparse
import math
import time
from pathlib import Path
import numpy as np
from cellquant.solver import (CalibrationPanel,PanelState,PanelOrbit,SolverConfig,solve_pair,fit_shared_rotation)
from cellquant.quantization import QuantSpec,quantize
from cellquant.events import WorkBudget
from cellquant.math_reference import givens,trig_value,PairProblem,solve_pair_reference
from cellquant.runtime import atomic_json,sha256_file,environment_report
from cellquant.gates import evaluate_file


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in [11,23,37]:
        rng=np.random.default_rng(seed)
        for case in range(8):
            placement='input' if case<4 else 'output';d=4;g=2 if case%2 else 4
            w=rng.normal(size=(2,d) if placement=='input' else (d,8));x=rng.normal(size=(w.shape[1],2))
            if case in [3,7]:w[0]=0;x[:,0]=0
            pair=(0,1) if case%2 else (0,3);spec=QuantSpec(group_size=g)
            p=CalibrationPanel(str(case),placement,w,x);st=PanelState(p,np.eye(d),spec,spec)
            limits={'budget_w':st.bw,'budget_a':st.ba};cfg=SolverConfig(theta_limit=.25)
            result=solve_pair([st],pair,limits,cfg)
            cfg.screen=False;full=solve_pair([st],pair,limits,cfg)
            orb=PanelOrbit(st,pair);events={-.25,.25}
            for group in orb.groups:events.update(group.enumerate(-.25,.25,True,WorkBudget())[0])
            error=0.
            ev=sorted(events)
            for lo,hi in zip(ev[:-1],ev[1:]):
                for k in range(len(orb.groups)):orb.replace(k,(lo+hi)/2)
                theta=lo+(hi-lo)*.37
                error=max(error,abs(float(trig_value(orb.loss_coeff,theta))-st.direct(pair,theta)['loss']))
            q=givens(d,pair,.123)
            lhs=w@q.T@q@x if placement=='input' else q.T@(q@w)@x
            prod=float(np.linalg.norm(lhs-w@x)/max(np.linalg.norm(w@x),1e-12))
            ref=None
            if placement=='input':ref=solve_pair_reference(PairProblem(w,x,pair,group_size=g,left=-.25,right=.25))
            records.append({'seed':seed,'case':case,'placement':placement,'shape':[list(w.shape),list(x.shape)],
                'fp64_product_rel_error':prod,'orthogonality':float(np.linalg.norm(q.T@q-np.eye(d))/math.sqrt(d)),
                'cell_error':error,'screen_error':abs(result['best']['loss']-full['best']['loss'])/max(result['best']['loss'],1e-30),
                'reference_error':None if ref is None else abs(ref['best']['loss']-result['best']['loss']),
                'budget_violation':any(result['best'][k]>limits[k]+1e-10 for k in limits),
                'increase':result['best']['loss']>result['baseline']['loss']+1e-12,'solver':result})
    boundary=quantize(np.array([[7.,.5,1.5,-2.5],[0.,0.,0.,0.]]))[1]
    rounding=bool(np.array_equal(boundary,np.array([[7,0,2,-2],[0,0,0,0]])))
    # Additional isolated tie regression is independently executed by pytest.
    from cellquant.events import GroupOrbit
    tie=GroupOrbit('w',0,0,np.array([[0.,5.,0.],[0.,0.,5.],[5.,0.,0.]]),QuantSpec(),True)
    rounding=rounding and bool(np.array_equal(tie.scale_coeff(0.),[5/7,0.,0.]))
    raw=atomic_json(output/'cases.json',{'source_type':'synthetic_math','cases':records,'nearest_even_and_isolated_tie':rounding})
    metrics={'math_seed_count':len({r['seed'] for r in records}),'math_case_count':len(records),
        'fp64_product_rel_error_max':max(r['fp64_product_rel_error'] for r in records),
        'fp64_orthogonality_rel_error_max':max(r['orthogonality'] for r in records),
        'cell_loss_prediction_abs_error_max':max(r['cell_error'] for r in records),
        'screened_vs_full_loss_rel_error_max':max(r['screen_error'] for r in records),
        'budget_violation_count':sum(r['budget_violation'] for r in records),
        'accepted_update_increase_count':sum(r['increase'] for r in records),
        'rounding_boundary_cases_pass':rounding and all(r['solver']['boundary_rounding_evaluations']>=2 for r in records),
        'input_output_direction_cases_pass':{r['placement'] for r in records}=={'input','output'} and all(r['fp64_product_rel_error']<1e-10 for r in records),
        'reference_implementation_audited':all(r['reference_error']<1e-8 for r in records if r['reference_error'] is not None)}
    evidence={'schema_version':'1.0','gate_id':'G0','source_type':'synthetic_math','metrics':metrics,
        'provenance':{'environment':environment_report(),'source_hashes':{p.name:sha256_file(p) for p in Path('src/cellquant').glob('*.py')},
                      'interval_certified':False},'artifact_refs':[{'path':str(raw),'sha256':sha256_file(raw)}]}
    atomic_json(output/'G0_evidence.json',evidence)
    gate=evaluate_file('G0',output/'G0_evidence.json');atomic_json(output/'G0.json',gate)
    print('G0:',gate['status']);return gate


def dimensional(output,d=4096,f=11008):
    # Full shapes, few tokens, tiny preregistered interval: an engineering smoke,
    # not evidence about LLM weights or useful full-range calibration throughput.
    rng=np.random.default_rng(20261115);start=time.perf_counter()
    panels=[CalibrationPanel('full_input','input',rng.normal(size=(2,d)),rng.normal(size=(d,2))),
            CalibrationPanel('full_down','output',rng.normal(size=(d,f)).astype(np.float32),rng.normal(size=(f,2)))]
    cfg=SolverConfig(theta_limit=.0001,seconds_per_pair=120.,wall_seconds=300.)
    result=fit_shared_rotation(panels,np.eye(d),QuantSpec(group_size=-1),[(3,128)],config=cfg)
    result.pop('q');result.update(source_type='synthetic_math',hidden_dimension=d,ffn_dimension=f,
        complete_groups=True,total_seconds=time.perf_counter()-start,scope='full-shaped synthetic engineering smoke; 2 tokens, narrow angle interval')
    atomic_json(Path(output)/'dimension_4096.json',result);print('Full dimension:',result['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='runs/production_checks');p.add_argument('--full-dimension',action='store_true')
    a=p.parse_args();run(a.output)
    if a.full_dimension:dimensional(a.output)
