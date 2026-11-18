"""Recalculate paired site-family effects and plot either archived G1 or a measured matching run."""
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cellquant.matching_study import (KIND, WARM, PRODUCT, HYBRID, CONTINUED, attribution, explain_matching,
                                     paired_site_gains, study_contrasts)
from cellquant.pilot_bootstrap import analyze_run
from cellquant.runtime import atomic_json, config_hash, read_json, sha256_file


def historical():
    run=ROOT/'evidence/colab_a100_g1_20261118'
    verified=analyze_run(run)
    dist=read_json(run/'G1_distributions.json')
    data=read_json(run/'heldout_blocks.json')
    sites={s:'output' if s.endswith('down_proj') else 'input' for s in dist['sites']}
    contrasts={'cell_vs_hadamard':('cellquant','hadamard_shared'),
               'grid_vs_hadamard':('angle_grid_matched','hadamard_shared'),
               'cell_vs_grid':('cellquant','angle_grid_matched')}
    result=paired_site_gains(data['records'],dist['seeds'],sites,contrasts,rng_seed=verified['bootstrap_seed'])
    return {**result,'scope':'Retrospective, original 7B G1; fixed fits and sites.',
            'source_type':'real_layer','heldout_blocks_content_sha256':config_hash(data)},[
        ('cell_vs_hadamard','CellQuant vs Hadamard'),('grid_vs_hadamard','Grid vs Hadamard'),('cell_vs_grid','CellQuant vs grid')]


def measured_study(run):
    header=read_json(run/'matching_summary.json')
    if header['status']!='STUDY_COMPLETE' or header['source_type']!='real_layer' or header['evidence_kind']!=KIND:
        raise ValueError('A complete real matching study is required')
    raw=[]
    for ref in header['matching_artifacts']:
        parts=Path(ref['path'].replace('\\','/')).parts[-3:]
        if len(parts)!=3 or parts[2]!='matching_result.json' or not parts[0].startswith('seed_') or '..' in parts:
            raise ValueError('Unexpected measurement path')
        path=run.joinpath(*parts)
        if sha256_file(path)!=ref['sha256']:
            raise ValueError(f'Measurement checksum mismatch: {path}')
        row=read_json(path)
        if ((f"seed_{row['seed']}",row['method'])!=parts[:2] or row['source_type']!='real_layer'
                or row['evidence_kind']!=KIND):
            raise ValueError('Measurement identity mismatch')
        for panel in row['heldout']:
            blocks = panel['blocks']
            energy = sum(b['target_energy'] for b in blocks)
            if energy <= 0:
                raise ValueError(f"Undefined held-out NMSE at {panel['panel_id']}")
            nmse = sum(b['squared_error'] for b in blocks) / energy
            if not math.isclose(nmse,panel['nmse'],rel_tol=1e-10,abs_tol=1e-14):
                raise ValueError(f"Held-out block sums disagree with {panel['panel_id']} NMSE")
        row['heldout_error_terms'] = attribution(row['heldout'])
        if any(v['closure_abs_error'] > 1e-12 for v in row['heldout_error_terms'].values()):
            raise ValueError('Error decomposition does not reconstruct measured held-out NMSE')
        raw.append(row)
    seeds=sorted({r['seed'] for r in raw})
    records=[{'seed':r['seed'],'method':r['method'],'panels':{p['panel_id']:p['blocks'] for p in r['heldout']}} for r in raw]
    placements={p['panel_id']:p['placement'] for p in raw[0]['heldout']}
    design=header['design']
    stats=paired_site_gains(records,seeds,placements,study_contrasts(design),
                            repetitions=design['bootstrap_draws'],rng_seed=design['bootstrap_seed'])
    stats['mechanism_by_seed']=explain_matching(raw,WARM,PRODUCT)
    if design.get('refine_continued_grid'):
        stats['hybrid_mechanism_by_seed']=explain_matching(raw,CONTINUED,HYBRID)
    return stats


def plot(stats, labels, title, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,len(labels),figsize=(3.65*len(labels),3.1),sharex=True,sharey=True,layout='constrained')
    for ax,(name,label) in zip(axes,labels):
        ax.set_title(label,fontsize=10)
        comparison=stats['comparisons'][name]
        if 'undefined' in comparison:
            ax.text(.5,.5,'Undefined relative gain',ha='center',transform=ax.transAxes)
            continue
        ax.axvline(0,color='#727985',lw=.8)
        for y,group in enumerate(('output','input','all')):
            values=comparison[group]
            point=values['median_gain_pct']
            lo,hi=values['paired_95_pct']
            ax.plot([lo,hi],[y,y],lw=2.2,color='#087e82')
            ax.plot(point,y,'o',ms=6,color='#12375c')
        ax.set_ylim(-.6,2.6)
        ax.set_yticks([0,1,2],['Output sites','Input sites','All sites'])
        ax.spines[['top','right','left']].set_visible(False)
        ax.grid(axis='x',alpha=.16)
    fig.supxlabel('Median NMSE reduction (%) with paired 95% document intervals',fontsize=10)
    fig.suptitle(f"{title} · {stats['distinct_documents']} documents, {len(stats['seeds_held_fixed'])} fixed seeds",fontsize=12)
    fig.savefig(output,dpi=200)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,help='Completed matching run, including raw per-arm results')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.study:
        labels=[('primary_vs_warm_grid','Matching vs warm grid'),('activation_information','Product vs weight selection'),
                ('vs_continued_grid','Matching vs continued grid')]
        stats={**measured_study(args.study),'source_type':'real_layer','evidence_kind':KIND,
               'run_receipt_sha256':sha256_file(args.study/'matching_summary.json')}
        if 'hybrid_vs_continued_grid' in stats['comparisons']:
            labels.append(('hybrid_vs_continued_grid','Hybrid vs continued grid'))
        title='Output matching on held-out documents'
    else:
        stats,labels=historical()
        title='LLaMA-2-7B · original A100 pilot'
    args.output.mkdir(exist_ok=True,parents=True)
    atomic_json(args.output/'site_effects.json',stats)
    plot(stats,labels,title,args.output/'site_effects.png')
    print(args.output)


if __name__=='__main__':
    main()
