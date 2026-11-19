"""Verify the A100 row measurements against the original document pilot and repeat the paired inference."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from cellquant.bilateral_documents import COHORT_FILE, DESIGN, FIT_FILE, GRID_FILE, block_identity, checked_json
from cellquant.bilateral_rows import PRECISIONS, PREFIX, ROW_SETS, SITE, read_reference, summarize_rows
from cellquant.bilateral_screen import DESIGN as ANGLES
from cellquant.data import token_hash
from cellquant.matching_study import CONTINUED
from cellquant.row_crossfit import paired_row_effect
from cellquant.runtime import atomic_json, config_hash, read_json, sha256_file


def audit(path, refs):
    seeds, candidate = ANGLES['seeds'], ANGLES['arms'][1]
    document_zip = refs/'bilateral_documents_7b.zip'
    reference = read_reference(document_zip, refs/GRID_FILE, refs/FIT_FILE, DESIGN, seeds, (candidate,CONTINUED))
    fitted = read_json(refs/'bilateral_fit_7b_summary.json')['by_seed']
    cohort = read_json(refs/COHORT_FILE)
    rows_by_seed, observed = {}, {}
    with ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError('Damaged row archive')
        result = json.loads(z.read('run/result.json'))
        cfg = json.loads(z.read('run/config.json'))
        protocol = checked_json(z,'inputs/protocol.json',result['protocol_sha256'])
        if (result['status'] != 'DIAGNOSTIC_COMPLETE' or result['source_type'] != 'real_layer'
                or result['evidence_kind'] != 'FROZEN_ROW_DIAGNOSTIC' or result['execution_mode'] != 'bilateral_row_diagnostic'
                or result['site'] != SITE or result['document_reference_sha256'] != sha256_file(document_zip)
                or result['fit_archive_sha256'] != sha256_file(refs/FIT_FILE) or result['reference_archive_sha256'] != sha256_file(refs/GRID_FILE)
                or config_hash(cfg) != result['config_sha256'] or protocol['bilateral_document_pilot'] != result['fit_source']
                or protocol['bilateral_document_pilot']['design'] != DESIGN or cfg['seeds'] != seeds
                or result['bootstrap_seed'] != DESIGN['bootstrap_seed'] or result['bootstrap_draws'] != DESIGN['bootstrap_draws']
                or set(result['row_artifacts']) != {f'seed_{s}/row_diagnostic.json' for s in seeds}):
            raise ValueError('Wrong experiment, references or analysis design')
        fingerprint = json.loads(z.read('run/execution_fingerprint.json'))
        if fingerprint['status'] != 'VERIFIED' or config_hash(fingerprint['payload']) != fingerprint['sha256']:
            raise ValueError('Invalid execution fingerprint')
        for seed in seeds:
            name = f'seed_{seed}/row_diagnostic.json'
            row = checked_json(z,'run/'+name,result['row_artifacts'][name])
            manifest = checked_json(z,f'inputs/seed_{seed}/data_manifest.json',protocol['token_manifest_sha256'][str(seed)])
            all_blocks = manifest['partitions']
            dev = all_blocks['dev']
            if [block_identity(b) for b in dev] != cohort['heldout'][str(seed)]:
                raise ValueError('Document identities differ from the document pilot')
            tokens = z.read(f'inputs/seed_{seed}/token_ids.npz')
            if hashlib.sha256(tokens).hexdigest() != manifest['tokens_sha256']:
                raise ValueError('Changed token archive')
            with np.load(io.BytesIO(tokens),allow_pickle=False) as cache:
                if any(token_hash(cache[b['array_key']]) != b['token_ids_sha256'] for blocks in all_blocks.values() for b in blocks):
                    raise ValueError('Invalid archived token sequence')
            if (row['seed'] != seed or row['site'] != SITE or set(row['methods']) != {candidate,CONTINUED,PREFIX}
                    or row['sampled_row_ids'] != np.linspace(0,11007,64,dtype=int).tolist()):
                raise ValueError('Unmatched rows, fits or projections')
            for method, measurement in row['methods'].items():
                expected_q = fitted[str(seed)]['replayed_q_sha256']['prefix' if method==PREFIX else method]
                counts = measurement['row_counts']
                if (measurement['q_sha256'] != expected_q or counts != {'sampled':64,'complement':10944,'all_rows':11008}
                        or counts != result['row_counts_by_seed'][str(seed)] or set(measurement['blocks']) != set(ROW_SETS)):
                    raise ValueError('Row counts or fitted rotation differ')
                blocks = measurement['blocks']
                for subset in blocks.values():
                    if [(b['block_id'],b['document_id'],b['sampled_tokens']) for b in subset] != [
                            (d['block_id'],d['document_id'],DESIGN['tokens_per_document']) for d in dev]:
                        raise ValueError('Unpaired articles or token counts')
                for a,b,c in zip(*(blocks[g] for g in ROW_SETS)):
                    for k in (*PRECISIONS,'target_energy'):
                        np.testing.assert_allclose(a[k]+b[k],c[k],rtol=1e-12,atol=1e-13)
                if method != PREFIX:
                    old = reference[seed,method]
                    if row['heldout_panel_sha256'] != old['heldout_panel_sha256']:
                        raise ValueError('Held-out panel hash changed')
                    panel = next(p for p in old['heldout'] if p['panel_id'] == SITE)
                    for a,b in zip(blocks['sampled'],panel['blocks']):
                        np.testing.assert_allclose([a['W4A4'],a['target_energy']], [b['squared_error'],b['target_energy']],rtol=1e-9,atol=1e-13)
            raw = row['methods']
            observed[str(seed)] = {}
            for group in ROW_SETS:
                a,b = (raw[m]['blocks'][group] for m in (candidate,CONTINUED))
                observed[str(seed)][group] = {'candidate_worse_documents':sum(x['W4A4'] > y['W4A4'] for x,y in zip(a,b)),
                    'grid_nmse':sum(x['W4A4'] for x in b)/sum(x['target_energy'] for x in b),
                    'candidate_nmse':sum(x['W4A4'] for x in a)/sum(x['target_energy'] for x in a)}
            rows_by_seed[seed] = row
    recomputed = summarize_rows(rows_by_seed,candidate,DESIGN)
    gap = {}
    for label, new, base in (('candidate_vs_full_grid',candidate,CONTINUED),('prefix_vs_full_grid',PREFIX,CONTINUED),
                             ('candidate_vs_prefix',candidate,PREFIX)):
        gap[label] = {}
        for precision in PRECISIONS:
            paired = paired_row_effect(rows_by_seed,new,base,ROW_SETS,precision,difference=('complement','sampled'),
                                       draws=DESIGN['bootstrap_draws'],rng_seed=DESIGN['bootstrap_seed'])
            gap[label][precision] = paired['difference_pp']
            for group in ROW_SETS:
                saved = result['row_comparisons'][label][group][precision]
                effect = recomputed[label][group][precision]
                np.testing.assert_allclose([saved['median_gain_pct'],*saved['paired_95_pct'],*(saved['seed_medians_pct'][str(s)] for s in seeds)],
                                           [effect['median_gain_pct'],*effect['paired_95_pct'],*(effect['seed_medians_pct'][str(s)] for s in seeds)],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose([paired[group]['estimate'],*paired[group]['paired_95']],
                                           [effect['median_gain_pct'],*effect['paired_95_pct']],rtol=1e-10,atol=1e-10)
    return {'source_type':'real_layer','run_id':result['run_id'],'archive_sha256':sha256_file(path),
            'document_reference_sha256':result['document_reference_sha256'],'code_git':fingerprint['payload']['code_git'],
            'device':fingerprint['payload']['environment']['cuda']['devices'][0]['name'],'wall_seconds':result['resources']['wall_seconds'],
            'site':SITE, 'scope':'post-hoc diagnostic on the same 32 articles; two fixed calibration seeds',
            'rows_by_seed':result['row_counts_by_seed'],'row_comparisons':recomputed,'complement_minus_sampled_gain_pp':gap,
            'observations_by_seed':observed}


def plot_audit(summary, path):
    import matplotlib.pyplot as plt
    comparison = summary['row_comparisons']['candidate_vs_full_grid']
    fig, axes = plt.subplots(1,2,figsize=(8.5,3.5),layout='constrained')
    labels = [('W4A4','Both sides INT4'), ('W4Afp','Weight only'), ('WfpA4','Activation only')]
    panels = [[('sampled','64 fitted rows','#286480',0.)],
              [('complement','Other 10,944 rows','#286480',.09),('all_rows','All 11,008 rows','#b94e27',-.09)]]
    for axis, series, title in zip(axes,panels,['Fitted rows','Remaining rows and full projection']):
        axis.axvline(0,color='.45',linewidth=1,linestyle='--')
        for group,label,color,offset in series:
            for i,(precision,_) in enumerate(labels):
                metric = comparison[group][precision]
                m, (lo,hi) = metric['median_gain_pct'],metric['paired_95_pct']
                axis.errorbar(m,2-i+offset,xerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color=color,label=label if i==0 else None)
        axis.set(yticks=[2,1,0],yticklabels=[name for _,name in labels],ylim=(-1.02,2.4),
                 xlabel='Relative NMSE reduction (%)',title=title)
        axis.spines[['right','top']].set_visible(False)
        axis.legend(frameon=False,loc='lower left',fontsize=8)
    fig.suptitle('LLaMA-2-7B, first MLP up projection: bilateral angles vs. 256-pair grid',fontsize=12)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    refs = Path(__file__).resolve().parents[1]/'evidence/output_matching_20261119'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path,nargs='?',default=refs/'bilateral_rows_7b.zip')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figure',type=Path)
    args = parser.parse_args()
    summary = audit(args.archive,refs)
    print(atomic_json(args.output,summary))
    if args.figure:
        plot_audit(summary,args.figure)
