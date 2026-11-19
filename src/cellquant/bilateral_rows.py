"""Post-hoc row generalization and one-sided RTN errors at the first MLP up projection."""
import json
from zipfile import ZipFile

import numpy as np
import torch

from .bilateral_documents import checked_json
from .matching_batches import q_digest
from .matching_study import CONTINUED, paired_site_gains
from .progress import report
from .quantization import torch_quantize
from .runtime import sha256_file
from .solver import panel_specs


SITE = 'layer_0.up_proj'
PREFIX = 'grid_prefix'
ROW_SETS = ('sampled', 'complement', 'all_rows')
PRECISIONS = ('W4A4', 'W4Afp', 'WfpA4')
TILE_ROWS = 256


def read_reference(path, grid_archive, fit_archive, design, seeds, methods):
    """Only the completed document run supplies the sampled-row replication targets."""
    with ZipFile(path) as archive:
        root = 'run/' if 'run/result.json' in archive.namelist() else ''
        receipt = json.loads(archive.read(root+'result.json'))
        if (receipt['status'] != 'STUDY_COMPLETE' or receipt['source_type'] != 'real_layer' or receipt['design'] != design
                or receipt['evidence_kind'] != 'FROZEN_BILATERAL_DOCUMENT_STUDY'
                or receipt['reference_archive_sha256'] != sha256_file(grid_archive)
                or receipt['fit_archive_sha256'] != sha256_file(fit_archive)):
            raise ValueError('Expected the document study for these frozen fits and articles')
        refs = {}
        for seed in seeds:
            manifest_name = f'inputs/seed_{seed}/data_manifest.json' if root else f'inputs_{seed}/data_manifest.json'
            dev = json.loads(archive.read(manifest_name))['partitions']['dev']
            document_ids = [b['document_id'] for b in dev]
            for method in methods:
                name = f'seed_{seed}/{method}/document_result.json'
                row = checked_json(archive, root+name, receipt['document_artifacts'][name])
                if (row['source_type'] != 'real_layer' or row['seed'] != seed or row['method'] != method
                        or (root and not row.get('heldout_panel_sha256'))
                        or row.get('heldout_documents', document_ids) != document_ids):
                    raise ValueError(f'Wrong document record: {name}')
                row['heldout_documents'] = document_ids
                refs[seed,method] = row
    return refs


def row_energies(weights, panel, q, specs, *, tile_rows=TILE_ROWS, alternate_rows=None):
    """Sum the projected errors and FP target energy per document across whole output rows."""
    if panel.panel_id != SITE or panel.placement != 'input' or weights.ndim != 2 or tile_rows < 1:
        raise ValueError('Expected a full input-side layer_0.up_proj weight and a positive row tile')
    weights = weights.detach().to(q.device)
    sampled = torch.as_tensor(panel.metadata['weight_row_ids'], device=q.device, dtype=torch.long)
    if (not 0 < len(sampled) < len(weights) or sampled.unique().numel() != len(sampled)
            or sampled.min() < 0 or sampled.max() >= len(weights)
            or not torch.equal(weights.index_select(0,sampled).float(), torch.as_tensor(panel.weight,device=q.device).float())):
        raise ValueError('Sampled output rows differ from the captured layer weights')
    mask = torch.ones(len(weights), dtype=torch.bool, device=q.device)
    mask[sampled] = False
    row_sets = {'sampled':sampled}
    if alternate_rows is not None:
        alternate = torch.as_tensor(alternate_rows, device=q.device)
        if (alternate.dtype != torch.long or alternate.shape != sampled.shape or alternate.unique().numel() != len(sampled)
                or alternate.min() < 0 or alternate.max() >= len(weights) or not mask[alternate].all()
                or 2*len(sampled) >= len(weights)):
            raise ValueError('The two fitted row sets must be disjoint, equally sized and leave unselected rows')
        row_sets['alternate'] = alternate
        mask[alternate] = False
    row_sets['complement'] = mask.nonzero().flatten()
    wspec, aspec = panel_specs(specs, panel)
    if wspec.bits != 4 or aspec.bits != 4:
        raise ValueError('The row diagnostic requires the original INT4 weight and activation quantizers')
    activation = torch.as_tensor(panel.activation, dtype=torch.float64, device=q.device)
    counts = panel.metadata['tokens_per_block']
    if (not counts or len(counts) != len(panel.metadata['block_ids']) or len(counts) != len(panel.metadata['document_ids'])
            or sum(counts) != activation.shape[1] or min(counts) < 1):
        raise ValueError('Sampled token counts disagree with the captured panel')
    rotated_activation = q @ activation
    quantized_activation = torch_quantize(rotated_activation.T, aspec)[0].T
    document_mask = torch.eye(len(counts), dtype=torch.float64, device=q.device).repeat_interleave(
        torch.tensor(counts, device=q.device), dim=1, output_size=activation.shape[1])
    totals = {}
    for label, rows in row_sets.items():
        total = torch.zeros((len(counts), 1+len(PRECISIONS)), device=q.device, dtype=torch.float64)
        for tile in rows.split(tile_rows):
            tile_weight = weights.index_select(0, tile).double()
            target = tile_weight @ activation
            rotated_weight = tile_weight @ q.T
            quantized_weight = torch_quantize(rotated_weight, wspec)[0]
            columns = [target, quantized_weight @ quantized_activation - target,
                       quantized_weight @ rotated_activation - target, rotated_weight @ quantized_activation - target]
            per_token = torch.stack([v.square().sum(dim=0) for v in columns], dim=1)
            total += document_mask @ per_token
        totals[label] = total
    totals['all_rows'] = sum(totals.values())
    blocks = {}
    for name, column in totals.items():
        values = column.cpu().numpy()
        if not np.isfinite(values).all() or (values[:,0] <= 0).any():
            raise ValueError(f'Invalid row diagnostic energies in {name}')
        blocks[name] = [{'block_id':bid,'document_id':doc,'sampled_tokens':count,'target_energy':v[0],
                         **dict(zip(PRECISIONS,v[1:]))}
                        for bid,doc,count,v in zip(panel.metadata['block_ids'],panel.metadata['document_ids'],counts,values.tolist())]
    return {'row_counts':{**{name:len(rows) for name,rows in row_sets.items()}, 'all_rows':len(weights)}, 'blocks':blocks}


def score_seed(seed, panels, panel_sha, weights, rotations, specs, reference, *, progress=None):
    panel = next(p for p in panels if p.panel_id == SITE)
    results = {}
    for method, q in rotations.items():
        report(progress, 'bilateral.row_diagnostic', seed=seed, site=SITE, method=method, rows=len(weights))
        digest = q_digest(q)
        row = None if method == PREFIX else reference[seed,method]
        if row and (digest != row['q_sha256'] or ('heldout_panel_sha256' in row and panel_sha != row['heldout_panel_sha256'])
                    or panel.metadata['document_ids'] != row['heldout_documents']):
            raise ValueError('Captured cohort, panel or rotation differs from the completed document pilot')
        result = row_energies(weights, panel, q, specs)
        if row:
            ref = next(p for p in row['heldout'] if p['panel_id'] == SITE)
            sampled = result['blocks']['sampled']
            if (len(sampled) != len(ref['blocks']) or ref['weight_shape'] != list(panel.weight.shape)
                    or any(any(a[k] != b[k] for k in ('block_id','document_id','sampled_tokens'))
                           or not np.allclose([a['W4A4'],a['target_energy']], [b['squared_error'],b['target_energy']],
                                              rtol=1e-9,atol=1e-13) for a,b in zip(sampled,ref['blocks']))):
                raise ValueError('Sampled-row W4A4 or target energy differs from the archived per-document values')
        results[method] = {**result, 'q_sha256':digest}
    return {'seed':seed,'site':SITE,'sampled_row_ids':panel.metadata['weight_row_ids'],
            'heldout_panel_sha256':panel_sha,'methods':results}


def summarize_rows(by_seed, candidate, design):
    contrasts = {'candidate_vs_full_grid':(candidate,CONTINUED), 'prefix_vs_full_grid':(PREFIX,CONTINUED),
                 'candidate_vs_prefix':(candidate,PREFIX)}
    summary = {label:{rows:{} for rows in ROW_SETS} for label in contrasts}
    for rows in ROW_SETS:
        for precision in PRECISIONS:
            records = []
            for seed, row in by_seed.items():
                for method, measurement in row['methods'].items():
                    blocks = [dict(b, squared_error=b[precision]) for b in measurement['blocks'][rows]]
                    records.append({'seed':seed, 'method':method, 'panels':{SITE:blocks}})
            # Identical document draws for all nine estimates. An undefined precision only affects its own contrast.
            stats = paired_site_gains(records, list(by_seed), {SITE:'input'}, contrasts,
                                      repetitions=design['bootstrap_draws'], rng_seed=design['bootstrap_seed'])
            for label, effect in stats['comparisons'].items():
                summary[label][rows][precision] = effect.get('all', effect)
    return summary
