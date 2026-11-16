"""Portable G1 evidence bundles; preserve source bytes and derive relocated refs.

Exports Q, token manifests/cache, measured records and G0 references, not base
weights or secrets. Recomputing a gate on import cannot upgrade its measurements.
"""
from pathlib import Path
import argparse
import shutil
from .runtime import atomic_json,read_json,sha256_file,build_manifest,verify_manifest


def export_pilot(run_dir,output):
    run_dir=Path(run_dir).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new bundle directory')
    result=read_json(run_dir/'result.json');config=read_json(run_dir/'config.json')
    if result.get('source_type')!='real_layer' or not result.get('raw_results'):raise ValueError('A measured real-layer pilot is required')
    output.mkdir(parents=True,exist_ok=True);mapping={};records=[]
    def copy(path,folder=None):
        original=str(path)
        path=Path(path).resolve()
        if str(path) in mapping:
            mapping[original]=mapping[str(path)];return mapping[str(path)]
        dest=output/(folder or ('payload/'+sha256_file(path)))/path.name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        mapping[str(path)]=dest.relative_to(output).as_posix();mapping[original]=mapping[str(path)];return mapping[str(path)]
    for filename in result['raw_results']:
        row=read_json(filename)
        for key,digest in [('q_path','q_sha256'),('data_manifest_path','data_manifest_sha256')]:
            if sha256_file(row[key])!=row[digest]:raise ValueError('Changed pilot artifact; export refused')
        records.append(copy(filename));copy(row['q_path'])
        data_path=Path(row['data_manifest_path']);folder='data/'+sha256_file(data_path)
        copy(data_path,folder);data=read_json(data_path);tokens=data_path.parent/data['tokens_path']
        if sha256_file(tokens)!=data['tokens_sha256']:raise ValueError('Token cache checksum mismatch')
        copy(tokens,folder)
    g0_path=Path(config.get('g0_evidence_path','runs/build_m2/G0_evidence.json')).resolve()
    g0=read_json(g0_path)
    for ref in g0.get('artifact_refs',[]):
        if not isinstance(ref,dict) or not ref.get('path'):raise ValueError('G0 artifact reference is not portable')
        if sha256_file(ref['path'])!=ref['sha256']:raise ValueError('G0 artifact integrity mismatch')
        copy(ref['path'])
    g0_relative=copy(g0_path)
    copy(run_dir/'result.json');copy(run_dir/'config.json')
    atomic_json(output/'relocation.json',{'source_run_dir':str(run_dir),'mapping':mapping,'pilot_records':records,'g0_evidence':g0_relative})
    files=[p.relative_to(output) for p in output.rglob('*') if p.is_file()]
    atomic_json(output/'manifest.json',build_manifest(output,files,run_kind='real_layer_evidence_transfer'))
    return {'status':'EXPORTED','bundle':str(output),'files':len(files)}


def import_pilot(bundle,output):
    from .pilot import build_g1_evidence
    bundle=Path(bundle).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new relocation output directory')
    manifest=read_json(bundle/'manifest.json');verified=verify_manifest(bundle,manifest)
    if verified['status']!='PASS':raise ValueError('Bundle integrity check failed: '+str(verified))
    relocation=read_json(bundle/'relocation.json');mapping=relocation['mapping']
    output.mkdir(parents=True,exist_ok=True)
    def relocated(original):
        if original not in mapping:raise ValueError('Unresolved external artifact: '+original)
        path=(bundle/mapping[original]).resolve()
        if not path.is_relative_to(bundle):raise ValueError('Unsafe relocated artifact path')
        return str(path)
    g0_source=bundle/relocation['g0_evidence'];g0=read_json(g0_source)
    for ref in g0['artifact_refs']:ref['path']=relocated(ref['path'])
    g0['provenance']['migration_source_sha256']=sha256_file(g0_source)
    g0_path=atomic_json(output/'G0_evidence.json',g0)
    paths=[]
    for index,relative in enumerate(relocation['pilot_records']):
        source=bundle/relative;row=read_json(source)
        for key in ['q_path','data_manifest_path']:row[key]=relocated(row[key])
        row['migration_source_sha256']=sha256_file(source)
        paths.append(atomic_json(output/f'pilot_{index}.json',row))
    gate=build_g1_evidence(paths,output,g0_evidence_path=g0_path)
    atomic_json(output/'migration.json',{'status':'IMPORTED','source_manifest_sha256':sha256_file(bundle/'manifest.json'),
        'gate_status':gate['status'],'scope':'Same measured Colab G1 evidence, not new AutoDL performance'})
    return gate


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    ex=sub.add_parser('export');ex.add_argument('--run-dir',required=True);ex.add_argument('--output',required=True)
    im=sub.add_parser('import');im.add_argument('--bundle',required=True);im.add_argument('--output',required=True)
    a=p.parse_args();print(export_pilot(a.run_dir,a.output) if a.action=='export' else import_pilot(a.bundle,a.output)['status'])
