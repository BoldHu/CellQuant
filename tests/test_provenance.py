from pathlib import Path
import pytest
from cellquant.runtime import atomic_json,sha256_file,build_manifest
from cellquant.migration import export_pilot,import_pilot
from cellquant.runner import execution_fingerprint


def test_fixture_cannot_be_exported_as_real_g1(tmp_path):
    source=tmp_path/'source';source.mkdir()
    atomic_json(source/'result.json',{'source_type':'unit_test_fixture','raw_results':['fixture']})
    atomic_json(source/'config.json',{})
    with pytest.raises(ValueError,match='measured real-layer'):export_pilot(source,tmp_path/'export')


def test_migration_checks_bytes_before_interpreting_records(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir();record=bundle/'record.json'
    atomic_json(record,{'source_type':'unit_test_fixture'})
    atomic_json(bundle/'manifest.json',build_manifest(bundle,['record.json']))
    record.write_bytes(b'changed')
    with pytest.raises(ValueError,match='integrity'):import_pilot(bundle,tmp_path/'imported')


def test_execution_fingerprint_checks_nested_token_cache(tmp_path,monkeypatch):
    import cellquant.runner as runner
    monkeypatch.setattr(runner,'environment_report',lambda:{k:{} for k in ['python','platform','machine','packages','cuda']})
    (tmp_path/'src').mkdir();(tmp_path/'src/a.py').write_text('x=1',encoding='utf-8')
    tokens=tmp_path/'token_ids.npz';tokens.write_bytes(b'unit-test-fixture')
    data=atomic_json(tmp_path/'data.json',{'source_type':'unit_test_fixture','tokens_path':tokens.name,'tokens_sha256':sha256_file(tokens)})
    bundle=atomic_json(tmp_path/'bundle.json',{'kind':'token_manifest_bundle','models':{'fixture':{'11':{'path':str(data),'sha256':sha256_file(data)}}}})
    protocol=atomic_json(tmp_path/'protocol.json',{'source_type':'unit_test_fixture'})
    provenance=atomic_json(tmp_path/'provenance.json',{'models':{'fixture':{'model_revision':'a'*40,'tokenizer_revision':'a'*40}},
        'data_manifest_path':str(bundle),'data_manifest_sha256':sha256_file(bundle),
        'protocol_manifest_path':str(protocol),'protocol_sha256':sha256_file(protocol),'dtype':'float32'})
    initial=execution_fingerprint(tmp_path,provenance)
    assert initial['status']=='VERIFIED'
    generated=tmp_path/'src/fixture.egg-info';generated.mkdir()
    (generated/'requires.txt').write_text('generated package metadata',encoding='utf-8')
    assert execution_fingerprint(tmp_path,provenance)['sha256']==initial['sha256']
    tokens.write_bytes(b'changed')
    check=execution_fingerprint(tmp_path,provenance)
    assert check['status']=='INCOMPLETE' and any('Token cache' in m for m in check['missing'])
