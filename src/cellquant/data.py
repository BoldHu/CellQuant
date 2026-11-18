"""Immutable HF identities, disjoint document sampling and token manifests."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from .runtime import atomic_json,read_json,sha256_file,config_hash
from .solver import atomic_npz


def immutable_revision(value):
    return isinstance(value,str) and re.fullmatch(r'[a-fA-F0-9]{40}|[a-fA-F0-9]{64}',value) is not None


def resolve_revision(repo_id,revision=None,*,repo_type='model',token=None):
    from huggingface_hub import HfApi
    info=HfApi().repo_info(repo_id,revision=revision or 'main',repo_type=repo_type,token=token)
    if not immutable_revision(info.sha):raise ValueError('HF did not return an immutable revision')
    return info.sha


def token_hash(tokens):
    return hashlib.sha256(np.ascontiguousarray(tokens,dtype=np.int64).tobytes()).hexdigest()


def wikitext_documents(rows):
    """WikiText article headings, not correlated paragraph rows, define units."""
    chunks=[];doc=0
    for row in rows:
        text=row['text'] if isinstance(row,dict) else str(row)
        stripped=text.strip()
        top=bool(re.match(r'^= [^=].*[^=] =$' ,stripped))
        if top and chunks:
            yield str(doc),'\n'.join(chunks);chunks=[];doc+=1
        if stripped:chunks.append(text)
    if chunks:yield str(doc),'\n'.join(chunks)


def tokenize_documents(documents,tokenizer,context_length,*,add_bos=False,max_blocks=None):
    """Never join unrelated documents; drop each document's final short block."""
    result=[]
    if context_length<2:raise ValueError('Context must contain at least two tokens')
    for doc_id,text in documents:
        ids=tokenizer(text,add_special_tokens=False)['input_ids']
        if add_bos:
            if tokenizer.bos_token_id is None:raise ValueError('Tokenizer has no BOS token')
            ids=[tokenizer.bos_token_id]+ids
        doc_hash=hashlib.sha256(text.encode('utf-8')).hexdigest()
        for start in range(0,len(ids)-context_length+1,context_length):
            tokens=np.array(ids[start:start+context_length],dtype=np.int64)
            result.append({'document_id':str(doc_id),'document_sha256':doc_hash,
                           'block_id':f'{doc_id}:{start}','start_token':start,'tokens':tokens,'token_ids_sha256':token_hash(tokens)})
            if max_blocks is not None and len(result)>=max_blocks:return result
    return result


def split_train_blocks(blocks,fit_count,heldout_count,seed):
    """Sample documents first; use distinct documents for fit and validation."""
    grouped={}
    for block in blocks:grouped.setdefault(block['document_id'],[]).append(block)
    if len(grouped)<fit_count+heldout_count:
        raise ValueError(f'Only {len(grouped)} independent eligible documents, need {fit_count+heldout_count}; no token-level substitution')
    rng=np.random.default_rng(seed);ids=rng.permutation(sorted(grouped)).tolist()
    chosen=[grouped[k][int(rng.integers(len(grouped[k])))] for k in ids[:fit_count+heldout_count]]
    return chosen[:fit_count],chosen[fit_count:]


def confirmation_partitions(blocks, previous, count):
    """Keep the original fits and draw globally unused train documents for a one-shot G1."""
    used_ids, used_docs, used_tokens = set(), set(), set()
    for parts in previous.values():
        for b in [*parts['fit'], *parts['dev']]:
            used_ids.add(b['document_id'])
            used_docs.add(b['document_sha256'])
            used_tokens.add(b['token_ids_sha256'])
    grouped = {}
    for b in blocks:
        if (b['document_id'] not in used_ids and b['document_sha256'] not in used_docs
                and b['token_ids_sha256'] not in used_tokens):
            grouped.setdefault(b['document_id'], []).append(b)
    result = {}
    for seed in sorted(previous):
        rng = np.random.default_rng(np.random.SeedSequence([20261118, int(seed)]))
        chosen = []
        for doc in rng.permutation(sorted(grouped)).tolist():
            if doc in used_ids:
                continue
            candidates = [b for b in grouped[doc] if b['document_sha256'] not in used_docs
                          and b['token_ids_sha256'] not in used_tokens]
            if not candidates:
                continue
            selected = candidates[int(rng.integers(len(candidates)))]
            chosen.append(selected)
            used_ids.add(doc)
            used_docs.add(selected['document_sha256'])
            used_tokens.add(selected['token_ids_sha256'])
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError('Insufficient fresh independent documents for G1 confirmation')
        result[seed] = (previous[seed]['fit'], chosen)
    return result


def save_token_manifest(output,partitions,identity,*,source_type='real_model'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if source_type not in {'real_model','unit_test_fixture'}:raise ValueError('Explicit token source required')
    records={};arrays={}
    for name,blocks in partitions.items():
        rows=[]
        for index,b in enumerate(blocks):
            key=f'{name}_{index}';arrays[key]=np.asarray(b['tokens'],dtype=np.int64)
            rows.append({**{k:v for k,v in b.items() if k!='tokens'},'array_key':key})
        records[name]=rows
    path=output/'token_ids.npz';atomic_npz(path,**arrays)
    manifest={'schema_version':'1.0','source_type':source_type,'identity':identity,'partitions':records,
              'tokens_path':path.name,'tokens_sha256':sha256_file(path),'tokenization':{'document_boundaries':'never_concatenate','short_tail':'drop','bos':False}}
    atomic_json(output/'data_manifest.json',manifest)
    return output/'data_manifest.json'


def load_token_manifest(path,*,allow_fixture=False):
    path=Path(path);manifest=read_json(path)
    if manifest.get('source_type')!='real_model' and not allow_fixture:raise ValueError('Real data provenance required')
    target=path.parent/manifest['tokens_path']
    if not target.is_file() or sha256_file(target)!=manifest['tokens_sha256']:raise ValueError('Token cache checksum mismatch')
    result={}
    with np.load(target,allow_pickle=False) as archive:
        for name,records in manifest['partitions'].items():
            result[name]=[]
            for row in records:
                tokens=archive[row['array_key']].copy()
                if token_hash(tokens)!=row['token_ids_sha256']:raise ValueError('Per-block token hash mismatch')
                result[name].append({**row,'tokens':tokens})
    return manifest,result


def verify_disjoint(fit,heldout):
    doc={b['document_id'] for b in fit};other={b['document_id'] for b in heldout}
    hashes={b['document_sha256'] for b in fit};other_hashes={b['document_sha256'] for b in heldout}
    if doc & other or hashes & other_hashes:raise ValueError('Calibration and held-out documents overlap')
    return True
