"""Audited adapter to Meta SpinQuant at one immutable commit.

The original checkout is read-only. A separate hashed overlay contains minimal
compatibility patches. GPTQ, activation/KV quantizers, R3/R4 and the full CE/SGDG
rotation baseline are upstream implementations, not alternate quantizers.
"""
from __future__ import annotations

from pathlib import Path
import contextlib
import difflib
import importlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

from .runtime import atomic_json, sha256_file, config_hash, read_json


UPSTREAM_COMMIT='8f47aa3f00e8662caf1a484153920a07e5281c3a'
UPSTREAM_URL='https://github.com/facebookresearch/SpinQuant'


def _git(source,*args):
    return subprocess.check_output(['git','-c',f'safe.directory={Path(source).resolve().as_posix()}',
                                    '-C',str(source),*args],text=True,encoding='utf-8').strip()


def audit_upstream(source):
    source=Path(source).resolve()
    commit=_git(source,'rev-parse','HEAD')
    if commit!=UPSTREAM_COMMIT:raise ValueError(f"SpinQuant revision mismatch: {commit}")
    dirty=_git(source,'status','--porcelain','--untracked-files=no')
    if dirty:raise ValueError('Upstream checkout has modified tracked files; use a separate overlay')
    paths=_git(source,'ls-files').splitlines()
    files={p:sha256_file(source/p) for p in paths if (source/p).is_file()}
    return {'url':UPSTREAM_URL,'commit':commit,'files':files,'tree_sha256':config_hash(files)}


def _replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Upstream patch context differs; re-audit required')
    return text.replace(old,new,1)


def prepare_overlay(source,output,*,fp32_port=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output==source or output.is_relative_to(source):raise ValueError('Overlay must be outside original checkout')
    upstream=audit_upstream(source)
    manifest_path=output/'cellquant_overlay.json'
    if manifest_path.exists():
        manifest=read_json(manifest_path)
        if manifest.get('patch_recipe_version')!=2:raise ValueError('Overlay recipe changed; choose a new output directory')
        if manifest['upstream']['tree_sha256']!=upstream['tree_sha256'] or manifest['fp32_port']!=fp32_port:
            raise ValueError('Overlay identity changed; choose a new output directory')
        for p,h in manifest['files'].items():
            if sha256_file(output/p)!=h:raise ValueError(f'Overlay file changed: {output/p}')
        return manifest
    if output.exists() and any(output.iterdir()):raise ValueError('Refusing to overwrite a nonempty unverified overlay')
    output.mkdir(parents=True,exist_ok=True)
    patches=[]
    for p in upstream['files']:
        src=source/p;dst=output/p;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    edits={}
    p='utils/fuse_norm_utils.py';old=(source/p).read_text(encoding='utf-8')
    block='    # Embedding fusion\n    for W in [model.model.embed_tokens]:\n        W_ = W.weight.data.double()\n        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)\n'
    new=_replace_once(old,block,'    # CellQuant: RMSNorm does not permit embedding row centering.\n')
    # Identical fusion precision for all our reproduced methods.
    edits[p]=(old,new.replace('.double()','.float()').replace('dtype=torch.float64','dtype=torch.float32'))
    p='utils/utils.py';old=(source/p).read_text(encoding='utf-8')
    edits[p]=(old,_replace_once(old,'from fast_hadamard_transform import hadamard_transform',
                                    'from cellquant.hadamard import hadamard_transform'))
    p='eval_utils/gptq_utils.py';old=(source/p).read_text(encoding='utf-8')
    new=_replace_once(old,'    class Catcher(nn.Module):','    class CapturedInput(Exception):\n        """Expected early exit only; genuine model errors propagate."""\n\n    class Catcher(nn.Module):')
    new=_replace_once(new,'            raise ValueError\n','            raise CapturedInput\n')
    new=_replace_once(new,'        except ValueError:\n            pass','        except CapturedInput:\n            continue')
    new=_replace_once(new,'(args.nsamples, 2048, model.config.hidden_size)','(args.nsamples, model.seqlen, model.config.hidden_size)')
    new=_replace_once(new,'        layer = layers[i].to(dev)\n        full =',
        '        layer = layers[i].to(dev)\n'
        '        if getattr(args, "resume_loader", None) is not None and args.resume_loader(i, layer):\n'
        '            for j in range(args.nsamples):\n'
        '                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]\n'
        '            layers[i] = layer.cpu()\n'
        '            inps, outs = outs, inps\n'
        '            continue\n'
        '        full =')
    new=_replace_once(new,'        layers[i] = layer.cpu()\n        del layer\n        del gptq',
        '        if getattr(args, "checkpoint_callback", None) is not None:\n'
        '            args.checkpoint_callback(i, layer)\n'
        '        layers[i] = layer.cpu()\n        del layer\n        del gptq')
    edits[p]=(old,new)
    if fp32_port:
        for p in ['train_utils/quant_linear.py','train_utils/modeling_llama_quant.py']:
            old=(source/p).read_text(encoding='utf-8')
            # Only explicit rotation casts, no changes to CE, R1/R2 or SGDG.
            edits[p]=(old,old.replace('.to(torch.float64)','.to(torch.float32)'))
    patch_dir=Path(__file__).resolve().parents[2]/'patches'
    patch_dir.mkdir(exist_ok=True)
    for p,(old,new) in edits.items():
        (output/p).write_text(new,encoding='utf-8',newline='\n')
        patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+p,tofile='b/'+p))
        name=('fp32_' if fp32_port and p.startswith('train_utils/') else '')+p.replace('/','__')+'.patch'
        (patch_dir/name).write_text(patch,encoding='utf-8',newline='\n')
        patches.append({'path':name,'sha256':sha256_file(patch_dir/name),'upstream_file':p})
    files={p:sha256_file(output/p) for p in upstream['files']}
    result={'patch_recipe_version':2,'upstream':upstream,'fp32_port':fp32_port,'patches':patches,'files':files,
            'scope':'our FP32 implementation' if fp32_port else 'upstream rotation FP64; identical FP32 norm fusion',
            'overlay_sha256':config_hash(files)}
    atomic_json(manifest_path,result)
    atomic_json(patch_dir/('manifest_fp32.json' if fp32_port else 'manifest.json'),
                {'upstream_commit':UPSTREAM_COMMIT,'patches':patches})
    return result


def activate_overlay(path):
    path=Path(path).resolve();manifest=read_json(path/'cellquant_overlay.json')
    if manifest['upstream']['commit']!=UPSTREAM_COMMIT:raise ValueError('Unapproved upstream commit')
    for p,h in manifest['files'].items():
        if sha256_file(path/p)!=h:raise ValueError(f'Overlay hash mismatch: {path/p}')
    if importlib.metadata.version('transformers')!='4.44.2':
        raise RuntimeError('Audited SpinQuant model path requires transformers==4.44.2 in an isolated environment')
    for name in ['utils','eval_utils','train_utils']:
        module=sys.modules.get(name)
        if module is not None:
            origin=getattr(module,'__file__',None)
            paths=list(getattr(module,'__path__',[]))
            if (origin and not Path(origin).resolve().is_relative_to(path)) or any(not Path(p).resolve().is_relative_to(path) for p in paths):
                raise RuntimeError('Upstream module collision; launch a new subprocess for each overlay')
    if str(path) not in sys.path:sys.path.insert(0,str(path))
    return manifest


def atomic_torch_save(payload,path):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.checkpoint-',suffix='.pt',dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload,tmp);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def export_spinquant_rotation(q,r2_per_layer,path,*,num_layers=None):
    import torch
    q=torch.as_tensor(q).detach().double()
    if q.ndim!=2 or q.shape[0]!=q.shape[1] or not torch.isfinite(q).all():raise ValueError('Finite square Q required')
    if num_layers is None:num_layers=len(r2_per_layer)
    if num_layers<1 or len(r2_per_layer)!=num_layers:raise ValueError('Every layer R2 is required')
    residual=float(torch.linalg.norm(q.T@q-torch.eye(len(q),dtype=q.dtype,device=q.device))/math.sqrt(len(q)))
    if residual>1e-5:raise ValueError('Nonorthogonal Q')
    state={'R1':q.T.contiguous().float().cpu()};r2_residual=[]
    for layer,r in enumerate(r2_per_layer):
        r=torch.as_tensor(r).detach().double()
        if r.ndim!=2 or r.shape[0]!=r.shape[1] or not torch.isfinite(r).all():raise ValueError('Invalid R2')
        error=float(torch.linalg.norm(r.T@r-torch.eye(len(r),dtype=r.dtype,device=r.device))/math.sqrt(len(r)))
        if error>1e-5:raise ValueError('Nonorthogonal R2')
        r2_residual.append(error);state[f'model.layers.{layer}.self_attn.R2']=r.float().contiguous().cpu()
    atomic_torch_save(state,path)
    return {'sha256':sha256_file(path),'keys':list(state),'orientation':'R1=Q.T',
            'q_orthogonality_residual':residual,'r2_orthogonality_residuals':r2_residual}


def fixed_r2(config,seed,*,device='cpu'):
    import torch
    from .solver import hadamard_rotation
    h=config.hidden_size//config.num_attention_heads
    if str(device).startswith('cuda'):
        from .tensor_solver import hadamard_rotation as tensor_hadamard
        return [tensor_hadamard(h,seed+i,device).T.contiguous().float() for i in range(config.num_hidden_layers)]
    # Upstream R2 is in row-vector orientation; save this exact same tensor for all controls.
    return [torch.from_numpy(hadamard_rotation(h,seed+i).T.copy()).float() for i in range(config.num_hidden_layers)]


def fuse_norms(model):
    from utils.fuse_norm_utils import fuse_layer_norms
    if getattr(model.config,'model_type',None)!='llama':raise ValueError('Only audited LLaMA RMSNorm graph supported')
    if getattr(model,'_cellquant_norm_fused',False):raise ValueError('Norms already fused')
    if model.config.tie_word_embeddings:
        import torch
        model.lm_head.weight=torch.nn.Parameter(model.lm_head.weight.detach().clone(),requires_grad=False)
        model.config.tie_word_embeddings=False
    fuse_layer_norms(model)
    model._cellquant_norm_fused=True


def rotate_model(model,checkpoint,*,device='cuda',compute_dtype='float32',chunk_rows=1024,global_only=False):
    """Chunked offline global R1, per-head R2, upstream down R4; zero online R1."""
    import torch
    from utils import hadamard_utils
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    dtype=getattr(torch,compute_dtype);r1=state['R1'].to(device=device,dtype=dtype)
    expected={'R1'}|{f'model.layers.{i}.self_attn.R2' for i in range(len(model.model.layers))}
    if set(state)!=expected:raise ValueError('Checkpoint requires R1 and every R2 key, with no extras')
    if r1.shape!=(model.config.hidden_size,)*2:raise ValueError('Wrong R1 shape')
    identity=torch.equal(r1,torch.eye(len(r1),device=r1.device,dtype=r1.dtype))
    with torch.no_grad():
        def right(module,rotation):
            weight=module.weight
            for start in range(0,weight.shape[0],chunk_rows):
                part=weight[start:start+chunk_rows].to(device=device,dtype=dtype)@rotation
                weight[start:start+chunk_rows].copy_(part.to(weight))
        def left(module,rotation):
            weight=module.weight
            for start in range(0,weight.shape[1],chunk_rows):
                part=rotation@weight[:,start:start+chunk_rows].to(device=device,dtype=dtype)
                weight[:,start:start+chunk_rows].copy_(part.to(weight))
            if module.bias is not None:module.bias.copy_((rotation@module.bias.to(device=device,dtype=dtype)).to(module.bias))
        if not identity:right(model.model.embed_tokens,r1);right(model.lm_head,r1)
        for idx,layer in enumerate(model.model.layers):
            if not identity:
                for module in [layer.self_attn.q_proj,layer.self_attn.k_proj,layer.self_attn.v_proj,layer.mlp.up_proj,layer.mlp.gate_proj]:right(module,r1)
                left(layer.self_attn.o_proj,r1.T);left(layer.mlp.down_proj,r1.T)
            if global_only:continue
            r2=state[f'model.layers.{idx}.self_attn.R2'].to(device=device,dtype=dtype);h=r2.shape[0]
            if h!=model.config.hidden_size//model.config.num_attention_heads:raise ValueError('Wrong R2 head dimension')
            for module,output in [(layer.self_attn.v_proj,True),(layer.self_attn.o_proj,False)]:
                w=module.weight.detach().to(device=device,dtype=dtype)
                if output:w=w.T
                shape=w.shape;w=(w.reshape(-1,h)@r2).reshape(shape)
                if output:w=w.T
                module.weight.copy_(w.to(module.weight))
            w=layer.mlp.down_proj.weight
            for start in range(0,w.shape[0],chunk_rows):
                part=w[start:start+chunk_rows].to(device=device,dtype=dtype)
                had,K=hadamard_utils.get_hadK(part.shape[-1])
                w[start:start+chunk_rows].copy_(hadamard_utils.matmul_hadU_cuda(part,had,K).to(w))
    return {'extra_online_r1_ops':0,'compute_dtype':compute_dtype,'checkpoint_sha256':sha256_file(checkpoint),'identity_r1_skipped':identity}


def install_online_graph(model,quantization,*,quantized=False):
    """Use upstream ActQuantWrapper and post-RoPE R3; supports all-FP verification."""
    from utils import quant_utils,hadamard_utils,utils
    from eval_utils import rotation_utils
    if getattr(model,'_cellquant_online_installed',False):raise ValueError('Online graph already installed')
    quant_utils.add_actquant(model)
    qlayers=quant_utils.find_qlayers(model,layers=[quant_utils.ActQuantWrapper])
    head_dim=model.config.hidden_size//model.config.num_attention_heads
    effective={}
    for name,layer in qlayers.items():
        group=quantization.get('a_group_size',-1)
        if 'down_proj' in name:
            had,K=hadamard_utils.get_hadK(model.config.intermediate_size)
            layer.online_full_had=True;layer.had_K=had;layer.K=K;layer.fp32_had=True
            if group>0:group=utils.llama_down_proj_groupsize(model,group)
        if 'o_proj' in name:group=head_dim
        bits=quantization.get('a_bits',4) if quantized and 'lm_head' not in name else 16
        layer.quantizer.configure(bits=bits,groupsize=group,sym=quantization.get('symmetric_a',True),clip_ratio=quantization.get('a_clip_ratio',1.))
        if 'v_proj' in name:
            layer.out_quantizer.configure(bits=quantization.get('v_bits',4) if quantized else 16,groupsize=head_dim,
                                          sym=quantization.get('symmetric_v',True),clip_ratio=quantization.get('v_clip_ratio',1.))
        effective[name]={'a_bits':bits,'a_group_size':group,'v_group_size':head_dim if 'v_proj' in name else None}
    for layer in model.model.layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(layer.self_attn,'apply_rotary_pos_emb',config=model.config,
            k_bits=quantization.get('k_bits',4) if quantized else 16,k_groupsize=quantization.get('k_group_size',head_dim),
            k_sym=quantization.get('symmetric_k',True),k_clip_ratio=quantization.get('k_clip_ratio',1.))
    model._cellquant_online_installed=True
    return effective


def configure_quantization(model,quantization):
    from utils import quant_utils,utils
    qlayers=quant_utils.find_qlayers(model,layers=[quant_utils.ActQuantWrapper])
    head_dim=model.config.hidden_size//model.config.num_attention_heads
    effective={}
    for name,layer in qlayers.items():
        group=quantization.get('a_group_size',-1)
        if 'o_proj' in name:group=head_dim
        if 'down_proj' in name and group>0:group=utils.llama_down_proj_groupsize(model,group)
        bits=16 if 'lm_head' in name else quantization['a_bits']
        layer.quantizer.configure(bits=bits,groupsize=group,sym=quantization['symmetric_a'],clip_ratio=quantization.get('a_clip_ratio',1.))
        if 'v_proj' in name:layer.out_quantizer.configure(bits=quantization['v_bits'],groupsize=head_dim,sym=quantization['symmetric_v'],clip_ratio=quantization.get('v_clip_ratio',1.))
        effective[name]={'a_bits':bits,'a_group_size':group,'v_group_size':head_dim if 'v_proj' in name else None}
    for layer in model.model.layers:
        wrapper=layer.self_attn.apply_rotary_pos_emb_qk_rotation_wrapper
        wrapper.k_bits=quantization['k_bits'];wrapper.k_sym=quantization['symmetric_k']
        wrapper.k_groupsize=quantization['k_group_size'];wrapper.k_clip_ratio=quantization.get('k_clip_ratio',1.)
        wrapper.k_quantizer.configure(bits=wrapper.k_bits,groupsize=-1,sym=wrapper.k_sym,clip_ratio=wrapper.k_clip_ratio)
    return effective


def gptq_quantize(model,blocks,quantization,*,device='cuda',checkpoint_dir=None,signature=None,on_layer_checkpoint=None):
    import torch
    from eval_utils.gptq_utils import gptq_fwrd
    if not torch.cuda.is_available() or not str(device).startswith('cuda'):raise RuntimeError('Audited upstream GPTQ needs CUDA')
    if not blocks:raise ValueError('Nonempty calibration blocks required')
    lengths={int(b.shape[-1]) for b in blocks}
    if len(lengths)!=1:raise ValueError('GPTQ calibration must use one fixed context length')
    model.seqlen=lengths.pop()
    args=SimpleNamespace(nsamples=len(blocks),w_bits=quantization['w_bits'],w_asym=not quantization['symmetric_w'],
        w_clip=quantization.get('final_weight_mse_clip',True),w_groupsize=quantization['w_group_size'],
        int8_down_proj=False,percdamp=quantization.get('gptq_percdamp',.01),act_order=quantization.get('gptq_act_order',False),export_to_et=False)
    if checkpoint_dir is not None:
        if not signature:raise ValueError('GPTQ resume requires full execution/config/token/rotation signature')
        directory=Path(checkpoint_dir);directory.mkdir(parents=True,exist_ok=True)
        def load_layer(index,layer):
            meta=directory/f'layer_{index}.json';path=directory/f'layer_{index}.pt'
            if not meta.exists():return False
            record=read_json(meta)
            if record['signature']!=signature or not path.is_file() or sha256_file(path)!=record['sha256']:
                raise ValueError(f'GPTQ checkpoint provenance/integrity mismatch: {path}')
            state=torch.load(path,map_location='cpu',weights_only=True)
            parameters=dict(layer.named_parameters())
            if state.keys()!=parameters.keys():raise ValueError('GPTQ layer parameter schema mismatch')
            with torch.no_grad():
                for name,p in parameters.items():p.copy_(state[name])
            return True
        def save_layer(index,layer):
            path=directory/f'layer_{index}.pt'
            atomic_torch_save({k:v.detach().cpu() for k,v in layer.named_parameters()},path)
            atomic_json(directory/f'layer_{index}.json',{'signature':signature,'sha256':sha256_file(path),'layer':index})
            if on_layer_checkpoint is not None:on_layer_checkpoint(index)
        args.resume_loader=load_layer;args.checkpoint_callback=save_layer
    old_cache=model.config.use_cache
    first_layer=model.model.layers[0]
    rotary=getattr(model.model,'rotary_emb',None)
    rotary_device=next(rotary.buffers()).device if rotary is not None else None
    if rotary is not None:rotary.to(device)
    try:return gptq_fwrd(model,[(b,) for b in blocks],device,args)
    finally:
        model.config.use_cache=old_cache
        model.model.layers[0]=first_layer
        if rotary is not None:rotary.to(rotary_device)


def compare_logits(reference,actual):
    import torch
    r,a=reference.double(),actual.double()
    return {'relative_rms':float(torch.linalg.norm(a-r)/torch.linalg.norm(r).clamp_min(1e-30)),
            'max_absolute':float((a-r).abs().max())}


def panel_quant_specs(model,panels,quantization):
    """Mirror the audited online graph's location-specific activation groups."""
    from .quantization import QuantSpec
    from utils import utils
    head_dim=model.config.hidden_size//model.config.num_attention_heads
    result={}
    for panel in panels:
        group=quantization['a_group_size']
        if panel.panel_id.endswith('o_proj'):group=head_dim
        if panel.panel_id.endswith('down_proj') and group>0:group=utils.llama_down_proj_groupsize(model,group)
        result[panel.panel_id]=(QuantSpec(quantization['w_bits'],quantization['w_group_size']),QuantSpec(quantization['a_bits'],group))
    return result


def measure_rtn_transfer(model,blocks,quantization,path,*,metadata,device='cuda'):
    """Measured RTN dev result; restore original weights and FP quantizer state."""
    import torch
    from eval_utils.gptq_utils import rtn_fwrd
    from .evaluation import evaluate_nll
    weights={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
    args=SimpleNamespace(w_bits=quantization['w_bits'],w_asym=not quantization['symmetric_w'],
        w_clip=quantization.get('final_weight_mse_clip',True),w_groupsize=quantization['w_group_size'],int8_down_proj=False,export_to_et=False)
    try:
        rtn_fwrd(model,device,args);configure_quantization(model,quantization)
        return evaluate_nll(model,blocks,path,metadata=metadata,device=device)
    finally:
        model.cpu()
        with torch.no_grad():
            for name,p in model.named_parameters():p.copy_(weights[name])
        fp={**quantization,**{k:16 for k in ['a_bits','k_bits','v_bits']}}
        configure_quantization(model,fp)
