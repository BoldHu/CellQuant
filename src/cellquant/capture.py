"""Layer-streamed frozen panels from the actual SpinQuant FP rotation graph."""
from __future__ import annotations
import math
import numpy as np
from .solver import CalibrationPanel
from .data import token_hash
from .runtime import config_hash
from .progress import report


def _tree_to(value,device):
    import torch
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,tuple):return tuple(_tree_to(v,device) for v in value)
    if isinstance(value,list):return [_tree_to(v,device) for v in value]
    if isinstance(value,dict):return {k:_tree_to(v,device) for k,v in value.items()}
    return value


def capture_panels(model,blocks,*,layers,projections=('q_proj','up_proj','down_proj'),
                   tokens_per_panel=128,input_weight_rows=64,seed=11,device='cuda',graph_identity=None,source_type='real_layer',progress=None,resident=False):
    import torch
    if not blocks or tokens_per_panel<len(blocks):raise ValueError('Sample at least one token per independent block')
    if not getattr(model,'_cellquant_online_installed',False):raise ValueError('Capture requires the audited fixed R2/R3/R4 graph')
    if graph_identity is None:raise ValueError('Record the fixed rotation/norm graph identity')
    if source_type not in {'real_layer','unit_test_fixture'}:raise ValueError('Explicit capture source type required')
    if len(set(layers))!=len(layers) or min(layers)<0 or max(layers)>=len(model.model.layers):raise ValueError('Invalid layers')
    rng=np.random.default_rng(seed);hidden=[];kwargs_list=[];handles=[]
    storage=device if resident else 'cpu'
    if resident and (not str(device).startswith('cuda') or any(p.device.type!='cuda' for p in model.parameters())):
        raise RuntimeError('Resident capture requires the complete model on CUDA; no CPU fallback')
    report(progress,'capture.embedding',device=device,blocks=len(blocks))
    class CapturedInput(Exception):pass
    def first_hook(module,args,kwargs):
        hidden.append(args[0].detach().to(storage));kwargs_list.append(_tree_to(kwargs,storage));raise CapturedInput
    if not resident:model.cpu()
    model.eval();model.config.use_cache=False
    initial=[model.model.embed_tokens,model.model.norm]
    if hasattr(model.model,'rotary_emb'):initial.append(model.model.rotary_emb)
    for module in initial:module.to(device)
    hook=model.model.layers[0].register_forward_pre_hook(first_hook,with_kwargs=True)
    try:
        with torch.inference_mode():
            for block in blocks:
                try:model(torch.tensor(block['tokens'][None,:],device=device),use_cache=False)
                except CapturedInput:continue
                raise RuntimeError('First-layer capture did not stop the forward')
    finally:
        hook.remove()
        if not resident:
            for module in initial:module.cpu()
    selection=[]
    for i,block in enumerate(blocks):
        count=tokens_per_panel//len(blocks)+int(i<tokens_per_panel%len(blocks))
        if count>len(block['tokens']):raise ValueError('Requested more tokens than full context')
        selection.append(np.sort(rng.choice(len(block['tokens']),count,replace=False)))
    panels=[]
    with torch.inference_mode():
        for lid,layer in enumerate(model.model.layers):
            report(progress,'capture.layer',device=device,layer=lid,layer_count=max(layers)+1,selected=lid in layers)
            layer.to(device);captured={};weights={};active=[0]
            try:
                if lid in layers:
                    for projection in projections:
                        parent=layer.self_attn if projection in {'q_proj','k_proj','v_proj','o_proj'} else layer.mlp
                        wrapper=getattr(parent,projection)
                        # Hook INSIDE wrapper, after its online R4/quantizer, not raw SwiGLU input.
                        module=wrapper.module;name=f'layer_{lid}.{projection}';captured[name]=[]
                        output=projection in {'down_proj','o_proj'}
                        rows=np.arange(module.weight.shape[0]) if output else np.linspace(0,module.weight.shape[0]-1,min(input_weight_rows,module.weight.shape[0]),dtype=int)
                        weight=module.weight.detach()[torch.tensor(rows,device=device)].float()
                        weights[name]=(weight if resident else weight.cpu().numpy(),rows,output)
                        def hook_input(mod,args,key=name):
                            z=args[0].detach().reshape(-1,args[0].shape[-1])
                            indices=torch.tensor(selection[active[0]],device=z.device)
                            value=z.index_select(0,indices).float()
                            captured[key].append(value if resident else value.cpu().numpy())
                        handles.append(module.register_forward_pre_hook(hook_input))
                for b in range(len(blocks)):
                    active[0]=b
                    out=layer(hidden[b].to(device),**_tree_to(kwargs_list[b],device))
                    hidden[b]=out[0].detach().to(storage)
                for name,pieces in captured.items():
                    w,rows,output=weights[name];x=(torch.cat(pieces,dim=0) if resident else np.concatenate(pieces,axis=0)).T
                    metadata={'layer':lid,'projection':name.split('.')[-1],'source_type':source_type,
                        'weight_row_ids':rows.tolist(),'full_hidden_axis':output,'graph_identity':graph_identity,
                        'block_ids':[b['block_id'] for b in blocks],'document_ids':[b['document_id'] for b in blocks],
                        'tokens_per_block':[len(s) for s in selection],'token_positions':[s.tolist() for s in selection]}
                    panels.append(CalibrationPanel(name,'output' if output else 'input',w,x,
                        config_hash([{'hash':b['token_ids_sha256'],'positions':s.tolist()} for b,s in zip(blocks,selection)]),w.shape[1],1.,metadata))
            finally:
                for handle in handles:handle.remove()
                handles.clear()
                if not resident:layer.cpu()
            if lid>=max(layers):break
    if len(panels)!=len(layers)*len(projections):raise RuntimeError('Incomplete panel capture')
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    report(progress,'capture.complete',panels=len(panels))
    return panels
