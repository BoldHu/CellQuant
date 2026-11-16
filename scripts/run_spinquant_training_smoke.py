"""Tiny real execution of upstream CE/SGDG R1+R2; no pretrained result."""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from cellquant.runtime import atomic_json,environment_report
from cellquant.spinquant_adapter import activate_overlay,fixed_r2
from cellquant.spinquant_train import train_rotations
from cellquant.solver import hadamard_rotation
from cellquant.data import token_hash


def run(args):
    os.environ['CELLQUANT_FHT_BACKEND']='torch_reference';activate_overlay(args.overlay)
    from transformers import LlamaConfig
    from train_utils.modeling_llama_quant import LlamaForCausalLM
    torch.manual_seed(11)
    cfg=LlamaConfig(hidden_size=16,intermediate_size=64,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,vocab_size=64)
    cfg._attn_implementation='eager';model=LlamaForCausalLM(cfg).float()
    quant={'a_bits':4,'k_bits':4,'v_bits':4,'a_group_size':-1,'k_group_size':8,'v_group_size':8,
           'symmetric_a':True,'symmetric_k':True,'symmetric_v':True}
    blocks=[]
    for i in range(4):
        ids=np.random.default_rng(i).integers(0,64,16);blocks.append({'tokens':ids,'token_ids_sha256':token_hash(ids)})
    checkpoint,report=train_rotations(model,blocks,hadamard_rotation(16,11),fixed_r2(cfg,37),args.output,
        quantization=quant,steps=2,accumulation_steps=2,dtype='float32',gradient_checkpointing=True,
        execution_fingerprint='synthetic_math_tiny_training',save_steps=1,resume=args.resume)
    report.update(source_type='synthetic_math',environment=environment_report(),scope='Tiny random LLaMA CE/SGDG execution only; no full-model quality claim')
    atomic_json(Path(args.output)/'smoke_report.json',report);print(report['status'],report['steps'],report['orthogonality_residuals'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--overlay',default='runs/spinquant_overlay_v2');p.add_argument('--output',default='runs/build_m4_training')
    p.add_argument('--resume');run(p.parse_args())
