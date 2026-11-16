"""Full upstream R1+R2 CE/SGDG baseline, with rotation-only checkpoints.

The supplied token manifest fixes the training stream. This is a single-device
protocol reproduction, not a claim of bitwise eight-GPU reproduction. FP32 port
is selected by an independently hashed overlay and must qualify numerically.
"""
from pathlib import Path
import hashlib
import math
from .spinquant_adapter import atomic_torch_save
from .runtime import atomic_json,config_hash,sha256_file,read_json


def train_rotations(model,blocks,q0,r2,output,*,quantization,seed=11,steps=100,
                    accumulation_steps=8,learning_rate=1.5,dtype='bfloat16',
                    train_r2=True,gradient_checkpointing=True,resume=None,execution_fingerprint=None,save_steps=None):
    import torch
    import transformers
    from transformers import Trainer,TrainingArguments,default_data_collator
    from train_utils.main import prepare_model
    from train_utils.optimizer import SGDG
    from utils.process_args import parser_gen
    if not torch.cuda.is_available():raise RuntimeError('Full SpinQuant rotation training requires CUDA')
    if not blocks:raise ValueError('Frozen CE calibration blocks required')
    if dtype=='bfloat16' and not torch.cuda.is_bf16_supported():raise ValueError('BF16 unsupported; choose explicit FP16 protocol and rerun all baselines')
    transformers.set_seed(seed)
    # The pinned parser_gen parses sys.argv itself and returns (args, unknown).
    import sys
    original_argv=sys.argv
    try:
        sys.argv=[original_argv[0]]
        args,_=parser_gen()
    finally:sys.argv=original_argv
    args.w_bits=16;args.a_bits=quantization['a_bits'];args.k_bits=quantization['k_bits'];args.v_bits=quantization['v_bits']
    args.a_groupsize=quantization['a_group_size'];args.k_groupsize=quantization['k_group_size'];args.v_groupsize=quantization['v_group_size']
    args.a_asym=not quantization['symmetric_a'];args.k_asym=not quantization['symmetric_k'];args.v_asym=not quantization['symmetric_v']
    args.a_clip_ratio=quantization.get('a_clip_ratio',1.);args.k_clip_ratio=quantization.get('k_clip_ratio',1.);args.v_clip_ratio=quantization.get('v_clip_ratio',1.)
    args.fp32_had=True;args.seed=seed
    if model.config.tie_word_embeddings:
        model.lm_head.weight=torch.nn.Parameter(model.lm_head.weight.detach().clone());model.config.tie_word_embeddings=False
    prepare_model(args,model)
    for parameter in model.parameters():parameter.requires_grad_(False)
    class RotationParameter(torch.nn.Module):
        def __init__(self,value,learn=True):
            super().__init__();self.weight=torch.nn.Parameter(torch.as_tensor(value,device='cuda',dtype=torch.float32).clone(),requires_grad=learn)
        def forward(self,x,transpose=False):return x@self.weight if transpose else self.weight@x
    model.R1=RotationParameter(torch.as_tensor(q0).T)
    for i,layer in enumerate(model.model.layers):layer.self_attn.R2=RotationParameter(r2[i],train_r2)
    parameters=[model.R1.weight]+([layer.self_attn.R2.weight for layer in model.model.layers] if train_r2 else [])
    model.config.use_cache=False;model.to('cuda');model.seqlen=len(blocks[0]['tokens'])
    optimizer=SGDG(parameters,lr=learning_rate,stiefel=True)
    class TokenDataset(torch.utils.data.Dataset):
        def __len__(self):return len(blocks)
        def __getitem__(self,i):
            ids=torch.as_tensor(blocks[i]['tokens'],dtype=torch.long)
            return {'input_ids':ids,'labels':ids.clone()}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    signature=config_hash({'execution':execution_fingerprint,'seed':seed,'steps':steps,'accumulation':accumulation_steps,
                           'lr':learning_rate,'dtype':dtype,'train_r2':train_r2,'quantization':quantization,
                           'q0':hashlib.sha256(q0.tobytes()).hexdigest(),
                           'r2':{i:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for i,v in enumerate(r2)},
                           'gradient_checkpointing':gradient_checkpointing,
                           'tokens':[b['token_ids_sha256'] for b in blocks]})
    class RotationTrainer(Trainer):
        def _save(self,output_dir=None,state_dict=None):
            target=Path(output_dir or self.args.output_dir)
            state={k:v.detach().cpu() for k,v in self.model.state_dict().items() if k=='R1.weight' or '.self_attn.R2.weight' in k}
            atomic_torch_save(state,target/'rotation_state.pt')
            atomic_json(target/'rotation_signature.json',{'signature':signature})
        def _load_from_checkpoint(self,resume_from_checkpoint,model=None):
            target=Path(resume_from_checkpoint)
            record=read_json(target/'rotation_signature.json')
            if record['signature']!=signature:raise ValueError('SpinQuant resume provenance mismatch')
            if not record.get('files'):raise ValueError('Checkpoint lacks safe RNG/integrity manifest; use a newly written checkpoint')
            for name,digest in record['files'].items():
                if sha256_file(target/name)!=digest:raise ValueError('SpinQuant checkpoint file checksum mismatch: '+name)
            state=torch.load(target/'rotation_state.pt',map_location='cpu',weights_only=True)
            own=self.model.state_dict()
            expected={'R1.weight'}|{f'model.layers.{i}.self_attn.R2.weight' for i in range(len(self.model.model.layers))}
            if set(state)!=expected:raise ValueError('Incomplete R1/R2 resume state')
            with torch.no_grad():
                for name,tensor in state.items():own[name].copy_(tensor)
        def _save_checkpoint(self,model,trial,metrics=None):
            import random
            import numpy as np
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            super()._save_checkpoint(model,trial,metrics)
            target=Path(self._get_output_dir(trial=trial))/f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
            n=np.random.get_state()
            # Explicit safe tensor/primitive representation avoids NumPy pickle
            # globals and remains loadable with torch>=2.6 weights_only=True.
            rng={'python':random.getstate(),'numpy':[n[0],torch.tensor(n[1].astype('int64')),n[2],n[3],n[4]],
                 'cpu':torch.random.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
            atomic_torch_save(rng,target/'rng_state_safe.pt')
            names=['rotation_state.pt','optimizer.pt','scheduler.pt','trainer_state.json','rng_state_safe.pt']
            atomic_json(target/'rotation_signature.json',{'signature':signature,'files':{n:sha256_file(target/n) for n in names}})
        def _load_rng_state(self,checkpoint):
            import random
            import numpy as np
            state=torch.load(Path(checkpoint)/'rng_state_safe.pt',map_location='cpu',weights_only=True)
            random.setstate(state['python']);n=state['numpy']
            np.random.set_state((n[0],n[1].numpy().astype('uint32'),n[2],n[3],n[4]))
            torch.random.set_rng_state(state['cpu']);torch.cuda.set_rng_state_all(state['cuda'])
    training=TrainingArguments(output_dir=str(output),per_device_train_batch_size=1,gradient_accumulation_steps=accumulation_steps,
        learning_rate=learning_rate,weight_decay=0.,lr_scheduler_type='cosine',max_steps=steps,seed=seed,data_seed=seed,
        fp16=dtype=='float16',bf16=dtype=='bfloat16',gradient_checkpointing=gradient_checkpointing,
        save_strategy='steps',save_steps=save_steps or max(1,min(10,steps)),save_total_limit=2,logging_steps=1,report_to=[],
        save_safetensors=False,dataloader_num_workers=0,remove_unused_columns=False)
    trainer=RotationTrainer(model=model,args=training,train_dataset=TokenDataset(),data_collator=default_data_collator,optimizers=(optimizer,None))
    trainer.train(resume_from_checkpoint=resume)
    state={'R1':model.R1.weight.detach().cpu()}
    for i,layer in enumerate(model.model.layers):state[f'model.layers.{i}.self_attn.R2']=layer.self_attn.R2.weight.detach().cpu()
    checkpoint=output/'R.bin';atomic_torch_save(state,checkpoint)
    residuals={key:float(torch.linalg.norm(v.T@v-torch.eye(len(v)))/math.sqrt(len(v))) for key,v in state.items()}
    report={'status':'COMPLETE','method':'spinquant_full' if train_r2 else 'spinquant_r1_control',
            'checkpoint_sha256':sha256_file(checkpoint),'steps':trainer.state.global_step,
            'trainable_parameters':sum(p.numel() for p in parameters),'orthogonality_residuals':residuals,
            'training_quantization':'W16A4KV4','lr_schedule':'cosine','optimizer':'upstream SGDG',
            'signature':signature,'log_history':trainer.state.log_history,'train_r2':train_r2,
            'distinct_available_blocks':len(blocks),'processed_microbatches':steps*accumulation_steps,
            'single_device_gradient_accumulation':accumulation_steps}
    atomic_json(output/'training_report.json',report)
    return checkpoint,report
