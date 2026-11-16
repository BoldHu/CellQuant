"""Exercise six pinned task definitions on explicit local unit-test fixtures.

No benchmark dataset is downloaded, and no result is eligible for G1--G4.
"""
from pathlib import Path
import argparse
from unittest.mock import patch
import datasets
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast,LlamaConfig,LlamaForCausalLM
from cellquant.evaluation import evaluate_tasks,TASK_METRICS,HARNESS_COMMIT
from cellquant.runtime import atomic_json,environment_report


def fixture_dataset(path,name=None,**kwargs):
    if path=='allenai/ai2_arc':rows=[{'id':str(i),'question':'A simple question','choices':{'label':['A','B'],'text':['one','two']},'answerKey':'A'} for i in range(2)]
    elif path=='Rowan/hellaswag':rows=[{'ctx_a':'A person walks','ctx_b':'then','activity_label':'Walking','endings':['one','two','three','four'],'label':'0'} for _ in range(2)]
    elif path=='baber/piqa':rows=[{'goal':'Open a box','sol1':'one','sol2':'two','label':0} for _ in range(2)]
    elif path=='allenai/winogrande':rows=[{'qID':str(i),'sentence':'A person chose _ to walk','option1':'one','option2':'two','answer':'1'} for i in range(2)]
    elif path=='EleutherAI/lambada_openai':rows=[{'text':'A person walks one'},{'text':'A person walks two'}]
    else:raise ValueError(f'Unexpected dataset access: {path}')
    ds=datasets.Dataset.from_list(rows)
    return datasets.DatasetDict(train=ds,validation=ds,test=ds)


def run(output):
    import faulthandler
    faulthandler.dump_traceback_later(45)
    torch.manual_seed(11)
    words=['[UNK]','[PAD]','[EOS]','A','person','walks','one','two','three','four','Question',':','Answer']
    inner=Tokenizer(WordLevel({w:i for i,w in enumerate(words)},unk_token='[UNK]'));inner.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=inner,unk_token='[UNK]',pad_token='[PAD]',eos_token='[EOS]',model_max_length=128)
    config=LlamaConfig(hidden_size=16,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
        vocab_size=len(words),max_position_embeddings=128,eos_token_id=2,pad_token_id=1)
    config._attn_implementation='eager';model=LlamaForCausalLM(config).eval()
    with patch('datasets.load_dataset',side_effect=fixture_dataset):
        result=evaluate_tasks(model,tokenizer,output,harness_path='third_party/lm-evaluation-harness',harness_commit=HARNESS_COMMIT,
            task_metric_keys=TASK_METRICS,task_revisions={k:'0'*40 for k in TASK_METRICS},
            metadata={'source_type':'unit_test_fixture','fixture_revision_sentinel':True},device='cpu')
    atomic_json(Path(output)/'smoke_report.json',{'status':'PASS_FIXTURE_INTERFACE_ONLY','source_type':'unit_test_fixture',
        'result':result,'environment':environment_report(),'scope':'Six pinned task adapters/HFLM/raw prediction serialization; no benchmark accuracy evidence'})
    print('PASS_FIXTURE_INTERFACE_ONLY',len(result['task_accuracy']))
    faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='runs/harness_smoke');a=p.parse_args();run(a.output)
