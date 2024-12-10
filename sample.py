
import os
from contextlib import nullcontext
import json
from itertools import combinations, product

import numpy as np
import torch
from seq2seq import Seq2SeqConfig, Seq2Seq
import itertools
import sys


#parse keyword command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, default=0)
parser.add_argument('--cp_dir', type=str, default='')
args = parser.parse_args()
cp_dir = args.cp_dir
task_id = args.task_id

from helper import encode, decode
from data_circuits.datagenerator import execute_code, StatevectorToKet, coeff_list, coeff_str_list, transform_array

os.environ['PATH'] += ':/sbin'

bias = False # do we use bias inside LayerNorm and Linear layers?
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
#if cuda is not available, use cpu
if not torch.cuda.is_available():
    device = 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster


print(f'cp_dir = {cp_dir}')
config_path = os.path.join(cp_dir, 'config.py')
ckpt_path = os.path.join(cp_dir, 'ckpt.pt')
print(f'ckpt_path = {ckpt_path}')
print(f'config_path = {config_path}')
# config file parsing (first argument is the python config file to load)
config_str = open(config_path).read()
for line in config_str.split('\n'):
    try:
        exec(line)
    except:
        pass

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
for k,v in config.items():
    print(f"{k} = {v}")
# -----------------------------------------------------------------------------

# torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

src_token_dict = json.load(open(src_token_dict))
tgt_token_dict = json.load(open(tgt_token_dict))
# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, src_len=src_len, tgt_len=tgt_len,
                  bias=bias, src_vocab_size=None, tgt_vocab_size=None, dropout=dropout, tgt_pad_token_id=src_token_dict['<PAD>']) # start with model_args from command line

checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
for k in ['n_layer', 'n_head', 'n_embd', 'src_len', 'tgt_len', 'bias', 'src_vocab_size', 'tgt_vocab_size']:
    model_args[k] = checkpoint_model_args[k]
# create the model
gptconf = Seq2SeqConfig(**model_args)
model = Seq2Seq(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
iter_num = checkpoint['iter_num']
best_val_loss = checkpoint['best_val_loss']
model.to(device)

#get task id
ddp_rank = os.environ.get('RANK', 0)
ddp_local_rank = os.environ.get('LOCAL_RANK', 0)

print(f'ddp_rank = {ddp_rank}')
print(f'ddp_local_rank = {ddp_local_rank}')

bl_string = '(+1/√2)|XX>+(+1/√2)|YY><SEP>(+1/√2)|XXX>+(+1/√2)|YYY><SEP>(+1/√2)|XXXX>+(+1/√2)|YYYY>'
bl = encode(bl_string, src_token_dict)
# print(bl)
print(len(bl))
#pad with 0 to make it 640
bl = np.pad(bl, (0, src_len - len(bl)), 'constant', constant_values=src_token_dict['<PAD>'])
# print(bl)
print(len(bl))
# print(bl)
bl = torch.from_numpy(bl.astype(np.int64)).unsqueeze(0).to(device)
# print(bl)
print('########NEW SAMPLE##########'*3)
topp = 0.5
temp = 0.2
print(f'temp = {temp}')
# print(f'topk = {topk}')
print(f'topp = {topp}')
for i in range(50):
    pred = model.generate(bl, start_token_id=1, end_token_id=2, top_p=topp, temperature=temp)
    pred = decode(pred.cpu().numpy().tolist()[0], tgt_token_dict)[5:-5]
    print(f'### Prediction {i} ###')
    print('\n')
    print(pred)

    for N in [1,2,3,4]:
        statevector = execute_code(pred, N)
        coeffs, coeff_strings = transform_array(statevector, coeff_list)
        kets = StatevectorToKet(statevector,N, coeff_strings)
        print(kets)
    print('\n')

