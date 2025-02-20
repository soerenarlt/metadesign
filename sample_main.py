
import os
from contextlib import nullcontext
import json
from itertools import combinations, product

import numpy as np
import torch
from seq2seq import Seq2SeqConfig, Seq2Seq
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, default=0, help='Task ID for mode selection')
args = parser.parse_args()

task_id = args.task_id

os.environ['PATH'] += ':/sbin'

bias = False # do we use bias inside LayerNorm and Linear layers?
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
#if cuda is not available, use cpu
if not torch.cuda.is_available():
    device = 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

ckpt_path = 'ckpt_main/ckpt_main.pt'
exec(open('ckpt_main/config.py').read()) # overrides from command line or config file
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# for k,v in config.items():
#     print(f"{k} = {v}")
# -----------------------------------------------------------------------------

from data_main.graphdata import *
from pytheus.fancy_classes import Graph, State
import traceback

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
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
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

# print(f'ddp_rank = {ddp_rank}')
# print(f'ddp_local_rank = {ddp_local_rank}')

# # # TOP-K
for i in range(10000):

    #generate the dicke states 
    strings_list = []
    states_list = []
    max_n = 5

    modes = ['ghz', 'w', 'dicke', 'dicke2d_half2', 'dicke2d_2vsrest2', 'dicke2d_3vsrest2', 'ghz/w', 'w/w', 'ghz/ghz', 'ghz3d/ghz3d', 'bellN', 'spin1/2', 'majumdar_ghosh', 'dyck', 'dyck_246', 'aklt', 'motzkin', 'motzkin_small']
    mode = modes[task_id%len(modes)]
    print(f'mode = {mode}')

    #random permutation of inds
    # perm = np.random.permutation(4)
    # perm = np.concatenate([perm, np.arange(4,12)])
    # print(perm)
    for ii, numvert in zip(range(3,8), iter([4,6,8,10,12])):
        print(f'generating state for {numvert} vertices')

        if mode == 'ghz':    
            # #GHZ
            base_ket = [0]*numvert
            kets = []
            weights = None
            for ghz_mode in range(2):
                new_ket = base_ket.copy()
                new_ket = [ghz_mode]*(numvert)
                kets.append(new_ket)
            weights = None

        
        if mode == 'w':
            # #W MODE
            base_ket = [0]*numvert
            kets = []
            weights = None
            for w_pos in range(numvert):
                new_ket = base_ket.copy()
                new_ket[w_pos] = 1
                kets.append(new_ket)
            weights = None


        if mode == 'dicke':
            # #DICKE MODE (1,1,1), (2,1,1), (3,1,1)
            inds = list(combinations(range(ii), 2))
            print(2*len(inds))
            base_ket = [0]*numvert
            kets = []
            for ind in inds:
                new_ket = base_ket.copy()
                new_ket[ind[0]] = 1
                new_ket[ind[1]] = 2
                # new_ket=''.join(map(str, new_ket))
                kets.append(new_ket)

                new_ket = base_ket.copy()
                new_ket[ind[1]] = 1
                new_ket[ind[0]] = 2
                # new_ket=''.join(map(str, new_ket))
                kets.append(new_ket)
            weights = None

        # TOO MANY KETS (70 for 8 vertices)
        # if mode == 'dicke2d_half':
        #     # all combinations of half 0 half 1
        #     base_ket = [0]*numvert
        #     kets = []
        #     for ind in combinations(range(numvert), numvert//2):
        #         new_ket = base_ket.copy()
        #         for i in ind:
        #             new_ket[i] = 1
        #         kets.append(new_ket)

        if mode == 'dicke2d_half2':
            # all combinations of half 0 half 1
            base_ket = [0]*numvert
            kets = []
            for ind in combinations(range(numvert-2), numvert//2 - 1):
                new_ket = base_ket.copy()
                for i in ind:
                    new_ket[i] = 2
                kets.append(new_ket)
            weights = None

        if mode == 'dicke2d_2vsrest':
            # all combinations of two 1s and the rest 0
            base_ket = [0]*numvert
            kets = []
            for ind in combinations(range(numvert), 2):
                new_ket = base_ket.copy()
                for i in ind:
                    new_ket[i] = 1
                kets.append(new_ket)
            weights = None

        if mode == 'dicke2d_2vsrest2':
            # all combinations of two 1s and the rest 0
            base_ket = [0]*numvert
            kets = []
            for ind in combinations(range(ii), 2):
                new_ket = base_ket.copy()
                for i in ind:
                    new_ket[i] = 2
                kets.append(new_ket)
            weights = None

        # TOO MANY KETS
        # if mode == 'dicke2d_3vsrest':
        #     # all combinations of three 1s and the rest 0
        #     base_ket = [0]*numvert
        #     kets = []
        #     for ind in combinations(range(numvert), 3):
        #         new_ket = base_ket.copy()
        #         for i in ind:
        #             new_ket[i] = 1
        #         kets.append(new_ket)
                
        if mode == 'dicke2d_3vsrest2':
            # all combinations of three 1s and the rest 0
            base_ket = [0]*numvert
            kets = []
            for ind in combinations(range(ii), 3):
                new_ket = base_ket.copy()
                for i in ind:
                    new_ket[i] = 2
                kets.append(new_ket)
            weights = None


        if mode == 'ghz/w':
            # #GHZ/W MODE
            base_ket = [0]*numvert
            kets = []
            for w_pos in range(numvert//2):
                for ghz_mode in range(2):
                    new_ket = base_ket.copy()
                    new_ket[w_pos+(numvert//2)] = 1
                    new_ket[:numvert//2] = [ghz_mode]*(numvert//2)
                    # new_ket=''.join(map(str, new_ket))
                    kets.append(new_ket)
                # kets.append(new_ket)
            weights = None

        if mode == 'w/w':
            # #W/W MODE
            base_ket = [0]*numvert
            kets = []
            for w_pos1 in range(numvert//2):
                for w_pos2 in range(numvert//2):
                    new_ket = base_ket.copy()
                    new_ket[w_pos1] = 1
                    new_ket[w_pos2+(numvert//2)] = 1
                    kets.append(new_ket)
            weights = None

        if mode == 'ghz/ghz':
            # #GHZ/GHZ MODE
            base_ket = [0]*numvert
            kets = []
            for ghz_mode1 in range(2):
                for ghz_mode2 in range(2):
                    new_ket = base_ket.copy()
                    new_ket[:numvert//2] = [ghz_mode1]*(numvert//2)
                    new_ket[numvert//2:] = [ghz_mode2]*(numvert//2)
                    kets.append(new_ket)
            weights = None

        if mode == 'ghz3d/ghz3d':
            base_ket = [0]*numvert
            kets = []
            for ghz_mode1 in range(3):
                for ghz_mode2 in range(3):
                    new_ket = base_ket.copy()
                    new_ket[:numvert//2] = [ghz_mode1]*(numvert//2)
                    new_ket[numvert//2:] = [ghz_mode2]*(numvert//2)
                    kets.append(new_ket)
            weights = None

        if mode == 'bellN':
            bell_terms = [[0,0],[1,1]]
            kets = []
            for comb in product(range(2), repeat=numvert//2):
                ket = []
                for c in comb:
                    ket += bell_terms[c]
                kets.append(ket)
            weights = None
        

        if mode == 'spin1/2':
        # spin 1/2 states
            base_ket = [0]*numvert
            kets = []
            #binaries from 0 to 2**numvert
            for jj in range(2**ii):
                new_ket = base_ket.copy()
                new_ket[:ii] = list(map(int, list(bin(jj)[2:].zfill(ii))))
                #check if [1, 1] is sublist of new_ket
                if [1, 1] in [new_ket[i:i+2] for i in range(len(new_ket)-1)]:
                    continue
                else:
                    kets.append(new_ket)
            print(len(kets))
            weights = None

        if mode == 'majumdar_ghosh':
            A = [0]*2
            #Aup
            A[1] = np.matrix([[0, 1, 0], [0, 0, -1], [0, 0 , 0]])
            #Ado
            A[0] = np.matrix([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

            A_names = {0:'Au',1:'Ad'}

            # L should be even
            L = numvert
            # D should be 2
            D = 2
            base = [list(range(D))]*L
            sigmas = list(product(*base))
            kets = []
            weights = []
            for sigma in sigmas:
                mat = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
                for part in sigma:
                    mat = np.matmul(mat,A[part])
                trace = round(np.trace(mat),3)
                if trace != 0:
                    #print("*".join([A_names[ind] for ind in sigma]))
                    # print(round(np.trace(mat),3))
                    # print(sigma)
                    # kets.append(''.join([str(el) for el in sigma]))
                    #pad sigma to have length numvert
                    sigma = list(sigma)
                    sigma += [0]*(numvert - len(sigma))
                    kets.append(sigma)
                    weights.append(trace)

        if mode == 'dyck':
            #dyck words mode
            #all possible dyck words of length numvert
            # dyck words mode
            kets = []
            def generate_dyck_words(word, open_count, close_count):
                if open_count == 0 and close_count == 0:
                    kets.append(word)
                    return
                if open_count > 0:
                    generate_dyck_words(word + [1], open_count - 1, close_count)
                if close_count > open_count:
                    generate_dyck_words(word + [2], open_count, close_count - 1)

            generate_dyck_words([], numvert // 2, numvert // 2)
            weights = None

        if mode == 'dyck_246':
            #dyck words mode
            #all possible dyck words of length numvert
            # dyck words mode
            kets = []
            def generate_dyck_words(word, open_count, close_count):
                if open_count == 0 and close_count == 0:
                    kets.append(word+[0,0])
                    return
                if open_count > 0:
                    generate_dyck_words(word + [1], open_count - 1, close_count)
                if close_count > open_count:
                    generate_dyck_words(word + [2], open_count, close_count - 1)

            generate_dyck_words([], numvert // 2 - 1, numvert // 2 - 1)
            weights = None

        if mode == 'aklt':

            A = [0]*3
            #A+
            A[0] = np.matrix([[0, 1/np.sqrt(2)], [0, 0]])
            #A0
            A[1] = np.matrix([[-1/2, 0], [0, 1/2]])
            #A-
            A[2] = np.matrix([[0, 0], [-1/np.sqrt(2), 0]])

            A_names = {0:'A+',1:'A0', 2:'A-'}

            L = ii - 1
            D = 3
            base = [list(range(D))]*L
            sigmas = list(product(*base))
            kets = []
            weights = []
            for sigma in sigmas:
                mat = np.matrix([[1,0],[0,1]])
                for part in sigma:
                    mat = np.matmul(mat,A[part])
                trace = round(np.trace(mat),12)
                if trace != 0:
                    #print("*".join([A_names[ind] for ind in sigma]))
                    # print(round(np.trace(mat),12))
                    # print(sigma)
                    # kets.append(''.join([str(el) for el in sigma]))
                    #pad sigma to have length numvert
                    sigma = list(sigma)
                    sigma += [0]*(numvert - len(sigma))

                    kets.append(sigma)
                    weights.append(int(trace*(2**(L-1)))) #MULTIPLY BY L TO GET int COEFFICIENTS

        if mode == 'motzkin':
            motzkin_symbols = ['(', ')', '-']

            def is_motzkin_word(word):
                depth = 0
                for symbol in word:
                    if motzkin_symbols[symbol] == '(':
                        depth += 1
                    elif motzkin_symbols[symbol] == ')':
                        depth -= 1
                        if depth < 0:
                            return False
                return depth == 0

            def generate_motzkin_words(N):
                canditates = list(itertools.product([0,1,2], repeat=N))
                words = []
                for candidate in canditates:
                    if is_motzkin_word(candidate):
                        words.append(candidate)
                print(len(words))
                return words

            motzkin_words = generate_motzkin_words(ii)
            base_ket = [0]*numvert
            kets = []
            for motzkin_word in motzkin_words:
                new_ket = base_ket.copy()
                new_ket[:ii] = motzkin_word
                kets.append(new_ket)
            weights = None

        if mode == 'motzkin_small':
            motzkin_symbols = ['(', ')', '-']

            def is_motzkin_word(word):
                depth = 0
                for symbol in word:
                    if motzkin_symbols[symbol] == '(':
                        depth += 1
                    elif motzkin_symbols[symbol] == ')':
                        depth -= 1
                        if depth < 0:
                            return False
                return depth == 0

            def generate_motzkin_words(N):
                canditates = list(itertools.product([0,1,2], repeat=N))
                words = []
                for candidate in canditates:
                    if is_motzkin_word(candidate):
                        words.append(candidate)
                print(len(words))
                return words

            motzkin_words = generate_motzkin_words(ii)
            base_ket = [0]*numvert
            kets = []
            for motzkin_word in motzkin_words:
                new_ket = base_ket.copy()
                new_ket[:ii] = motzkin_word
                kets.append(new_ket)
            weights = None



        #random permutation of kets
        if ii%2 == 0:
            np.random.shuffle(kets)
            ket_sorting = 'shuffled'
        else:
            kets.sort()
            ket_sorting = 'sorted'
        # print(kets)
        if weights is not None:
            state = State({tuple([(pos, dim) for pos, dim in enumerate(ket)]):weights[ii] for ii, ket in enumerate(kets)})
        else:
            state = State({tuple([(pos, dim) for pos, dim in enumerate(ket)]):1 for ket in kets}, normalize=False)
        states_list.append(state)

        # print(state)
        if numvert <= 8:
            statestring = build_state_string(state)
            simple_statestring = simplify_state_string(statestring)
            print(statestring)
            # print(simple_statestring)
            strings_list.append(statestring)
        
        state.normalize()

        time.sleep(1)
            
        # positions = 'abcdefgh'
        # modes = 'xyz'

        # statestring = ''
        # for ket in kets:
        #     statestring += '+1['
        #     for ii, val in enumerate(ket):
        #         statestring += positions[ii]+modes[val]
        #     statestring += ']'

    bl_string = '|'.join(strings_list)
    bl = tokenize_string(bl_string, src_token_dict)
    #pad with 0 to make it 640
    bl = np.pad(bl, (0, src_len - len(bl)), 'constant', constant_values=src_token_dict['<PAD>'])
    bl = torch.from_numpy(bl.astype(np.int64)).unsqueeze(0).to(device)
    topp = 0.5
    temp = 0.2
    print(f'temp = {temp}')
    # print(f'topk = {topk}')
    print(f'topp = {topp}')
    # print(f'ket_sorting = {ket_sorting}')
    pred = model.generate(bl, start_token_id=1, end_token_id=2, top_p=topp, temperature=temp)
    pred = detokenize_indices(pred.cpu().numpy().tolist()[0], src_token_dict)[5:-5]

    print(f'### Prediction {i} ###')
    print('Code generated by the model')
    print(pred)
    try:
        fidelities = np.zeros(max_n)
        for N in range(max_n):
            print(f'N = {N}')
            #gg_pred is pytheus Graph, has useful methods for computing perfect matchings / fidelities etc. 
            gg_pred = generate_graph(pred,N)
            gg_pred.state.normalize()
            fidelity = (gg_pred.state@states_list[N])**2
            print(f'fidelity = {fidelity}')
            fidelities[N] = fidelity
        #if all fidelities are >0.99 then we have a perfect match
        if np.all(fidelities > 0.99):
            print('PERFECT MATCH!!!')

        #print boolean array of fidelities
        print(fidelities > 0.99)
    except Exception as e:
        print(e)
        traceback.print_exc()
        print('failed to generate valid states')
        print(pred)
        continue
    print('------------------')


