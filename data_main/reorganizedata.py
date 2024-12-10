import h5py
import os
import json
from graphdata import detokenize_indices, tokenize_string
import math
import numpy as np


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, default=0)
args = parser.parse_args()
node_id = args.task_id
local_id = os.environ.get('SLURM_LOCALID', 0)
print(f"Local Task ID: {local_id}")

#get task id
array_id = os.environ.get('SLURM_ARRAY_TASK_ID', 0)
print(f'Task ID: {array_id}', flush=True)


task_id = 50*int(array_id) + int(local_id)

def e(pos1,pos2,col1,col2,sign):
    edge_list.append(tuple([pos1,pos2,col1,col2,sign]))


#token dict
with open('tok.json', 'r') as f:
    token_dict = json.load(f)

#load all files data/shuffled_data_*.h5
h5_files = [f for f in os.listdir('data') if f.endswith('.h5') and f.startswith('shuffled_data_')]
print(f'Found {len(h5_files)} HDF5 files', flush=True)

ii = task_id
h5_file = h5_files[ii]
print(f'Processing {h5_file}', flush=True)

#find dataset names
for dataset_name in h5py.File(f'data/{h5_file}', 'r').keys():
    print(f'Dataset: {dataset_name}')

with h5py.File(f'data/{h5_file}', 'r') as f:
    statedata = f['state'][:]

    #read out shape
    sequencelen = statedata.shape[1]
    print(f'statedata.shape: {statedata.shape}')

    for kk in range(statedata.shape[0]):
        if kk % 1000 == 0:
            print(f'Processing state {kk}/{statedata.shape[0]}', flush=True)
        X = statedata[kk]
        #detokenize indices
        Xstr = detokenize_indices(X, token_dict)[5:-5]
        Xstrsplit = Xstr.split('|')
        Xstrsplit = [x.split(']') for x in Xstrsplit]
        Xstrsplit = [[x+']' for x in x if x!=''] for x in Xstrsplit]

        X_weighted = []
        for i in range(len(Xstrsplit)):
            X_weighted.append([])
            for j in range(len(Xstrsplit[i])):
                weight, ket = Xstrsplit[i][j].split('[')
                ket = '['+ket
                # print(weight, ket)
                X_weighted[i].append((int(weight), ket))

        new_str = ''
        for i in range(len(X_weighted)):
            weights = []
            for j in range(len(X_weighted[i])):
                weights.append(X_weighted[i][j][0])
            # print(weights)
            #compute gcd of weights
            gcd = weights[0]
            for j in range(1,len(weights)):
                gcd = math.gcd(gcd, weights[j])
            if gcd > 1:
                # print('gcd > 1')
                pass

            #divide all weights by gcd
            for j in range(len(weights)):
                X_weighted[i][j] = (X_weighted[i][j][0]//gcd, X_weighted[i][j][1])

            #sort by second element
            X_weighted[i] = sorted(X_weighted[i], key=lambda x: x[1])

            for j in range(len(X_weighted[i])):
                if X_weighted[i][j][0] > 0:
                    new_str += '+'
                new_str += str(X_weighted[i][j][0])+X_weighted[i][j][1]
            new_str += '|'
        new_str = new_str[:-1]

        tok_state = tokenize_string(new_str, token_dict)
        #padding, concatenate
        tok_state = np.concatenate([tok_state, np.zeros((sequencelen-len(tok_state),), dtype='int8')])
        # print(list(X))
        # print(list(tok_state))

        # print(detokenize_indices(tok_state, token_dict))
        # print(detokenize_indices(X, token_dict))
        statedata[kk] = tok_state

#save to new file
new_h5_file = f'reorganized_{ii}'
#copy file
os.system(f'cp data/{h5_file} data/{new_h5_file}.h5')
#open file
with h5py.File(f'data/{new_h5_file}.h5', 'a') as f:
    #write to file
    f['state'][:] = statedata

print(f'Saved to {new_h5_file}')