import numpy as np
import itertools
# from html_plot import plot_graph
from fancy_classes import Graph
from valpos_res import val_verts_0, val_verts_1

import os
import time
import json
import h5py

from generate_topologies import generate_graph, pos_0_temp_list, pos_1_temp_list

sign_dict = {-1:'-', 1:'+'}
sign_dict_inv = {'-':-1, '+':1}
positions = 'abcdefgh'
position_dict = {i:positions[i] for i in range(8)}
modes = 'xyz'
mode_dict = {i:modes[i] for i in range(3)}

def build_state_string(state):
    # write state as string
    str_state = ''
    for key in state.kets:
        weight = state[key]
        if weight != 0:
            ket = [position_dict[el[0]]+mode_dict[el[1]] for el in key]
            ket = '['+''.join(ket)+']'
            if weight > 0:
                str_state += '+'
            str_state += str(weight) + str(ket)
    return str_state

def tokenize_string(input_str, token_dict):
    indices = []
    i = 0
    #sos token
    indices.append(token_dict['<SOS>'])
    while i < len(input_str):
        found = False
        for token, index in token_dict.items():
            if input_str.startswith(token, i):
                indices.append(index)
                i += len(token)
                found = True
                break
        if not found:
            i += 1  # Move to the next character if no matching token is found
            print('unknown token found')
            print(input_str[i-1])
            print(input_str[i-2:i+2])
            #raise exception
            raise Exception('unknown token found')
    #eos token
    indices.append(token_dict['<EOS>'])
    return np.array(indices,dtype='int8')

def detokenize_indices(indices, token_dict):
    # Create a reverse mapping from indices to tokens
    reverse_dict = {index: token for token, index in token_dict.items()}
    
    #remove padding tokens
    indices = [index for index in indices if index != token_dict['<PAD>']]

    # Convert the list of indices to the corresponding tokens
    output_str = ''.join(reverse_dict.get(index, '') for index in indices)
    
    return output_str


if __name__ == '__main__':
    #task id
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=0)
    args = parser.parse_args()
    node_id = args.task_id
    local_id = os.environ.get('SLURM_LOCALID', 0)
    print(f"Local Task ID: {local_id}")
    SLURMID = str(32*int(node_id) + int(local_id))

    token_dict = json.load(open('tok.json'))

    def build_config_dict(ind):
        possible_values = {
            'CODELEN': ['SHORT', 'LONG'],
            'DEGREE': ['DEG1', 'DEG2'],
            'DIMENSION': ['2D', '3D'],
            'EDGEWEIGHT': ['WEIGHTED', 'UNWEIGHTED'],
            'MAX_KETS': ['8-16-32', '6-6-6']
        }
        #all possible combinations
        all_combinations = list(itertools.product(*possible_values.values()))
        print(f'number of combinations: {len(all_combinations)}')
        print(f'chosen index: {ind} of {len(all_combinations)} (possibly applied modulo')
        #get the combination
        combination = all_combinations[ind%len(all_combinations)]
        #create the dictionary
        config_dict = {key:combination[i] for i,key in enumerate(possible_values.keys())}
        return config_dict, len(all_combinations)
    
    config_dict, num_combinations = build_config_dict(int(SLURMID))
    FILE_IND = int(SLURMID)//num_combinations
    # config_dict = {
    #         'CODELEN': 'LONG',
    #         'DEGREE': 'DEG2',
    #         'DIMENSION': '2D',
    #         'EDGEWEIGHT': 'UNWEIGHTED',
    #         'MAX_KETS': '8-16-32'
    #     }
    globals().update(config_dict)
    topology_dict = {('LONG', 'DEG2'): 0, ('SHORT', 'DEG2'): 1, ('LONG', 'DEG1'): 2, ('SHORT', 'DEG1'): 3}
    topology_letter = topology_dict[(CODELEN, DEGREE)]
    print(f'topology letter: {topology_letter}')
    MAX_KETS = tuple([int(val) for val in MAX_KETS.split('-')])

    DIR_NAME = 'data'
    #join the config values to create the directory name
    TOP_FILENAME = 'topologies/'+CODELEN+'_'+DEGREE+'.txt'
    OUT_DIR = DIR_NAME +'/'+'_'.join([str(val) for val in config_dict.values()])
    if FILE_IND == 0 and not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    time.sleep(3)
    FILE_NAME = OUT_DIR+'/'+str(FILE_IND)+'.h5'
    LOG_FILE_PATH = OUT_DIR+'/'+str(FILE_IND)+'.txt'
    print(f'file name: {FILE_NAME}')
    print(f'topology file: {TOP_FILENAME}')
    with h5py.File(f'{FILE_NAME}', 'a') as f:
        #save token dict, config dict and val_verts to the file
        f.attrs['token_dict'] = json.dumps(token_dict)
        f.attrs['config_dict'] = json.dumps(config_dict)
        f.attrs['val_verts_0'] = json.dumps(val_verts_0)
        f.attrs['val_verts_1'] = json.dumps(val_verts_1)

    NUM_SAMPLES = 50_000

    ranges = ['N']
    VERTEX_NUMS = [4, 6, 8]
    if EDGEWEIGHT == 'WEIGHTED':
        WEIGHTS = [-1,1]
    else:
        WEIGHTS = [1]
    REPS = 100

    used_topologies = []
    valid_codes = 0
    data_buffer = []
    tt = time.time()
    max_src = 0
    max_tgt = 0
    tt = time.time()
    while valid_codes < NUM_SAMPLES:
        #read data/code.txt
        with open(TOP_FILENAME, 'r') as file:
            data = file.read()
            for line_ind, line in enumerate(data.split('\n')):
                valid = False
                # print(f'line {line_ind}')
                if DIMENSION == '3D':
                    COLORS = [0,1,2]
                elif DIMENSION == '2D':
                    #choose 2 colors randomly from 0,1,2
                    COLORS = np.random.choice([0,1,2], 2, replace=False)  

                for i in range(REPS):
                    if line == '':
                        continue
                    # print(line)
                    #split at '|'
                    parts = line.split('|')
                    # print(parts)
                    layer_0 = eval(parts[0])
                    layer_1 = eval(parts[1])
                    #shuffle the layers
                    np.random.shuffle(layer_0)
                    np.random.shuffle(layer_1)

                    #shuffle within the layers
                    for layer in [layer_0, layer_1]:
                        for posvals in layer:
                            np.random.shuffle(posvals)

                    #adding random colors and weights
                    layer_0_extra = [[np.random.choice(COLORS), np.random.choice(COLORS), np.random.choice(WEIGHTS)] for _ in range(len(layer_0))]
                    layer_1_extra = [[np.random.choice(COLORS), np.random.choice(COLORS), np.random.choice(WEIGHTS)] for _ in range(len(layer_1))]

                    valid = True
                    for N in range(3):
                        # print(N)
                        graph = generate_graph(layer_0, layer_1, N, layer_0_extra, layer_1_extra)
                        graph = [tuple([int(el) for el in edge]) for edge in graph]
                        try:
                            g = Graph(graph)
                            g.getState()
                        except:
                            valid = False
                            break

                        non_zero_amps = [amp for amp in g.state.amplitudes if amp != 0]
                        if len(non_zero_amps) > MAX_KETS[N] or len(non_zero_amps) == 0:
                            valid = False
                            break

                    if valid:
                        valid_codes += 1
                        # print(f'CODE {valid_codes} FOUND',end='\r')
                        state_str = ''

                        topology_ind = np.array([topology_letter, line_ind], dtype='int8')
                        used_topologies.append(line_ind)
                        line_count = np.array([len(layer_0),len(layer_1)], dtype='int8')

                        num_kets = np.zeros(3, dtype='int8')
                        num_zero_kets = np.zeros(3, dtype='int8')
                        num_pms = np.zeros(3, dtype='int8')
                        degrees_list = np.zeros((3, 8), dtype='int8')
                        for N in range(3):
                            graph = generate_graph(layer_0, layer_1, N, layer_0_extra, layer_1_extra)
                            graph = [tuple([int(el) for el in edge]) for edge in graph]
                            # print(graph)
                            g = Graph(graph)
                            g.getState()
                            num_kets[N] = len([amp for amp in g.state.amplitudes if amp != 0])
                            num_zero_kets[N] = len([amp for amp in g.state.amplitudes if amp == 0])
                            num_pms[N] = len(g.perfect_matchings)

                            verts = sum([list(edge[:2]) for edge in g.edges], [])
                            _, degrees = np.unique(verts, return_counts=True)
                            #sort the degrees
                            degrees = np.sort(degrees)
                            degrees_list[N,:(4+2*N)] = degrees
                        
                            state_str += build_state_string(g.state) + '|'
                            # g = Graph(graph)
                            # plot_graph(graph, f'graph_{N}.html')
                        state_str = state_str[:-1]
                        # print(state_str)

                        code_str = ''
                        for pos, extra in zip(layer_0, layer_0_extra):
                            code_str += f'e({val_verts_0[pos[0]]},{val_verts_0[pos[1]]},{extra[0]},{extra[1]},{extra[2]})\n'
                        code_str += 'for ii in range(N):\n'
                        for pos, extra in zip(layer_1, layer_1_extra):
                            code_str += f'    e({val_verts_1[pos[0]]},{val_verts_1[pos[1]]},{extra[0]},{extra[1]},{extra[2]})\n'

                        # print(code_str)

                        code = tokenize_string(code_str, token_dict)
                        state = tokenize_string(state_str, token_dict)

                        sample = {
                            'code': code,
                            'state': state,
                            'topology_ind': topology_ind,
                            'line_count': line_count,
                            'num_kets': num_kets,
                            'num_zero_kets': num_zero_kets,
                            'num_pms': num_pms,
                            'degrees_list': degrees_list
                        }

                        # print(sample)

                        data_buffer.append(sample)
                        break
                if len(data_buffer) >= 1000:
                    # print(sample)
                    MAX_TOKS = 1024
                    maxshape_dict = {
                        'code': (None, MAX_TOKS),
                        'state': (None, MAX_TOKS),
                        'topology_ind': (None,2),
                        'line_count': (None, 2),
                        'num_kets': (None, 3),
                        'num_zero_kets': (None, 3),
                        'num_pms': (None, 3),
                        'degrees_list': (None, 3, 8)
                    }

                    #write to txt file
                    time_elapsed = time.time() - tt
                    unique_topologies = len(set(used_topologies))
                    with open(LOG_FILE_PATH, 'a') as file:
                        file.write(f'{valid_codes} samples written to file in {time_elapsed} seconds, {round((valid_codes/(time_elapsed/3600))/1000,2)}k samples per hour, {unique_topologies} unique topologies\n')
                    #create the dataset if it does not exist
                    with h5py.File(FILE_NAME, 'a') as f:
                        for key, val in data_buffer[0].items():
                            if key not in f:
                                init_shape = (0, *maxshape_dict[key][1:])
                                print(f'creating dataset {key} with shape {init_shape}')
                                f.create_dataset(key, init_shape, maxshape=maxshape_dict[key], dtype='int8')
                            # print(key, val, val.shape)
                            f[key].resize((f[key].shape[0] + len(data_buffer)), axis = 0)
                            #padding
                            if key in ['code', 'state']:
                                data = np.zeros((len(data_buffer), MAX_TOKS), dtype='int8')
                                for i, sample in enumerate(data_buffer):
                                    data[i,:len(sample[key])] = sample[key]
                            else:
                                data = np.array([sample[key] for sample in data_buffer], dtype='int8')
                            f[key][-len(data_buffer):] = data

                    data_buffer = []
                    if valid_codes >= NUM_SAMPLES:
                        break
    print('DONE')