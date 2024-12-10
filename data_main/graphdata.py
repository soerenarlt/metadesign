import fancy_classes as fc
import random
import h5py
import os
import numpy as np
import json
import time
from termcolor import colored
import collections

# from html_plot import plot_graph
from fancy_classes import Graph

sign_dict = {-1:'-', 1:'+'}
sign_dict_inv = {'-':-1, '+':1}

positions = 'abcdefghijkl'
position_dict = {i:pos for i, pos in enumerate(positions)}
modes = 'xyz'
mode_dict = {i:mode for i, mode in enumerate(modes)}


def build_state_string(state, weight_sorted=False):
    # write state as string
    kets = state.kets
    if weight_sorted:
        kets_pos = [ket for ket in kets if state[ket] > 0]
        kets_neg = [ket for ket in kets if state[ket] < 0]
        kets = kets_pos + kets_neg

    str_state = ''
    for key in kets:
        weight = state[key]
        if weight != 0:
            ket = [position_dict[el[0]]+mode_dict[el[1]] for el in key]
            ket = '['+''.join(ket)+']'
            if weight > 0:
                str_state += '+'
            str_state += str(weight) + str(ket)
    return str_state

def simplify_state_string(state_str):
    #remove all characters in positions string
    for pos in positions:
        state_str = state_str.replace(pos,'')
    return state_str

def build_graph_string(edge_list):
    if WEIGHTED:
        if WEIGHT_SORTED:
            edge_list_pos = [edge for edge in edge_list if edge[4] == 1]
            edge_list_neg = [edge for edge in edge_list if edge[4] == -1]
            edge_list = edge_list_pos + edge_list_neg
        edge_list = ['('+position_dict[int(edge[0])]+mode_dict[int(edge[2])]+position_dict[int(edge[1])]+mode_dict[int(edge[3])]+sign_dict[int(edge[4])]+')' for edge in edge_list]
    else:
        edge_list = ['('+position_dict[int(edge[0])]+mode_dict[int(edge[2])]+position_dict[int(edge[1])]+mode_dict[int(edge[3])]+')' for edge in edge_list]
    graph_str = ''.join(edge_list)
    return graph_str

def edge_list_from_graph_string(graph_str, weighted=True):
    edge_list = []
    for i in range(len(graph_str)):
        if graph_str[i] == '(':
            vert1 = positions.index(graph_str[i+1])
            col1 = modes.index(graph_str[i+2])
            vert2 = positions.index(graph_str[i+3])
            col2 = modes.index(graph_str[i+4])
            if weighted:
                sign = sign_dict_inv[graph_str[i+5]]
            else:
                sign = 1
            edge_list.append(tuple([vert1, vert2, col1, col2, sign]))
    return edge_list

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
            #raise exception
            raise Exception('unknown token found')
    #eos token
    indices.append(token_dict['<EOS>'])
    return np.array(indices,dtype='int16')

def detokenize_indices(indices, token_dict):
    # Create a reverse mapping from indices to tokens
    reverse_dict = {index: token for token, index in token_dict.items()}
    
    #remove padding tokens
    indices = [index for index in indices if index != token_dict['<PAD>']]

    # Convert the list of indices to the corresponding tokens
    output_str = ''.join(reverse_dict.get(index, '') for index in indices)
    
    return output_str

def generate_graph(pred_str,N):

    def e(pos1,pos2,col1,col2,sign):
        edge_list.append(tuple([pos1,pos2,col1,col2,sign]))

    edge_list = []
    exec(pred_str)
    print(edge_list)
    # plot_graph(edge_list, 'pred'+str(N)+'.html')
    gg_pred = Graph(edge_list, catch_zero_weights = False)
    gg_pred.getState()
    print(f'graph generates: {build_state_string(gg_pred.state)}')
    return gg_pred


def print_diff(src, tgt, pred, token_dict):

    def e(pos1,pos2,col1,col2,sign):
        edge_list.append(tuple([pos1,pos2,col1,col2,sign]))

    src_str = detokenize_indices(src, token_dict)
    pred_str = detokenize_indices(pred, token_dict)
    if tgt == []:
        tgt_str = ''
        print('no target')
    else:
        tgt_str = detokenize_indices(tgt, token_dict)

    print(src_str)
    print(pred_str)
    print(tgt_str)

    NUMS = [0,1,2]
    for N in NUMS:
        gg_pred = generate_graph(pred_str[5:-5],N)

        if not tgt_str:
            # print(gg_pred.state)
            pass
        else:
            gg_pred.state.normalize()
            edge_list = []
            exec(tgt_str[:-5])
            print(edge_list)
            # plot_graph(edge_list, 'tgt'+str(N)+'.html')
            gg_tgt = Graph(edge_list, catch_zero_weights = False)
            gg_tgt.getState()
            print(build_state_string(gg_tgt.state))
            gg_tgt.state.normalize()
            fidelity = (gg_pred.state@gg_tgt.state)**2
            print('fidelity:', fidelity)

if __name__ == '__main__':
    #import token dict from json
    token_dict = json.load(open('tok.json'))

    src = [1, 40, 4, 37, 21, 30, 31, 24, 38, 40, 4, 37, 29, 22, 31, 24, 38, 39, 40, 5, 37, 21, 22, 31, 32, 33, 26, 38, 41, 5, 37, 21, 30, 23, 24, 25, 34, 38, 40, 4, 37, 21, 30, 23, 32, 33, 34, 38, 40, 4, 37, 21, 22, 31, 32, 33, 34, 38, 41, 5, 37, 21, 30, 31, 24, 25, 26, 38, 40, 4, 37, 21, 30, 31, 32, 33, 26, 38, 39, 41, 4, 37, 21, 22, 31, 32, 25, 26, 35, 36, 38, 41, 5, 37, 21, 22, 31, 32, 25, 26, 35, 28, 38, 40, 4, 37, 21, 22, 31, 32, 33, 26, 35, 36, 38, 41, 4, 37, 21, 22, 31, 24, 33, 26, 35, 36, 38, 41, 4, 37, 21, 22, 31, 24, 33, 26, 35, 28, 38, 41, 4, 37, 21, 22, 31, 32, 33, 26, 35, 28, 38, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tgt = [46, 40, 6, 40, 5, 42, 49, 48, 40, 3, 40, 6, 42, 49, 48, 4, 48, 5, 48, 4, 47, 44, 46, 40, 6, 40, 5, 42, 49, 48, 40, 4, 40, 4, 42, 49, 48, 4, 48, 5, 48, 4, 47, 44, 46, 40, 3, 40, 3, 42, 49, 48, 40, 5, 40, 3, 42, 49, 48, 4, 48, 5, 48, 4, 47, 44, 46, 40, 5, 40, 5, 42, 49, 48, 40, 4, 40, 3, 42, 49, 48, 5, 48, 4, 48, 4, 47, 44, 43, 44, 45, 46, 40, 3, 40, 3, 42, 49, 40, 6, 42, 50, 48, 40, 6, 40, 5, 42, 49, 40, 3, 42, 50, 48, 4, 48, 5, 48, 4, 47, 44, 45, 46, 40, 4, 40, 3, 42, 49, 40, 4, 42, 50, 48, 40, 5, 40, 3, 42, 49, 40, 6, 42, 50, 48, 5, 48, 4, 48, 41, 4, 47, 44, 45, 46, 40, 3, 40, 3, 42, 49, 40, 3, 42, 50, 48, 40, 4, 40, 4, 42, 49, 40, 4, 42, 50, 48, 4, 48, 5, 48, 4, 47, 44, 45, 46, 40, 3, 40, 5, 42, 49, 40, 3, 42, 50, 48, 40, 5, 40, 5, 42, 49, 40, 4, 42, 50, 48, 4, 48, 5, 48, 4, 47, 44, 45, 46, 40, 5, 40, 5, 42, 49, 40, 3, 42, 50, 48, 40, 3, 40, 5, 42, 49, 40, 6, 42, 50, 48, 5, 48, 4, 48, 41, 4, 47, 44, 45, 46, 40, 6, 40, 3, 42, 49, 40, 5, 42, 50, 48, 40, 3, 40, 5, 42, 49, 40, 3, 42, 50, 48, 5, 48, 5, 48, 4, 47, 44, 45, 46, 40, 6, 40, 4, 42, 49, 40, 5, 42, 50, 48, 40, 5, 40, 4, 42, 49, 40, 3, 42, 50, 48, 4, 48, 4, 48, 4, 47, 44, 45, 46, 40, 6, 40, 3, 42, 49, 40, 4, 42, 50, 48, 40, 4, 40, 6, 42, 49, 40, 3, 42, 50, 48, 4, 48, 4, 48, 4, 47, 44, 45, 46, 40, 5, 40, 4, 42, 49, 40, 3, 42, 50, 48, 40, 4, 40, 6, 42, 49, 40, 3, 42, 50, 48, 5, 48, 5, 48, 41, 4, 47, 44, 45, 46, 40, 4, 40, 3, 42, 49, 40, 4, 42, 50, 48, 40, 3, 40, 3, 42, 49, 40, 3, 42, 50, 48, 5, 48, 4, 48, 41, 4, 47, 44, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pred =[1, 46, 40, 4, 40, 3, 42, 49, 48, 40, 6, 40, 5, 42, 49, 48, 4, 48, 4, 48, 41, 4, 47, 44, 46, 40, 6, 40, 5, 42, 49, 48, 40, 3, 40, 5, 42, 49, 48, 5, 48, 4, 48, 41, 4, 47, 44, 46, 40, 3, 40, 3, 42, 49, 48, 40, 4, 40, 4, 42, 49, 48, 4, 48, 5, 48, 41, 4, 47, 44, 46, 40, 4, 40, 6, 42, 49, 48, 40, 5, 40, 4, 42, 49, 48, 4, 48, 4, 48, 4, 47, 44, 46, 40, 3, 40, 4, 42, 49, 48, 40, 4, 40, 5, 42, 49, 48, 5, 48, 4, 48, 41, 4, 47, 44, 46, 40, 5, 40, 5, 42, 49, 48, 40, 3, 40, 4, 42, 49, 48, 5, 48, 5, 48, 41, 4, 47, 44, 46, 40, 5, 40, 5, 42, 49, 48, 40, 4, 40, 5, 42, 49, 48, 4, 48, 4, 48, 41, 4, 47, 44, 46, 40, 6, 40, 5, 42, 49, 48, 40, 3, 40, 3, 42, 49, 48, 4, 48, 4, 48, 4, 47, 44, 46, 40, 5, 40, 3, 42, 49, 48, 40, 6, 40, 3, 42, 49, 48, 5, 48, 4, 48, 41, 4, 47, 44, 43, 44, 45, 46, 40, 3, 40, 5, 42, 49, 40, 6, 42, 50, 48, 40, 3, 40, 4, 42, 49, 40, 4, 42, 50, 48, 5, 48, 5, 48, 41, 4, 47, 44, 45, 46, 40, 5, 40, 4, 42, 49, 40, 6, 42, 50, 48, 40, 4, 40, 3, 42, 49, 40, 5, 42, 50, 48, 5, 48, 4, 48, 4, 47, 44, 45, 46, 40, 3, 40, 3, 42, 49, 40, 6, 42, 50, 48, 40, 5, 40, 5, 42, 49, 40, 3, 42, 50, 48, 4, 48, 5, 48, 41, 4, 47, 44, 45, 46, 40, 5, 40, 4, 42, 49, 40, 5, 42, 50, 48, 40, 5, 40, 5, 42, 49, 40, 4, 42, 50, 48, 4, 48, 4, 48, 4, 47, 44, 45, 46, 40, 4, 40, 3, 42, 49, 40, 5, 42, 50, 48, 40, 6, 40, 3, 42, 49, 40, 4, 42, 50, 48, 4, 48, 5, 48, 41, 4, 47, 44, 45, 46, 40, 6, 40, 4, 42, 49, 40, 5, 42, 50, 48, 40, 3, 40, 5, 42, 49, 40, 3, 42, 50, 48, 5, 48, 4, 48, 4, 47, 44, 45, 46, 40, 4, 40, 5, 42, 49, 40, 4, 42, 50, 48, 40, 4, 40, 3, 42, 49, 40, 5, 42, 50, 48, 4, 48, 4, 48, 4, 47, 44, 45, 46, 40, 3, 40, 6, 42, 49, 40, 4, 42, 50, 48, 40, 6, 40, 4, 42, 49, 40, 3, 42, 50, 48, 5, 48, 5, 48, 41, 4, 47, 44, 2]

    # print_diff(src, tgt, pred, token_dict)
    print(detokenize_indices(src, token_dict))
    print(detokenize_indices(pred, token_dict))
    print(detokenize_indices(tgt, token_dict))

    print_diff(src, tgt, pred, token_dict)
          
    