import numpy as np
import itertools
# from html_plot import plot_graph
from pytheus.fancy_classes import Graph
from valpos_res import val_verts_0, val_verts_1

import os
import time


# layer 0:
#     * verts: (0,1,2,3) + (0,1,2)*N
#     * colors: (x,y,z)
#     * weights: (1,-1)

# layer 1:
#     * verts: (0,1,2,3,4) + (0,1,2)*N + (0,1,2)*ii
#     * colors: (x,y,z)
#     * weights: (1,-1)

ranges = ['N']

LEVEL = [0,1,2]
VERTEX_NUMS = [4, 6, 8]

pos_0_temp_list = []
for N in range(3):
    pos_0_temp_list.append([eval(pos.replace('N', str(N))) for pos in val_verts_0])
    
pos_1_temp_list = []
for N in range(3):
    entry_base = [pos.replace('N', str(N)) for pos in val_verts_1]
    pos1_temp_list_2 = []
    for ii in range(N):
        entry = [eval(pos.replace('ii', str(ii))) for pos in entry_base]
        pos1_temp_list_2.append(entry)
    pos_1_temp_list.append(pos1_temp_list_2)
        


def generate_graph(layer_0, layer_1, N, layer_0_extra = None, layer_1_extra = None):
    pos_0_temp = pos_0_temp_list[N]

    graph = []
    if layer_0_extra is None:
        layer_0_extra = [[]]*len(layer_0)
    if layer_1_extra is None:
        layer_1_extra = [[]]*len(layer_1)

    for line, extra in zip(layer_0, layer_0_extra):
        edge = [pos_0_temp[line[0]], pos_0_temp[line[1]]]
        edge = edge + extra
        graph.append(edge)
    for line, extra in zip(layer_1, layer_1_extra):
        for ii in range(N): #CHECK THIS WHEN CHANGING RANGES
            pos1_temp2 = pos_1_temp_list[N][ii]
            edge =[pos1_temp2[line[0]], pos1_temp2[line[1]]]
            edge = edge + extra
            graph.append(edge)
    return graph   


def check_self_loops(graph):
    for edge in graph:
        if edge[0] == edge[1]:
            return False
    return True

def check_degree_and_vertcount(graph,vertex_count, min_degree=2):
    verts = sum(graph, [])
    verts, degrees = np.unique(verts, return_counts=True)
    if len(verts) != vertex_count:
        return False
    if not all(degrees >= min_degree):
        return False
    return True

def check_pms(graph):
    graph_temp = [tuple(edge+[0,0]) for edge in graph]
    g = Graph(graph_temp)
    pm_edges = []
    for pm in g.perfect_matchings:
        pm_edges += pm
    pm_edges = list(set(pm_edges))
    for edge in g.edges:
        if edge not in pm_edges:
            return False
    return True

if __name__ == '__main__':
    #task id
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='A')
    args = parser.parse_args()
    node_id = args.task_id
    local_id = os.environ.get('SLURM_LOCALID', 0)
    print(f"Local Task ID: {local_id}")
    SLURMID = str(32*int(node_id) + int(local_id))

    save_dir = 'topologies'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    configA = {
        'filename' : 'LONG_DEG2',
        'line_num_bounds_0' : [4, 12],
        'line_num_bounds_1' : [2, 12],
        'MIN_DEGREE' : 2
    }
    configB = {
        'filename' : 'SHORT_DEG2',
        'line_num_bounds_0' : [4, 8],
        'line_num_bounds_1' : [2, 6],
        'MIN_DEGREE' : 2
    }
    configC = {
        'filename' : 'LONG_DEG1',
        'line_num_bounds_0' : [4, 12],
        'line_num_bounds_1' : [2, 12],
        'MIN_DEGREE' : 1
    }
    configD = {
        'filename' : 'SHORT_DEG1',
        'line_num_bounds_0' : [4, 8],
        'line_num_bounds_1' : [2, 6],
        'MIN_DEGREE' : 1
    }

    configs = {
        'A' : configA,
        'B' : configB,
        'C' : configC,
        'D' : configD
    }

    config = configs[args.config]

    failed_at_loop = 0
    failed_at_degree = 0
    failed_at_pms = 0
    tt = time.time()
    for i in range(1):
        # tt = time.time()
        valid = False
        num_tries = 0
        while not valid:
            num_tries += 1
            if num_tries % 100000 == 0:
                print(f'num_tries: {num_tries}')
                # print(f'failed_at_loop: {failed_at_loop}')
                # print(f'failed_at_degree: {failed_at_degree}')
                # print(f'failed_at_pms: {failed_at_pms}')
            num_lines_0 = np.random.randint(*(config['line_num_bounds_0']))
            num_lines_1 = np.random.randint(*(config['line_num_bounds_1']))

            #shape (num_lines_0, 2)
            layer_0 = np.random.randint(len(val_verts_0), size=(num_lines_0, 2))
            layer_1 = np.random.randint(len(val_verts_1), size=(num_lines_1, 2))

            #[28, 79, 85, 83]|[928, 123, 29, 141]
            # layer_0 = [28, 79, 85, 83]
            # layer_1 = [928, 123, 29, 141]

            edge_num = []
            for N, vertex_count in enumerate(VERTEX_NUMS):
                # if N == 1:
                #     print('N:', N)
                # if N == 2:
                #     print('N:', N)
                #     #plot all graphs
                #     for nn, vc in enumerate(VERTEX_NUMS):
                #         gg = generate_graph(layer_0, layer_1, nn)
                #         gg = [sorted(edge)+[0,0] for edge in gg]
                #         plot_graph(gg, f'graph_{nn}.html')
                gg = generate_graph(layer_0, layer_1, N)
                gg = [sorted(edge) for edge in gg]
                # check if self loops present
                if not check_self_loops(gg):
                    valid = False
                    failed_at_loop += 1
                    break
                if not check_degree_and_vertcount(gg,vertex_count, min_degree=config['MIN_DEGREE']):
                    valid = False
                    failed_at_degree += 1
                    break
                if not check_pms(gg):
                    valid = False
                    failed_at_pms += 1
                    break
                valid = True
        print('CODE FOUND')
        #save layer_0, layer_1 to file
        with open(os.path.join(save_dir, f'topology_{config['filename']}.txt'), 'a') as f:
            layer_0 = [list(edge) for edge in layer_0]
            layer_1 = [list(edge) for edge in layer_1]
            f.write(f'{layer_0}|{layer_1}\n')

    print(f'time taken: {time.time()-tt} seconds')


