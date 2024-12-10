import itertools
import numpy as np
import random
import h5py
import json
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import os

coeff_list=[0]+[1/np.sqrt(x) for x in range(1,17)]+[2/np.sqrt(x) for x in range(5,16)]+[3/np.sqrt(x) for x in range(10,16)]
coeff_str_list = ['0']+['1/√'+str(x) for x in range(1,17)]+['2/√'+str(x) for x in range(5,16)]+['3/√'+str(x) for x in range(10,16)]

def find_coeff_index(value, coeff_list):
    """
    Finds the index of the closest coefficient in coeff_list to the given value.
    """
    # Handle the case where the value is effectively 0
    if np.abs(value) < 1e-15:
        return 0
    # Find the index of the closest value in coeff_list
    index = np.argmin([np.abs(value - coeff) for coeff in coeff_list])
    #assert that coeff_list[index] - value < 1e-15
    if np.abs(coeff_list[index] - value) > 1e-10:
        print(f'np.abs(coeff_list[index] - value): {np.abs(coeff_list[index] - value)}')
        print(f'coeff_list[index]: {coeff_list[index]}')
        print(f'value: {value}')
        print(coeff_list)
    assert np.abs(coeff_list[index] - value) < 1e-10
    return index

def transform_array(statevector_array, coeff_list, string_output=True):
    transformed_array = []
    string_array = []
    for entry in statevector_array:
        # Determine the sign of the real and imaginary parts
        real_sign = +1 if entry.real >= 0 else -1
        imag_sign = +1 if entry.imag >= 0 else -1
        
        # Special case for handling zero imaginary part with +1 sign
        imag_sign = +1 if np.abs(entry.imag) < 1e-15 else imag_sign
        
        # Find the best matching coefficient indices
        real_coeff_index= find_coeff_index(np.abs(entry.real), coeff_list)
        imag_coeff_index = find_coeff_index(np.abs(entry.imag), coeff_list)

        real_coeff_str = coeff_str_list[real_coeff_index]
        imag_coeff_str = coeff_str_list[imag_coeff_index]

        if string_output:
            letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            letters = ['[amp '+x+']' for x in letters]
            real_coeff_index = letters[real_coeff_index]
            imag_coeff_index = letters[imag_coeff_index]
            coeffstr = ''
            if real_coeff_str != '0':
                if real_sign == 1:
                    coeffstr += '+'
                else:
                    coeffstr += '-'
                coeffstr += f'{real_coeff_str}'
            if imag_coeff_str != '0':
                if imag_sign == 1:
                    coeffstr += '+'
                else:
                    coeffstr += '-'
                coeffstr += f'{imag_coeff_str}*i'
            if coeffstr == '':
                coeffstr = '0'
            string_array.append(f'({coeffstr})')
            real_sign = '[sgn +]' if real_sign == 1 else '[sgn -]'
            imag_sign = '[sgn +]' if imag_sign == 1 else '[sgn -]'
            # Append the string representation of the transformed entry
            transformed_array.append(''.join([real_sign, real_coeff_index, imag_sign, imag_coeff_index]))

        else:
            # Append the tuple (real_sign, real_coeff_index, imag_sign, imag_coeff_index)
            transformed_array.append((real_sign, real_coeff_index, imag_sign, imag_coeff_index))
    return transformed_array, string_array


def translate_code_to_qiskit(code):
    global global_code
    global_code=''
    exec(code)
    return global_code


def execute_code(code, curr_NN):
    # print('Executing code...')
    # print(f'code: {code}')
    curr_code=code.replace('NN',str(curr_NN))
    # print(f'curr_code: {curr_code}')
    qiskit_code=translate_code_to_qiskit(curr_code)
    # print(f'qiskit_code: {qiskit_code}')
    #print(f'qiskit_code: {qiskit_code}')
    qc = QuantumCircuit(curr_NN+1)
    # print('Qiskit code:')
    # print(qiskit_code)
    if len(qiskit_code)==0:
        print('No cricuit generated.')
        raise ValueError("No qiskit code.")
        
    try:
        exec(qiskit_code)
    except Exception as e:
        raise ValueError(f"Error in executing qiskit code: {e}")
        

    # Execute the circuit on the state vector simulator
    # result = execute(qc, Aer.get_backend('statevector_simulator')).result()
    # statevector = result.get_statevector(qc)
    qc.remove_final_measurements()  # no measurements allowed
    statevector = Statevector(qc)
    
    # Convert the Statevector object to a numpy array explicitly
    statevector_array = np.asarray(statevector)        

    return statevector_array



def generate_lines(num_lines, val_verts, prob_cnot = 0.3):
    lines = []
    for i in range(num_lines):
        if random.random() < prob_cnot:
            #select two random vertices
            v1 = random.choice(val_verts)
            v2 = random.choice(val_verts)
            lines.append([v1,v2])
        else:
            v = random.choice(val_verts)
            lines.append([v])
    return lines

def pick_single_vert_gate():
    return random.choice(['qX','qZ','qH'])

def build_code(top):
    code = ''
    for i, t in enumerate(top):
        # print(t)
        if isinstance(t[0], str):
            code += f'for ii in range({t[0]}):' + '\n'
            t = t[1]
            loop = True
        else:
            loop = False
        for l in t:
            if loop:
                code += ' '*4
            if len(l) > 1:
                code += f'qCNOT({l[0]},{l[1]})' + '\n'
            else:
                code += f'{pick_single_vert_gate()}({l[0]})' + '\n'

    return code

def qX(qubit):
    global global_code
    global_code += f'qc.x({qubit})\n'
    
def qZ(qubit):
    global global_code
    global_code += f'qc.z({qubit})\n'
    
def qH(qubit):
    global global_code
    global_code += f'qc.h({qubit})\n'
    
def qCNOT(qubit1, qubit2):
    global global_code
    global_code += f'qc.cx({qubit1}, {qubit2})\n'

def StatevectorToKet(statevector,NN, coeffs):
    ket_notation = ""
    for i, amplitude in enumerate(statevector):
        if np.abs(amplitude) > 1e-6:  # Check if the amplitude is non-zero
            binary_state = format(i, '0' + str(NN+1) + 'b')
            #replace 0 with X and 1 with Y
            binary_state = binary_state.replace('0','X').replace('1','Y')
            ket_notation += f"{coeffs[i]}|" + binary_state + ">+"
    # Remove the trailing " + " if it exists
    if ket_notation.endswith("+"):
        ket_notation = ket_notation[:-1]
    return ket_notation

def print_circuit_ascii(code, num_qubits):
    lines = code.split('\n')[:-1]
    for qubit in range(num_qubits):
        # print(f'q{qubit}', end=' ')
        for l in lines:
            gate = l.split('(')
            gate[1] = '[' + gate[1][:-1] + ']'
            # print(gate)
            if qubit in eval(gate[1]):
                if gate[0] == 'qc.x':
                    print('---X---', end='')
                elif gate[0] == 'qc.z':
                    print('---Z---', end='')
                elif gate[0] == 'qc.h':
                    print('---H---', end='')
                elif gate[0] == 'qc.cx':
                    if qubit == eval(gate[1])[0]:
                        print('---o---', end='')
                    else:
                        print('---x---', end='')
            else:
                print('-------', end='')
        print('\n', end='')
        #     if f'({qubit})' in l:
        #         print(l[3:], end=' ')
        # print()
    # print('---------------------')
    # print(lines)



NNvec = [1,2,3]

#LAYER 0
ints = ['-1','','1','2']
scale = ['','+NN','+2*NN']
combs = itertools.product(ints, scale)
combs = [''.join(c) for c in combs]
#remove ''
combs = [c for c in combs if c != '']
combs += ['0']
#if comb starts with +, remove it
combs = [c[1:] if c[0] == '+' else c for c in combs]


combs_filtered = []
for c in combs:
    valid = True
    for NN in NNvec:
        val = eval(c.replace('NN',str(NN)))
        if val >= NN+1 or val < 0:
            valid = False
            break
    if valid:
        combs_filtered.append(c)

val_verts_0 = combs_filtered
print(val_verts_0)        

#LAYER 1

#RANGES
ints = ['','1','2'] 
scale = ['','+NN','+NN']
ranges = itertools.product(ints, scale)
ranges = [''.join(r) for r in ranges]
#remove '' and '1'
ranges = [r for r in ranges if r != '' and r != '1']
#if range starts with +, remove it
ranges = [r[1:] if r[0] == '+' else r for r in ranges]
print(f'Ranges: {ranges}')

ints = ['-1','','1','2']
scale = ['','+NN']
its = ['','+ii','+2*ii']
combs = itertools.product(ints, scale, its)
combs = [''.join(c) for c in combs]
#remove ''
combs = [c for c in combs if c != '']
combs += ['0'] 
#if comb starts with +, remove it
combs = [c[1:] if c[0] == '+' else c for c in combs]
print(f'Combs: {combs}')

val_verts_1 = {}
for r in ranges:
    combs_filtered = []
    for c in combs:
        valid = True
        for NN in NNvec:
            range_val = eval(r.replace('NN',str(NN)))
            for ii in range(range_val):
                val = eval(c.replace('NN',str(NN)).replace('ii',str(ii)))
                if val >= NN+1 or val < 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            combs_filtered.append(c)
    val_verts_1[r] = combs_filtered
# print(val_verts_1)

# print(ranges)

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

def generate_valid_code():
    attempts = 0
    while True:
        attempts += 1
        num_for_loops = random.randint(1,2)
        min_lines, max_lines = 1, 4
        num_lines = np.random.randint(min_lines, max_lines+1, size=[num_for_loops,2]).tolist()
        num_lines = [[np.random.randint(min_lines, max_lines+1)]]+num_lines
        # print(num_lines)

        top = []
        for n in num_lines:
            if len(n) > 1:
                limit = random.choice(list(val_verts_1.keys()))
                lines = generate_lines(n[1], val_verts_1[limit])
                top.append([limit,lines])
            lines = generate_lines(n[0], val_verts_0)
            top.append(lines)

        # print(top)
        # print('---------------------')

        code = build_code(top)

        valid = True
        for NN in NNvec:
            global global_code
            global_code = ''
            exec(code)
            #filter out all lines starting with qc.cx
            lines = global_code.split('\n')
            cnots = [eval(l[5:]) for l in lines if 'qc.cx' in l]
            # print(global_code)
            # print_circuit_ascii(global_code, N+1)
            # print('---------------------')
            if len(lines) > 10:
                valid = False
                break
            for l in cnots:
                if l[0] == l[1]:
                    valid = False
                    break

        if valid:
            # print('##################'*3)
            # print('VALID CODE')
            # print(code)
            # print('##################'*3)
            states = []
            for curr_NN in NNvec:
                statevector = execute_code(code,curr_NN=curr_NN)
                # print(statevector)

                coeffs, coeff_strings = transform_array(statevector, coeff_list)
                # print()
                ket_notation = StatevectorToKet(statevector,curr_NN,coeff_strings)
                # print(ket_notation)
                states.append(ket_notation)

            break
    return code, states

#task id
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, default=0)
args = parser.parse_args()
node_id = args.task_id
local_id = os.environ.get('SLURM_LOCALID', 0)
print(f"Local Task ID: {local_id}")
SLURMID = str(32*int(node_id) + int(local_id))
print(f"Task ID: {SLURMID}")

#load json
src_tok = json.load(open('src_tok_v3.json'))
tgt_tok = json.load(open('tgt_tok_v3.json'))

SEP_TOKEN = '<SEP>'
START_TOKEN = '<SOS>'
END_TOKEN = '<EOS>'
PAD_TOKEN = '<PAD>'

MAX_TOKS = 100

NUM_SAMPLES = 100000
val_samples = 0
data_buffer = []
while val_samples < NUM_SAMPLES:
    code, states = generate_valid_code()
    states = SEP_TOKEN.join(states)
    # print(code)
    # print(states)

    tok_tgt = tokenize_string(code,tgt_tok)
    tok_src = tokenize_string(states,src_tok)
    if len(tok_tgt) > MAX_TOKS or len(tok_src) > MAX_TOKS:
        # print('skipping')
        continue
    #pad the sequences
    tok_tgt = np.pad(tok_tgt, (0, MAX_TOKS-len(tok_tgt)), 'constant', constant_values=(tgt_tok[PAD_TOKEN]))
    tok_src = np.pad(tok_src, (0, MAX_TOKS-len(tok_src)), 'constant', constant_values=(src_tok[PAD_TOKEN]))
    sample = {
        'code': tok_tgt,
        'state': tok_src
    }

    data_buffer.append(sample)
    if len(data_buffer) % 1000 == 0:
        maxshape_dict = {
                        'code': (None, MAX_TOKS),
                        'state': (None, MAX_TOKS)
                    }
        print(f'Generated {val_samples} samples.', flush=True)
        #save to hdf5
        with h5py.File(f'data/data_{SLURMID}.hdf5', 'a') as f:
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
                #print new shape
                print(f'{key} shape: {f[key].shape}')

        data_buffer = []
    val_samples += 1

