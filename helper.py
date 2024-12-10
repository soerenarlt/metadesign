import numpy as np

def decode(indices, token_dict):
    # Create a reverse mapping from indices to tokens
    reverse_dict = {index: token for token, index in token_dict.items()}
    
    #remove padding tokens
    indices = [index for index in indices if index != token_dict['<PAD>']]

    # Convert the list of indices to the corresponding tokens
    output_str = ''.join(reverse_dict.get(index, '') for index in indices)
    
    return output_str


def encode(input_str, token_dict):
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