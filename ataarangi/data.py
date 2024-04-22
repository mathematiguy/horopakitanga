import json
import torch
import numpy as np
import pandas as pd
from itertools import chain
from torch.utils.data import Dataset
from ataarangi.utils import split_chunks


COLOURS = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'pink']
color_map = dict(zip(COLOURS, range(len(COLOURS))))


class RākauDataset(Dataset):
    def __init__(self, tokens, tokenizer):
        self.tokens = tokens
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        input_ids = self.tokens[idx]

        # Create token type IDs and attention masks
        cls_token_position = input_ids.index(self.tokenizer.cls_token_id) + 1
        token_type_ids = [0] * (cls_token_position) + [1] * (len(input_ids) - cls_token_position)
        attention_mask = [1] * len(input_ids)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


def custom_collate_fn(batch):
    # Extracting input_ids, token_type_ids, and attention_mask from the batch
    input_ids = [item['input_ids'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    # Find the maximum sequence length in this batch
    max_len = max(len(ids) for ids in input_ids)

    # Pad all sequences to this maximum length
    padded_input_ids = torch.stack([torch.cat([ids, torch.zeros(max_len - len(ids), dtype=torch.long)]) for ids in input_ids])
    padded_token_type_ids = torch.stack([torch.cat([ids, torch.zeros(max_len - len(ids), dtype=torch.long)]) for ids in token_type_ids])
    padded_attention_mask = torch.stack([torch.cat([mask, torch.zeros(max_len - len(mask), dtype=torch.long)]) for mask in attention_mask])

    return {
        'input_ids': padded_input_ids,
        'token_type_ids': padded_token_type_ids,
        'attention_mask': padded_attention_mask
    }


def load_data(train_path, dev_path):

    # Load training and dev data
    train_data = pd.read_csv(train_path)
    dev_data = pd.read_csv(dev_path)

    train_data['rākau'] = train_data['rākau'].apply(json.loads)
    dev_data['rākau'] = dev_data['rākau'].apply(json.loads)

    return train_data, dev_data


class SequenceTokenizer:
    def __init__(self, worldstate_file='data/worldstate_tokens.txt', text_file='data/tokens.txt'):
        # Load tokens from files
        with open(worldstate_file, 'r') as f:
            worldstate_tokens = f.read().strip().split('\n')

        with open(text_file, 'r') as f:
            text_tokens = f.read().strip().split('\n')

        # Combine the tokens ensuring no overlap in indices
        all_tokens =  worldstate_tokens + text_tokens
        self.token_map = {token: i for i, token in enumerate(all_tokens)}
        self.id_map = {i: token for token, i in self.token_map.items()}
        self.vocab_size = len(self.token_map)
        self.cls_token_id = self.token_map['[CLS]']

    def tokenize(self, input_sequence):
        tokens = [self.token_map['[SOS]']]
        previous_type = None

        for element in input_sequence:
            current_type = 'world_state' if isinstance(element, dict) else 'text'

            # Insert [CLS] token between changes from text to world state or vice versa
            if previous_type is not None and previous_type != current_type:
                if previous_type == 'world_state':
                    tokens.append(self.token_map['[CLS]'])
                elif previous_type == 'text':
                    tokens.append(self.token_map['[EOS]'])

            if current_type == 'world_state':  # World state element
                tokens.extend([
                    self.token_map['[SELECTED]' if element['selected'] else '[NOT_SELECTED]'],
                    self.token_map[f"[COLOUR_{element['color'].upper()}]"],
                    self.token_map[f"[HEIGHT_{element['height']}]"]
                ])
            else:  # Text element
                tokens.extend(self.token_map[token] for token in element.split(' '))

            previous_type = current_type

        return tokens + [self.token_map['[EOS]']]

    def decode(self, ids):
        decoded_tokens = [self.id_map[id] for id in ids if id in self.id_map]
        if '[COLOUR_' in decoded_tokens[0]:  # Assuming world state output
            return [{'colour': token.split('_')[1].strip(']'), 'height': int(token.split('_')[2].strip(']'))}
                    for token in decoded_tokens if 'COLOUR' in token]
        else:  # Text output
            return ' '.join(decoded_tokens)


def create_mask_matrix(tokenizer, class_successors_json="data/class_successors.json", token_to_class_json="data/token_to_class.json"):

    class_successor_dict = json.load(open(class_successors_json))
    token_to_class_dict = json.load(open(token_to_class_json))

    class_to_tokens_dict = {k: list(chain.from_iterable([token_to_class_dict[l] for l in v])) for k, v in class_successor_dict.items()}

    source_to_target_token_dict = {k2: v1 for k1, v1 in class_to_tokens_dict.items() for k2 in token_to_class_dict[k1]}
    source_to_target_index_dict = {tokenizer.token_map[k]: [tokenizer.token_map[item] for item in v] for k, v in source_to_target_token_dict.items()}

    tuple_list = [(k, v) for k, vs in source_to_target_index_dict.items() for v in vs]

    # Initialize the mask matrix with zeros
    mask_matrix = torch.zeros((tokenizer.vocab_size, tokenizer.vocab_size), dtype=torch.float32)

    # Set valid transitions based on the tuple list
    for src, dest in tuple_list:
        mask_matrix[src, dest] = 1

    return mask_matrix


def encode_color(color):
    color_vector = [1 if color == col else 0 for col in COLOURS]
    return color_vector


def encode_world_state(sticks, num_locations=20, num_cols=10):
    num_colors = len(COLOURS)
    num_features = num_colors + 1  # Additional one for height

    # Initialize the matrix: Rows for colors + 1 for height, columns for locations
    matrix = np.zeros((num_features, num_locations))

    for stick in sticks:
        location = stick['location'] - 1  # Assuming locations are 1-indexed
        color_index = color_map[stick['color']]
        height = stick['height']

        # Set the color (one-hot encoding) and height
        matrix[color_index, location] = 1
        matrix[-1, location] = height  # Last row for height

    # We want to remove empty columns
    # Check which columns are all zeros
    is_zero_column = np.all(matrix == 0, axis=0)

    # Filter out columns that are all zeros
    matrix = matrix[:, ~is_zero_column]

    # Calculate how many columns we need to pad
    current_cols = matrix.shape[1]
    cols_to_add = num_cols - current_cols

    # Check if we need to add columns
    if cols_to_add > 0:
        # Create a zero matrix of the same number of rows and the deficit in columns
        zero_padding = np.zeros((matrix.shape[0], cols_to_add))

        # Concatenate the original matrix with the zero matrix on the right
        matrix = np.hstack((matrix, zero_padding))

    return matrix
