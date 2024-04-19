import torch
import numpy as np
from torch.utils.data import Dataset


COLOURS = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'pink']
color_map = dict(zip(COLOURS, range(len(COLOURS))))

 
class RākauDataset(Dataset):
    def __init__(self, raw_world_states, raw_text_data, world_state_tokenizer, text_tokenizer):
        self.world_state_tokenizer = world_state_tokenizer
        self.text_tokenizer = text_tokenizer

        # Tokenize raw data
        self.world_state_data = [self.world_state_tokenizer.tokenize(ws) for ws in raw_world_states]
        self.text_data = [self.text_tokenizer.tokenize(text) for text in raw_text_data]

    def __len__(self):
        return len(self.world_state_data)

    def __getitem__(self, idx):
        world_state_tokens = self.world_state_data[idx]
        text_tokens = self.text_data[idx]

        # Combine tokens with special tokens
        cls_token_id = self.world_state_tokenizer.token_map[self.world_state_tokenizer.cls_token]
        end_token_id = self.text_tokenizer.token_map[self.text_tokenizer.end_token]
        
        input_ids = world_state_tokens + [cls_token_id] + text_tokens + [end_token_id]
        
        # Create token type IDs and attention masks
        token_type_ids = [0] * (len(world_state_tokens) + 1) + [1] * (len(text_tokens) + 1)
        attention_mask = [1] * len(input_ids)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


def custom_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    # Pad sequences
    input_ids = torch.tensor(dataset.pad_sequences(input_ids), dtype=torch.long)
    token_type_ids = torch.tensor(dataset.pad_sequences(token_type_ids), dtype=torch.long)
    attention_mask = torch.tensor(dataset.pad_sequences(attention_mask), dtype=torch.long)

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }


class WorldStateTokenizer:
    def __init__(self, token_file='../data/worldstate_tokens.txt'):
        self.token_file = token_file

        # Load tokens from a file
        with open(self.token_file, 'r') as f:
            self.tokens = f.read().strip().split('\n')
            self.cls_token = self.tokens[-1]  # Assuming the last token is a special end token

        # Create a mapping from tokens to their indices
        self.token_map = {token: i + 1 for i, token in enumerate(self.tokens)}

    def tokenize(self, world_state):
        token_sequence = []
        tokens = []
        world_state = sorted(world_state, key=lambda x: x['location'])
        for stick in world_state:
            # Generate tokens for each attribute of the stick
            color_token = f"color_{stick['color']}"
            height_token = f"height_{stick['height']}"

            # Append token indices to the sequence
            tokens.extend([
                self.token_map[color_token],
                self.token_map[height_token]
            ])

        # Add the end token index
        tokens.append(self.token_map[self.cls_token])
            
        return tokens


class TextTokenizer:
    
    def __init__(self, token_file='../data/tokens.txt', min_index=20):
        self.token_file = token_file
        self.min_index = min_index

        with open(self.token_file, 'r') as f:
            self.tokens = f.read().strip().split('\n')
            self.end_token = self.tokens[-1]

        self.token_map = dict(zip(self.tokens, [self.min_index + i for i in range(len(self.tokens))]))

    def tokenize(self, s):
        tokens = [self.token_map[token] for token in s.split(' ')] + [self.token_map[self.end_token]]
        return tokens


def generate_token_file(token_file_path):
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'pink']
    heights = range(1, 11)
    locations = range(1, 11)

    tokens = []
    for color in colors:
        tokens.append(f"color_{color}")
    for height in heights:
        tokens.append(f"height_{height}")
    for location in locations:
        tokens.append(f"location_{location}")

    tokens.append('<END>')  # Add an end token

    with open(token_file_path, 'w') as f:
        for token in tokens:
            f.write(token + '\n')


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
