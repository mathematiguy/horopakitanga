import json
import torch
import random
import numpy as np
import pandas as pd
import itertools as it
from torch.utils.data import Dataset
from ataarangi.utils import split_chunks
from ataarangi.rakau import WorldState, Rākau
from torch.nn.utils.rnn import pad_sequence


COLOURS = ["red", "blue", "green", "yellow", "black", "white", "brown", "pink"]
MĀORI_COLOURS = [
    "whero",
    "kikorangi",
    "kākāriki",
    "kōwhai",
    "pango",
    "mā",
    "parauri",
    "māwhero",
]
color_map = dict(zip(COLOURS, range(len(COLOURS))))
māori_colour_map = dict(zip(COLOURS, MĀORI_COLOURS))
colour_to_id = dict(zip(COLOURS, range(len(COLOURS))))
id_to_colour = {v: k for k, v in colour_to_id.items()}


def create_permutation_matrix(elements):
    # Generate all permutations of the array of colors
    elem_indices = np.arange(len(elements))
    all_perms = list(it.permutations(elem_indices))
    num_perms = len(all_perms)

    # Create a 3D array to hold the permutation transformation matrices
    transformation_tensor = np.zeros(
        (num_perms, len(elements), len(elements)), dtype=int
    )

    for i, perm in enumerate(all_perms):
        # Fill the appropriate places in the matrix to indicate the movement
        transformation_tensor[i, elem_indices, perm] = 1

    return transformation_tensor


def generate_colour_permutations(indices, elements=COLOURS):
    """
    Generate all permutations of an array of unique elements of a given length, and filter
    such that each permutation moves at least one of the elements in `indices`.
    """
    permutation_matrix = create_permutation_matrix(elements)
    permutations = [
        dict(zip(elements, [elements[idx] for idx in np.where(perm)[1]]))
        for perm in permutation_matrix
    ]
    permutations = [
        {k: v for k, v in d.items() if colour_to_id[k] in indices} for d in permutations
    ]
    return [dict(t) for t in {tuple(d.items()) for d in permutations}]


def generate_ordered_size_permutations(sizes, min_size, max_size):
    length = len(set(sizes))
    sizes = list(set(sizes))
    from itertools import combinations

    # Generate all possible unique ordered combinations of sizes
    # within the given range that preserve order.
    all_sizes = range(min_size, max_size + 1)
    size_map = [
        dict(zip(range(length), comb)) for comb in combinations(all_sizes, length)
    ]
    return [{sizes[k]: v for k, v in d.items()} for d in size_map]


def generate_rākau_permutations(state, description, sample_size=1000):

    tokens = description.split()

    state_colours = [colour_to_id[r.color] for r in state.ngā_rākau]
    state_sizes = [r.height for r in state.ngā_rākau]
    state_locations = [r.location for r in state.ngā_rākau]
    colour_permutations = generate_colour_permutations(state_colours)
    size_permutations = generate_ordered_size_permutations(state_sizes, 1, 10)

    results = []
    for _ in range(sample_size):
        colour_map = random.choice(colour_permutations)
        size_map = random.choice(size_permutations)

        if not any([side in tokens for side in ["mauī", "matau"]]):
            location_map = dict(
                zip(
                    range(1, len(state_locations) + 1),
                    tuple(random.choices(range(10), k=len(state_locations))),
                )
            )
        else:
            if random.random() > 0.5:
                # Keep the order
                location_map = dict(
                    zip(
                        range(1, len(state_locations) + 1),
                        range(1, len(state_locations) + 1),
                    )
                )
            else:
                # Reverse the order
                location_map = dict(
                    zip(
                        range(1, len(state_locations) + 1),
                        list(range(1, len(state_locations) + 1))[::-1],
                    )
                )
                tokens = [
                    (
                        "mauī"
                        if token == "matau"
                        else "matau" if token == "mauī" else token
                    )
                    for token in tokens
                ]

        translated_colour_map = {
            māori_colour_map[k]: māori_colour_map[v] for k, v in colour_map.items()
        }
        translated_tokens = [
            token if not token in MĀORI_COLOURS else translated_colour_map[token]
            for token in tokens
        ]

        ngā_rākau = WorldState()
        for rākau in state.ngā_rākau:
            ngā_rākau.add_rākau(
                Rākau(
                    color=colour_map[rākau.color],
                    height=size_map[rākau.height],
                    location=location_map[rākau.location],
                    selected=rākau.selected,
                )
            )
        results.append({"state": ngā_rākau, "description": " ".join(translated_tokens)})

    return results


class RākauDataset(Dataset):

    def __init__(self, srcs, tgts, tokenizer):
        self.tokenizer = tokenizer
        self.srcs = srcs.apply(tokenizer.tokenize)
        self.tgts = tgts.apply(lambda x: tokenizer.tokenize([x]))

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        src_ids = self.srcs[idx]
        tgt_ids = self.tgts[idx]

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
        }

    def __getitem__(self, idx):
        src_ids = self.srcs[idx]
        tgt_ids = self.tgts[idx]

        # Create a combined sequence with special tokens if necessary
        # Example: [SOS] + src_ids + [SEP] + tgt_ids + [EOS]
        input_ids = src_ids + tgt_ids

        # Creating token type masks: 0 for src, 1 for tgt
        token_type_ids = [0] * len(src_ids) + [1] * len(tgt_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }


def rākau_collate_fn(batch):
    # Extract input_ids and token_type_ids from the batch
    input_ids = [item["input_ids"] for item in batch]
    token_type_ids = [item["token_type_ids"] for item in batch]

    # Pad the sequences. This requires all tensors to be of the same length
    # pad_sequence automatically pads with 0, which is typical for input IDs and should be fine for token_type_ids
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    token_type_ids_padded = pad_sequence(
        token_type_ids, batch_first=True, padding_value=1
    )

    return {"input_ids": input_ids_padded, "token_type_ids": token_type_ids_padded}


def load_data(train_path, dev_path):

    # Load training and dev data
    train_data = pd.read_csv(train_path)
    dev_data = pd.read_csv(dev_path)

    train_data["rākau"] = train_data["rākau"].apply(json.loads)
    dev_data["rākau"] = dev_data["rākau"].apply(json.loads)

    return train_data, dev_data


class SequenceTokenizer:
    def __init__(
        self, worldstate_file="data/worldstate_tokens.txt", text_file="data/tokens.txt"
    ):
        # Load tokens from files
        with open(worldstate_file, "r") as f:
            worldstate_tokens = f.read().strip().split("\n")

        with open(text_file, "r") as f:
            text_tokens = f.read().strip().split("\n")

        # Combine the tokens ensuring no overlap in indices
        all_tokens = worldstate_tokens + text_tokens
        self.token_map = {token: i for i, token in enumerate(all_tokens)}
        self.id_map = {i: token for token, i in self.token_map.items()}
        self.vocab_size = len(self.token_map)
        self.sep_token_id = self.token_map["[SEP]"]
        self.eos_token_id = self.token_map["[EOS]"]
        self.sos_token_id = self.token_map["[SOS]"]

    def tokenize(self, input_sequence):
        current_type = "world_state" if isinstance(input_sequence[0], dict) else "text"
        tokens = [self.token_map["[SOS]"]] if current_type == "world_state" else []
        previous_type = None

        for element in input_sequence:
            current_type = "world_state" if isinstance(element, dict) else "text"

            # Insert [SEP] token between changes from text to world state or vice versa
            if previous_type is not None and previous_type != current_type:
                if previous_type == "world_state":
                    tokens.append(self.token_map["[SEP]"])
                elif previous_type == "text":
                    tokens.append(self.token_map["[EOS]"])

            if current_type == "world_state":  # World state element
                tokens.extend(
                    [
                        self.token_map[
                            "[SELECTED]" if element["selected"] else "[NOT_SELECTED]"
                        ],
                        self.token_map[f"[COLOUR_{element['color'].upper()}]"],
                        self.token_map[f"[HEIGHT_{element['height']}]"],
                    ]
                )
            else:  # Text element
                tokens.extend(self.token_map[token] for token in element.split(" "))

            previous_type = current_type

        if previous_type == "text":
            tokens += [self.token_map["[EOS]"]]

        if previous_type == "world_state":
            tokens += [self.token_map["[SEP]"]]

        return tokens

    def decode(self, ids):
        decoded_tokens = [self.id_map[id] for id in ids if id in self.id_map]
        if "[COLOUR_" in decoded_tokens[0]:  # Assuming world state output
            return [
                {
                    "colour": token.split("_")[1].strip("]"),
                    "height": int(token.split("_")[2].strip("]")),
                }
                for token in decoded_tokens
                if "COLOUR" in token
            ]
        else:  # Text output
            return " ".join(decoded_tokens)
