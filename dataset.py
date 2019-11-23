import os
import glob
import json
import torch.utils.data
from natsort import natsorted
from preprocess.preprocess_document import (
    process_form_relation
)


def invert_dict(d):
    return {v: k for k, v in d.items()}


class DocumentDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, vocab, max_text_length=100):
        self.vocab = vocab
        self.max_text_length = max_text_length
        self.paths = natsorted(glob.glob(os.path.join(root_dir, "*.json")))

    def __getitem__(self, index):
        path = self.paths[index]
        data = json.load(open(path, encoding="utf-8"))

        question, vertex_vectors, edge_matrices, mask = \
            process_form_relation(data, self.vocab, self.max_text_length)
        return question, vertex_vectors, edge_matrices, mask

    def __len__(self):
        return len(self.paths)
