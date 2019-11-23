import torch
import torch.nn as nn
from itertools import chain
from utils.misc import reverse_padded_sequence


class BiRnnEncoder(nn.Module):
    def __init__(self, dim_word, dim_hidden):
        super().__init__()
        self.forward_gru = nn.GRU(dim_word, dim_hidden//2)
        self.backward_gru = nn.GRU(dim_word, dim_hidden//2)
        for name, param in chain(self.forward_gru.named_parameters(), self.backward_gru.named_parameters()):
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)
        return

    def forward(self, input_sequence, input_embedded, sequence_lens):
        """
            Input:
                input_seqs: [seq_max_len, batch_size]
                input_seq_lens: [batch_size]
        """
        # [seq_max_len, batch_size, word_dim]
        embedded = input_embedded
        
        # [seq_max_len, batch_size, dim_hidden/2]
        forward_outputs = self.forward_gru(embedded)[0]
        backward_embedded = reverse_padded_sequence(embedded, sequence_lens)
        
        backward_outputs = self.backward_gru(backward_embedded)[0]
        backward_outputs = reverse_padded_sequence(backward_outputs, sequence_lens)
        
        # [seq_max_len, batch_size, dim_hidden]
        outputs = torch.cat([forward_outputs, backward_outputs], dim=2)
        
        # indexing outputs via input_seq_lens
        hidden = []
        for i, l in enumerate(sequence_lens):
            hidden.append(
                torch.cat([forward_outputs[l-1, i], backward_outputs[0, i]], dim=0))
        # (batch_size, dim)
        hidden = torch.stack(hidden)
        return outputs, hidden


class BowEncoder(nn.Module):
    def __init__(self, num_tokens, hidden_dim):
        super(BowEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_tokens, hidden_dim),
            nn.ReLU())

    def forward(self, x):
        out = self.fc(x)
        return out
