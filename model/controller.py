import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Controller(nn.Module):
    def __init__(self, num_modules, lstm_dim, max_step, use_gumbel, device):
        super().__init__()
        self.num_modules = num_modules
        self.lstm_dim = lstm_dim
        self.max_step = max_step
        self.use_gumbel = use_gumbel
        self.device = device

        self.fc_q_list = []
        # W_1^{(t)} q + b_1
        for t in range(self.max_step):
            self.fc_q_list.append(nn.Linear(self.lstm_dim, self.lstm_dim))
            self.add_module("fc_q_%d" % t, self.fc_q_list[t])

        # W_2 [q;c] + b_2
        self.fc_q_cat_c = nn.Linear(2 * self.lstm_dim, self.lstm_dim)
        self.fc_module_weight = nn.Sequential(
            nn.Linear(self.lstm_dim, self.lstm_dim),
            nn.ReLU(),
            nn.Linear(self.lstm_dim, self.num_modules))
        self.fc_raw_cv = nn.Linear(self.lstm_dim, 1)
        self.c_init = nn.Parameter(torch.zeros(1, self.lstm_dim).normal_(mean=0, std=np.sqrt(1 / self.lstm_dim)))

    def forward(self, lstm_seq, q_encoding, embed_seq, seq_length_batch):
        """        
        Input:
            lstm_seq: [seq_max_len, batch_size, d]
            q_encoding: [batch_size, d]
            embed_seq: [seq_max_len, batch_size, e]
            seq_length_batch: [batch_size]
        """
        batch_size, seq_max_len = lstm_seq.size(1), lstm_seq.size(0)
        # [seq_max_len, batch_size]
        seq_length_batch = seq_length_batch.view(1, batch_size).expand(seq_max_len, batch_size)
        # (batch_size, dim)
        c_prev = self.c_init.expand(batch_size, self.lstm_dim)
        module_logit_list = []
        module_prob_list = []
        c_list, cv_list = [], []

        for t in range(self.max_step):
            q_i = self.fc_q_list[t](q_encoding)
            q_i_c = torch.cat([q_i, c_prev], dim=1)  # [batch_size, 2d]
            cq_i = self.fc_q_cat_c(q_i_c)  # [batch_size, d]
            module_logit = self.fc_module_weight(cq_i)  # [batch_size, num_module]
            module_prob = F.gumbel_softmax(module_logit, hard=self.use_gumbel)  # [batch_size, num_module]

            elem_prod = cq_i.unsqueeze(0) * lstm_seq  # [seq_max_len, batch_size, dim]
            raw_cv_i = self.fc_raw_cv(elem_prod).squeeze(2)  # [seq_max_len, batch_size]
            invalid_mask = torch.arange(seq_max_len).long().to(self.device).view(
                -1, 1).expand_as(raw_cv_i).ge(seq_length_batch)
            raw_cv_i.data.masked_fill_(invalid_mask, -float("inf"))
            cv_i = F.softmax(raw_cv_i, dim=0).unsqueeze(2)  # [seq_max_len, batch_size, 1]
            c_i = torch.sum(lstm_seq * cv_i, dim=0)  # [batch_size, d]
            assert c_i.size(0) == batch_size and c_i.size(1) == self.lstm_dim

            c_prev = c_i
            # add into results
            module_logit_list.append(module_logit)
            module_prob_list.append(module_prob)
            c_list.append(c_i)
            cv_list.append(cv_i.squeeze(2).permute(1, 0))

        return (torch.stack(module_logit_list),  # [T_ctrl, batch_size, num_module]
                torch.stack(module_prob_list),  # [T_ctrl, batch_size, num_module]
                torch.stack(c_list),  # [T_ctrl, batch_size, d]
                torch.stack(cv_list))  # [T_ctrl, batch_size, seq_max_len]
