import torch
import torch.nn as nn
import exp_document.model.basic_modules as modules
from exp_document.model.question_encoder import BiRnnEncoder
from exp_document.model.controller import Controller


class VQA(nn.Module):
    def __init__(self, num_tokens, word_dim, hidden_dim,
                 stack_size, text_len, device):
        super(VQA, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.max_steps = self.stack_size = stack_size
        self.text_len = text_len
        self.device = device

        self.token_embedding = nn.Embedding(num_tokens, word_dim)
        self.question_encoder = BiRnnEncoder(word_dim, hidden_dim)

        self.module_names = ["attend_key", "search_neighbors"]
        self.model_modules = [
            modules.AttendKey(),
            modules.SearchNeighbors(hidden_dim),
        ]
        for name, module in zip(self.module_names, self.model_modules):
            self.add_module(name, module)

        self.num_module = len(self.model_modules)
        self.controller = Controller(self.num_module, hidden_dim, self.max_steps, True, device)

    def forward(self, questions, questions_len, vertex_vectors, edge_matrices):
        batch_size = questions.shape[0]
        num_node = vertex_vectors.shape[1]

        # node encoding
        node_feat = torch.zeros([batch_size, num_node, self.hidden_dim]).float().to(self.device)
        for i in range(0, vertex_vectors.shape[1]):
            vertex_text = vertex_vectors[:, i, :].permute(1, 0)
            embedding = self.token_embedding(vertex_text)
            embedding = torch.tanh(embedding)
            text_len = torch.full([batch_size], self.text_len).long().to(self.device)
            text_outputs, text_hidden = self.question_encoder(vertex_text, embedding, text_len)
            node_feat[:, i, :] = text_hidden

        # question encoding
        questions = questions.permute(1, 0)
        questions_embedding = self.token_embedding(questions)
        questions_embedding = torch.tanh(questions_embedding)
        questions_outputs, questions_hidden = self.question_encoder(questions, questions_embedding, questions_len)
        module_logits, module_probs, c_list, cv_list = self.controller(
            questions_outputs, questions_hidden, questions_embedding, questions_len)

        # stack initialization
        att_stack = torch.full([batch_size, num_node, self.stack_size], 0.0).to(self.device)
        stack_ptr = torch.zeros(batch_size, self.stack_size).to(self.device)
        stack_ptr[:, 0] = 1
        value_mem = torch.zeros(batch_size, num_node).to(self.device)
        # cache for visualization
        cache_module_prob = []
        cache_attn = []

        for t in range(0, self.max_steps):
            c_i = questions_hidden
            module_prob = module_probs[t].permute(1, 0)

            # run all modules
            out = [m(
                node_feat, c_i, edge_matrices, att_stack, stack_ptr, value_mem)
                for m in self.model_modules]
            att_stack_avg = torch.sum(
                module_prob.view(self.num_module, batch_size, 1, 1) * torch.stack([r[0] for r in out]), dim=0)
            stack_ptr_avg = torch.sum(
                module_prob.view(self.num_module, batch_size, 1) * torch.stack([r[1] for r in out]), dim=0)
            stack_ptr_avg = modules.sharpen_ptr(stack_ptr_avg, hard=False)
            value_mem_avg = torch.sum(
                module_prob.view(self.num_module, batch_size, 1) * torch.stack([r[2] for r in out]), dim=0)
            att_stack, stack_ptr, value_mem = att_stack_avg, stack_ptr_avg, value_mem_avg

            # cache for visualization
            cache_module_prob.append(module_prob)
            attentions = []
            for r in out:
                att = modules.read_from_stack(r[0], r[1])
                attentions.append(att)
            cache_attn.append(attentions)

        value_mem = torch.sigmoid(value_mem)
        return value_mem, cache_module_prob, cache_attn

    def infer(self, questions, questions_len, vertex_vectors, edge_matrices, raw_nodes):
        batch_size = questions.shape[0]
        num_node = vertex_vectors.shape[1]

        # node encoding
        node_feat = torch.zeros([batch_size, num_node, self.hidden_dim]).float().to(self.device)
        for i in range(0, vertex_vectors.shape[1]):
            vertex_text = vertex_vectors[:, i, :].permute(1, 0)
            embedding = self.token_embedding(vertex_text)
            embedding = torch.tanh(embedding)
            text_len = torch.full([batch_size], self.text_len).long().to(self.device)
            text_outputs, text_hidden = self.question_encoder(vertex_text, embedding, text_len)
            node_feat[:, i, :] = text_hidden

        # question encoding
        questions = questions.permute(1, 0)
        questions_embedding = self.token_embedding(questions)
        questions_embedding = torch.tanh(questions_embedding)
        questions_outputs, questions_hidden = self.question_encoder(questions, questions_embedding, questions_len)
        module_logits, module_probs, c_list, cv_list = self.controller(
            questions_outputs, questions_hidden, questions_embedding, questions_len)

        # stack initialization
        att_stack = torch.full([batch_size, num_node, self.stack_size], 0.0).to(self.device)
        stack_ptr = torch.zeros(batch_size, self.stack_size).to(self.device)
        stack_ptr[:, 0] = 1
        value_mem = torch.zeros(batch_size, num_node).to(self.device)

        for t in range(0, self.max_steps):
            c_i = questions_hidden
            module_prob = module_probs[t].permute(1, 0)

            # run all modules
            out = [m(
                node_feat, c_i, edge_matrices, att_stack, stack_ptr, value_mem)
                for m in self.model_modules]
            att_stack_avg = torch.sum(
                module_prob.view(self.num_module, batch_size, 1, 1) * torch.stack([r[0] for r in out]), dim=0)
            stack_ptr_avg = torch.sum(
                module_prob.view(self.num_module, batch_size, 1) * torch.stack([r[1] for r in out]), dim=0)
            stack_ptr_avg = modules.sharpen_ptr(stack_ptr_avg, hard=False)
            value_mem_avg = torch.sum(
                module_prob.view(self.num_module, batch_size, 1) * torch.stack([r[2] for r in out]), dim=0)
            att_stack, stack_ptr, value_mem = att_stack_avg, stack_ptr_avg, value_mem_avg

            # visualization
            print("\nStep %i" % t)
            print("Question attention")
            print(cv_list[t].cpu().numpy().tolist())
            print("Module probability")
            print(module_prob.data.cpu().numpy().tolist())
            print("Stack pointer")
            print(stack_ptr.cpu().numpy().tolist())

            print("Key attention")
            key_attn = modules.read_from_stack(att_stack, stack_ptr)
            key_attn = key_attn.squeeze(-1)
            top_attn, top_indices = key_attn.topk(3, sorted=True)
            top_indices = top_indices.cpu().numpy()
            key_nodes = raw_nodes[top_indices[0, :]]
            print(key_nodes.tolist(), top_attn.cpu().numpy()[0, :])
            print(key_attn.min(), key_attn.max())

            print("Value attention")
            value_attn = torch.sigmoid(value_mem)
            good_indices = torch.nonzero(value_attn > 0.6)
            if good_indices.numel() == 0:
                print("None")
                continue
            good_indices = good_indices.cpu().numpy()[:, 1]
            value_nodes = raw_nodes[good_indices]
            good_attn = value_attn.cpu().numpy()[0, good_indices]
            print(value_nodes.tolist(), good_attn)

        value_mem = torch.sigmoid(value_mem)
        return value_mem
