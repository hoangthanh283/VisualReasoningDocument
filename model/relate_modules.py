import torch
import torch.nn as nn
import torch.nn.functional as F


def _move_ptr_fw(stack_ptr):
    """
    Move the stack pointer forward (i.e. to push to stack).
    stack_ptr: (batch_size, stack_len)
    Return: (batch_size, stack_len)
    """
    filter_fw = torch.tensor([1, 0, 0]).float().view(1, 1, 3).to(stack_ptr.device)
    batch_size, stack_len = stack_ptr.shape

    new_stack_ptr = F.conv1d(
        stack_ptr.view(batch_size, 1, stack_len), filter_fw, padding=1).view(batch_size, stack_len)
    # when the stack pointer is already at the stack top, keep
    # the pointer in the same location (otherwise the pointer will be all zero)
    stack_top_mask = torch.zeros(stack_len).to(stack_ptr.device)
    stack_top_mask[stack_len - 1] = 1
    new_stack_ptr += stack_top_mask * stack_ptr
    return new_stack_ptr


def _move_ptr_bw(stack_ptr):
    """
    Move the stack pointer backward (i.e. to pop from stack).
    """
    filter_fw = torch.tensor([0, 0, 1]).float().view(1, 1, 3).to(stack_ptr.device)
    batch_size, stack_len = stack_ptr.shape

    new_stack_ptr = F.conv1d(
        stack_ptr.view(batch_size, 1, stack_len), filter_fw, padding=1).view(batch_size, stack_len)
    # when the stack pointer is already at the stack bottom, keep
    # the pointer in the same location (otherwise the pointer will be all zero)
    stack_bottom_mask = torch.zeros(stack_len).to(stack_ptr.device)
    stack_bottom_mask[0] = 1
    new_stack_ptr += stack_bottom_mask * stack_ptr
    return new_stack_ptr


def read_from_stack(att_stack, stack_ptr):
    """
    Read the value at the given stack pointer.
    """
    batch_size, stack_len = stack_ptr.shape
    stack_ptr_expand = stack_ptr.view(batch_size, 1, stack_len)
    # The stack pointer is a one-hot vector, so just do dot product
    att = torch.sum(att_stack * stack_ptr_expand, dim=-1).unsqueeze(-1)
    # (batch_size, att_dim, glimpse)
    return att


def write_to_stack(att_stack, stack_ptr, att):
    """
    Write value 'att' into the stack at the given stack pointer. Note that the
    result needs to be assigned back to att_stack
    """
    batch_size, stack_len = stack_ptr.size()
    stack_ptr_expand = stack_ptr.view(batch_size, 1, stack_len)
    att_stack = att.expand_as(att_stack) * stack_ptr_expand + att_stack * (1 - stack_ptr_expand)
    # (batch_size, att_dim, stack_len)
    return att_stack


def convert_to_one_hot(indices, num_classes):
    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.data.new(batch_size, num_classes).zero_().scatter_(1, indices.data, 1)
    return one_hot


def sharpen_ptr(stack_ptr, hard):
    """
    Sharpen the stack pointers into (nearly) one-hot vectors, using argmax
    or softmax. The stack values should always sum up to one for each instance.
    """
    if hard:
        # hard (non-differentiable) sharpening with argmax
        stack_len = stack_ptr.size(1)
        new_stack_ptr_indices = torch.argmax(stack_ptr, dim=1)[1]
        new_stack_ptr = convert_to_one_hot(new_stack_ptr_indices, stack_len)
    else:
        # soft (differentiable) sharpening with softmax
        temperature = 0.1
        new_stack_ptr = F.softmax(stack_ptr / temperature, dim=1)
    return new_stack_ptr


"""
Args:
    node_feat: (num_node, dim_v)
    query: (dim_v,)
    edge_matrices: (num_node, num_node)
    att_stack: (num_node, stack_len)
    stack_ptr: (stack_len,)
    value_mem: (num_node,)
"""


class NoOp(nn.Module):
    def forward(self, node_feat, query, edge_matrices,
                att_stack, stack_ptr):
        return att_stack, stack_ptr


class AttendKey(nn.Module):
    def __init__(self):
        super(AttendKey, self).__init__()

    def forward(self, node_feat, query, edge_matrices,
                att_stack, stack_ptr):
        logits = torch.matmul(node_feat, query.unsqueeze(-1))
        attn = torch.sigmoid(logits)

        stack_ptr = _move_ptr_fw(stack_ptr)
        att_stack = write_to_stack(att_stack, stack_ptr, attn).to(node_feat.device)
        return att_stack, stack_ptr


class TransferEdge(nn.Module):
    def __init__(self, input_dim):
        super(TransferEdge, self).__init__()
        self.linear = nn.Linear(input_dim, 4)

    def forward(self, node_feat, query, edge_matrices,
                att_stack, stack_ptr):
        query = F.softmax(self.linear(query), dim=-1)
        query = query.view(1, 1, -1).expand_as(edge_matrices)
        edge_attn = torch.sum(edge_matrices * query, dim=3)
        edge_attn = torch.sigmoid(edge_attn)

        key_attn = read_from_stack(att_stack, stack_ptr)
        key_attn = key_attn.permute(0, 2, 1)
        new_attn = torch.matmul(key_attn, edge_attn)
        new_attn = new_attn.permute(0, 2, 1)

        norm = torch.max(new_attn).detach()
        norm = 1 if norm <= 1 else norm
        new_attn = new_attn / norm

        att_stack = write_to_stack(att_stack, stack_ptr, new_attn).to(node_feat.device)
        return att_stack, stack_ptr
