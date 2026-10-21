import copy

import torch
import torch.nn as nn

from env import VALUE_MIN, VALUE_MAX, NUM_ACTIONS

NUM_VALUES = VALUE_MAX - VALUE_MIN + 1
BOS = 0
VAL_BASE = 1
TGT_BASE = VAL_BASE + NUM_VALUES
ACT_BASE = TGT_BASE + NUM_VALUES
VOCAB = ACT_BASE + NUM_ACTIONS


def value_token(v):
    return VAL_BASE + (v - VALUE_MIN)


def target_token(t):
    return TGT_BASE + (t - VALUE_MIN)


def action_token(a):
    return ACT_BASE + a


class Policy(nn.Module):
    def __init__(self, d_model=64, heads=4, layers=2, max_len=48):
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(VOCAB, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, NUM_ACTIONS)
        self.value_head = nn.Linear(d_model, 1)

    def hidden(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        x = self.token_embed(tokens) + self.pos_embed(positions)
        mask = torch.triu(torch.full((length, length), float("-inf"), device=tokens.device), diagonal=1)
        h = self.encoder(x, mask=mask)
        return self.norm(h)

    def action_logits(self, states):
        return self.action_head(states)

    def state_value(self, states):
        return self.value_head(states).squeeze(-1)


def reference_copy(policy):
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    return ref
