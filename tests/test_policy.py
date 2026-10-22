import torch

from env import NUM_ACTIONS, VALUE_MIN, VALUE_MAX
from policy import (Policy, reference_copy, value_token, target_token, action_token,
                    VOCAB, BOS)


def test_token_ids_are_distinct_and_in_range():
    ids = [BOS, value_token(VALUE_MIN), value_token(VALUE_MAX),
           target_token(VALUE_MIN), target_token(VALUE_MAX),
           action_token(0), action_token(NUM_ACTIONS - 1)]
    assert all(0 <= i < VOCAB for i in ids)
    assert len(set(ids)) == len(ids)


def test_hidden_shape():
    policy = Policy(d_model=32, heads=2, layers=1)
    tokens = torch.tensor([[BOS, value_token(3), target_token(9)]])
    h = policy.hidden(tokens)
    assert h.shape == (1, 3, 32)


def test_action_logits_and_value_shapes():
    policy = Policy(d_model=32, heads=2, layers=1)
    tokens = torch.tensor([[BOS, value_token(3), target_token(9)]])
    states = policy.hidden(tokens)[0]
    assert policy.action_logits(states).shape == (3, NUM_ACTIONS)
    assert policy.state_value(states).shape == (3,)


def test_causal_mask_blocks_future_tokens():
    policy = Policy(d_model=32, heads=2, layers=1)
    torch.manual_seed(0)
    base = torch.tensor([[BOS, value_token(3), target_token(9)]])
    changed = torch.tensor([[BOS, value_token(3), target_token(1)]])
    hb = policy.hidden(base)
    hc = policy.hidden(changed)
    assert torch.allclose(hb[0, 0], hc[0, 0], atol=1e-5)
    assert torch.allclose(hb[0, 1], hc[0, 1], atol=1e-5)
    assert not torch.allclose(hb[0, 2], hc[0, 2], atol=1e-5)


def test_reference_copy_is_frozen_and_matches():
    policy = Policy(d_model=32, heads=2, layers=1)
    ref = reference_copy(policy)
    assert all(not p.requires_grad for p in ref.parameters())
    tokens = torch.tensor([[BOS, value_token(3), target_token(9)]])
    assert torch.allclose(policy.hidden(tokens), ref.hidden(tokens))
