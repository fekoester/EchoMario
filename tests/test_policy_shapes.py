import torch

from echomario.agents.cnn_policy import TileCNNActorCritic
from echomario.agents.readout_policy import ReadoutActorCritic


def test_discrete_policy_shapes():
    model = ReadoutActorCritic(reservoir_dim=100, num_actions=7)
    x = torch.randn(4, 100)
    logits, values = model(x)
    assert logits.shape == (4, 7)
    assert values.shape == (4,)


def test_continuous_policy_shapes_and_bounds():
    model = ReadoutActorCritic(reservoir_dim=30, num_actions=3, continuous=True)
    x = torch.randn(5, 30)
    action, log_probs, entropy, values = model.get_action_and_value(x)

    assert action.shape == (5, 3)
    assert log_probs.shape == (5,)
    assert entropy.shape == (5,)
    assert values.shape == (5,)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)


def test_continuous_log_prob_recompute_matches():
    model = ReadoutActorCritic(reservoir_dim=12, num_actions=3, continuous=True)
    x = torch.randn(2, 12)
    action, log_probs, _entropy, _values = model.get_action_and_value(x)
    _action2, recomputed_log_probs, _entropy2, _values2 = model.get_action_and_value(x, actions=action)
    assert torch.allclose(log_probs, recomputed_log_probs, atol=1e-5)


def test_cnn_continuous_policy_shapes_and_bounds():
    model = TileCNNActorCritic(
        screen_channels=10,
        screen_height=18,
        screen_width=36,
        state_dim=8,
        num_actions=3,
        continuous=True,
    )
    x = torch.randn(4, 10 * 18 * 36 + 8)
    action, log_probs, entropy, values = model.get_action_and_value(x)

    assert action.shape == (4, 3)
    assert log_probs.shape == (4,)
    assert entropy.shape == (4,)
    assert values.shape == (4,)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
