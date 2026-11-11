"""Regression tests for the PPO fixes: raw continuous actions, truncation bootstrapping, recurrent PPO."""
import numpy as np
import torch

from echomario.agents.gru_policy import GRUActorCritic
from echomario.agents.readout_policy import ReadoutActorCritic
from echomario.reservoirs.esn import IdentityReservoir
from echomario.training.ppo import ppo_update
from echomario.training.rollout import collect_rollout_parallel


def test_raw_actions_reproduce_log_probs_when_saturated():
    torch.manual_seed(0)
    policy = ReadoutActorCritic(reservoir_dim=4, num_actions=3, continuous=True, log_std_init=0.0)
    with torch.no_grad():
        policy.policy.bias.fill_(6.0)  # strongly saturated policy: tanh(raw) ~ 1
    x = torch.randn(256, 4)
    stored, env_action, log_prob, _ = policy.act(x)
    _, exact, _, _ = policy.get_action_and_value(x, stored, actions_are_raw=True)
    _, lossy, _, _ = policy.get_action_and_value(x, env_action)
    assert torch.allclose(exact, log_prob, atol=1e-4)
    # The old atanh reconstruction is visibly wrong in this regime.
    assert (lossy - log_prob).abs().max() > 1.0


class _TimeLimitEnv:
    """Constant reward 1, never terminates, truncates every ``limit`` steps."""

    def __init__(self, limit=3):
        from gymnasium import spaces

        self.observation_space = spaces.Box(-1, 1, (2,), np.float32)
        self.action_space = spaces.Discrete(2)
        self.limit, self.t = limit, 0

    def reset(self, **_kwargs):
        self.t = 0
        return np.ones(2, np.float32), {}

    def step(self, _action):
        self.t += 1
        return np.ones(2, np.float32), 1.0, False, self.t >= self.limit, {}


class _ConstantValue(ReadoutActorCritic):
    def __init__(self):
        super().__init__(reservoir_dim=2, num_actions=2)

    def forward(self, states):
        return torch.zeros(states.shape[0], 2), torch.full((states.shape[0],), 5.0)


def _rollout(gamma):
    env = _TimeLimitEnv()
    obs, _ = env.reset()
    return collect_rollout_parallel(
        envs=[env], reservoir=IdentityReservoir(2), policy=_ConstantValue(), obs_batch=obs[None],
        rollout_steps=6, device=torch.device("cpu"), reset_fn=lambda _i: env.reset()[0], gamma=gamma,
    )


def test_truncation_bootstraps_value():
    rewards = _rollout(gamma=0.9).rewards[:, 0].tolist()
    assert rewards == [1.0, 1.0, 1.0 + 0.9 * 5.0, 1.0, 1.0, 1.0 + 0.9 * 5.0]
    assert _rollout(gamma=None).rewards[:, 0].tolist() == [1.0] * 6  # legacy behaviour


def test_gru_sequence_evaluation_matches_stepwise_rollout():
    torch.manual_seed(0)
    policy = GRUActorCritic(obs_dim=5, num_actions=3, continuous=True, hidden_dim=16, encoder_dim=8)
    T, N = 12, 2
    states = torch.randn(T, N, 5)
    dones = torch.zeros(T, N)
    dones[4, 0] = 1.0
    policy.reset_state(batch_size=N)
    hiddens, actions, log_probs = [], [], []
    with torch.no_grad():
        for t in range(T):
            hiddens.append(policy.get_hidden(N, "cpu"))
            raw, _env, lp, _v = policy.act(states[t])
            actions.append(raw)
            log_probs.append(lp)
            if dones[t].any():
                policy.reset_state(mask=torch.nonzero(dones[t]).flatten().tolist())
    prev_dones = torch.cat([torch.zeros(1, N), dones[:-1]])
    lp_seq, _, _ = policy.evaluate_sequence(
        states, hiddens[0], prev_dones, torch.stack(actions), actions_are_raw=True
    )
    assert torch.allclose(lp_seq, torch.stack(log_probs), atol=1e-4)


def test_recurrent_ppo_update_runs_and_keeps_ratio_one_on_first_minibatch():
    torch.manual_seed(0)
    env = _TimeLimitEnv(limit=5)
    obs, _ = env.reset()
    policy = GRUActorCritic(obs_dim=2, num_actions=2, hidden_dim=8, encoder_dim=8)
    rollout = collect_rollout_parallel(
        envs=[env], reservoir=IdentityReservoir(2), policy=policy, obs_batch=obs[None],
        rollout_steps=16, device=torch.device("cpu"), reset_fn=lambda _i: env.reset()[0], gamma=0.99,
    )
    stats = ppo_update(
        policy=policy, optimizer=torch.optim.Adam(policy.parameters(), lr=1e-3), rollout=rollout,
        gamma=0.99, gae_lambda=0.95, update_epochs=1, minibatch_size=1000, clip_eps=0.2,
        value_coef=0.5, entropy_coef=0.0, max_grad_norm=0.5, seq_len=8,
    )
    assert stats.update_steps == 1
    assert stats.approx_kl < 1e-6  # recomputed log-probs match the rollout exactly
