from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import optim
from echomario.agents.readout_policy import ReadoutActorCritic
from echomario.training.rollout import Rollout


@dataclass
class PPOStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    update_steps: int
    stopped_early: bool


def compute_gae(rollout: Rollout, gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = rollout.rewards if rollout.rewards.ndim == 2 else rollout.rewards.unsqueeze(-1)
    dones = rollout.dones if rollout.dones.ndim == 2 else rollout.dones.unsqueeze(-1)
    values = rollout.values if rollout.values.ndim == 2 else rollout.values.unsqueeze(-1)
    last_value = rollout.last_value if rollout.last_value.ndim == 1 else rollout.last_value.unsqueeze(0)
    last_value = last_value.reshape(-1)

    T, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(num_envs, dtype=rewards.dtype, device=rewards.device)
    next_value = last_value

    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]

    returns = advantages + values
    return advantages.reshape(-1), returns.reshape(-1)


def ppo_update(
    *,
    policy: ReadoutActorCritic,
    optimizer: optim.Optimizer,
    rollout: Rollout,
    gamma: float,
    gae_lambda: float,
    update_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    value_clip_eps: float = 0.0,
    target_kl: float = 0.0,
) -> PPOStats:
    advantages, returns = compute_gae(rollout, gamma=gamma, gae_lambda=gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = rollout.states.shape[0]
    idx = torch.arange(n, device=rollout.states.device)
    old_log_probs = rollout.log_probs.reshape(-1)
    old_values = rollout.values.reshape(-1)
    last_stats = None
    update_steps = 0
    stopped_early = False

    for _epoch in range(update_epochs):
        perm = idx[torch.randperm(n, device=rollout.states.device)]
        for start in range(0, n, minibatch_size):
            mb_idx = perm[start : start + minibatch_size]
            _, new_log_probs, entropy, new_values = policy.get_action_and_value(
                rollout.states[mb_idx],
                rollout.actions[mb_idx],
            )
            log_ratio = new_log_probs - old_log_probs[mb_idx]
            ratio = log_ratio.exp()
            mb_adv = advantages[mb_idx]
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss_unclipped = (new_values - returns[mb_idx]).pow(2)
            if value_clip_eps > 0.0:
                value_clipped = old_values[mb_idx] + torch.clamp(
                    new_values - old_values[mb_idx],
                    -value_clip_eps,
                    value_clip_eps,
                )
                value_loss_clipped = (value_clipped - returns[mb_idx]).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = 0.5 * value_loss_unclipped.mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio_detached = log_ratio.detach()
                approx_kl = ((log_ratio_detached.exp() - 1.0) - log_ratio_detached).mean()
                clip_fraction = (
                    (ratio.detach() - 1.0).abs() > clip_eps
                ).float().mean()
            update_steps += 1
            last_stats = PPOStats(
                loss=float(loss.detach().cpu()),
                policy_loss=float(policy_loss.detach().cpu()),
                value_loss=float(value_loss.detach().cpu()),
                entropy=float(entropy_loss.detach().cpu()),
                approx_kl=float(approx_kl.cpu()),
                clip_fraction=float(clip_fraction.cpu()),
                update_steps=update_steps,
                stopped_early=stopped_early,
            )
            if target_kl > 0.0 and float(approx_kl.detach().cpu()) > target_kl:
                stopped_early = True
                last_stats.stopped_early = True
                break
        if stopped_early:
            break

    assert last_stats is not None
    return last_stats
