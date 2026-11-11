# Architecture

EchoMario compares three ways of turning observations into actions, all trained with the same PPO loop:

| model | config | what is trained |
|---|---|---|
| **Frozen reservoir + readout** | `reservoir.type: sparse_esn` | only the linear policy/value readout |
| **Trained GRU** | `reservoir.type: none`, `agent.model: gru` | encoder, GRU and heads (truncated BPTT) |
| **Feed-forward baselines** | `reservoir.type: none`, `agent.model: readout` / `cnn` | linear or CNN actor-critic on the current observation |

## Frozen reservoir (echo state network)

```
u_t = observation features
x_t = (1 - a) x_{t-1} + a tanh(W_res x_{t-1} + W_in u_t)      # fixed random W_res, W_in
s_t = x_t / rms(x_t)
policy_t = W_policy s_t + b_policy
value_t  = W_value  s_t + b_value
```

`W_res` is ~1 % dense and is applied as a sparse matrix. Tile observations are ~90 % zeros, so with few
environments the input projection `W_in u_t` uses a sparse path; with large batches it uses a dense torch matmul
(optionally on GPU via `reservoir.device: cuda`). Both give the same result as the dense formula.

## Trained GRU baseline

```
h_t = GRU(ReLU(W_enc u_t), h_{t-1})
policy_t, value_t = heads(h_t)
```

During rollouts the hidden state before every step is stored. PPO re-runs the GRU over chunks of
`training.seq_len` steps starting from the stored hidden state (resetting it where episodes ended), so the
recomputed log-probabilities match the rollout exactly.

## PPO details

- Continuous actions use a tanh-squashed Gaussian. Rollouts store the **raw** (pre-tanh) sample and PPO
  re-evaluates it exactly; reconstructing it with `atanh` is lossy once the policy saturates.
- **Time-limit truncation** is bootstrapped: on `truncated and not terminated` the reward is augmented with
  `gamma * V(s_T)`, where `s_T` is computed without advancing the reservoir/GRU. Set
  `training.bootstrap_truncation: false` to reproduce the pre-0.2 behaviour.
- Evaluation saves and restores recurrent policy state, so mid-training evaluation does not disturb rollouts.
