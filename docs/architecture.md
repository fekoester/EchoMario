# Architecture

EchoMario trains only the readout from a fixed reservoir.

u_t = observation features
x_t = reservoir(u_t, x_{t-1})
policy_logits_t = W_policy x_t + b_policy
value_t = W_value x_t + b_value
