# Causal assumptions

E0 uses a fully specified structural causal world. Every exogenous noise draw is
explicitly seeded, and paired counterfactual arms reuse the same draw. The action
structural equation alone is replaced at the intervention step.

Later experiments must label every sample as exact propensity, estimated propensity,
unknown propensity, or hidden-confounded. Importance sampling and causal point
estimates must reject incompatible samples instead of silently treating them as
identified. Observational chat analyses use the term *predictive contribution*.
