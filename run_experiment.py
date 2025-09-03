
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(42)

# -----------------------------
# Experiment config
# -----------------------------
N_EPISODES = 720

# "Return" caps (think: max useful value per session)
RETURN_CAP = 9.5   # ~ top of y-axis

# Relational: faster ramp, lower variance
alpha_rel = 0.030       # learning rate
plateau_rel = 0.92      # asymptotic success prob
noise_rel = 0.9         # shock size for returns (lower = steadier)

# Transactional: slower ramp, higher variance
alpha_tx = 0.018
plateau_tx = 0.86
noise_tx = 1.8

# Rare deep dips (bad nights)
deep_dip_prob_rel = 0.010
deep_dip_prob_tx  = 0.025
deep_dip_size = 10.0  # how far returns can plunge from current mean

def logistic_curve(alpha, n, midpoint=120):
    x = np.arange(n)
    return 1.0 / (1.0 + np.exp(-alpha * (x - midpoint)))

def simulate_condition(alpha, plateau, noise_scale, deep_dip_prob):
    base = logistic_curve(alpha, N_EPISODES)  # 0..1
    p_success = np.clip(base * plateau, 0, 0.99)

    returns = np.zeros(N_EPISODES)
    successes = np.zeros(N_EPISODES, dtype=int)

    for t in range(N_EPISODES):
        mu = p_success[t] * RETURN_CAP
        if rng.random() < deep_dip_prob:
            shock = -deep_dip_size * rng.random()
        else:
            shock = rng.normal(0, noise_scale)
        ret = np.clip(mu + shock, -22, RETURN_CAP)
        returns[t] = ret
        successes[t] = 1 if rng.random() < p_success[t] else 0

    return returns, successes

# Simulate both modes
ret_rel, suc_rel = simulate_condition(alpha_rel, plateau_rel, noise_rel, deep_dip_prob_rel)
ret_tx,  suc_tx  = simulate_condition(alpha_tx,  plateau_tx,  noise_tx,  deep_dip_prob_tx)

def cumulative_rate(successes):
    ep = np.arange(1, len(successes)+1)
    return np.cumsum(successes) / ep

rate_rel = cumulative_rate(suc_rel)
rate_tx  = cumulative_rate(suc_tx)

# Save tidy data
out_dir_data = Path("data")
out_dir_figs = Path("figs")
out_dir_data.mkdir(parents=True, exist_ok=True)
out_dir_figs.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame({
    "episode": np.arange(N_EPISODES),
    "return_relational": ret_rel,
    "return_transactional": ret_tx,
    "success_relational": suc_rel,
    "success_transactional": suc_tx,
    "rate_relational": rate_rel,
    "rate_transactional": rate_tx
})
df.to_csv(out_dir_data / "results.csv", index=False)

# Plot: Episode Returns
plt.figure(figsize=(12, 6))
plt.plot(df["episode"], df["return_transactional"], label="Transactional", linewidth=1.3, alpha=0.9)
plt.plot(df["episode"], df["return_relational"], label="Relational", linewidth=1.3, alpha=0.9)
plt.title("Episode Returns")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend(frameon=True, loc="lower right")
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig(out_dir_figs / "episode_returns.png", dpi=180)

# Plot: Success Rate
plt.figure(figsize=(12, 6))
plt.plot(df["episode"], df["rate_transactional"], label="Transactional", linewidth=1.6, alpha=0.95)
plt.plot(df["episode"], df["rate_relational"], label="Relational", linewidth=1.6, alpha=0.95)
plt.title("Success Rate")
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.ylim(0.0, 1.0)
plt.legend(frameon=True, loc="lower right")
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig(out_dir_figs / "success_rate.png", dpi=180)

print("Done. Wrote figs/ and data/results.csv")
