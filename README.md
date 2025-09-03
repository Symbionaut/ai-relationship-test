
# Relationship vs. Transactional AI — Continuity Stress Test (RMT-001)

**Goal:** Show, with a simple repeatable simulation, how a *relationship-based* AI (**Relational**) outperforms a *transactional* AI (**Transactional**) on stability and success under vague, real‑world prompts.

This repo generates two figures and a CSV from a synthetic but controlled experiment that models episode-by-episode learning with noise and rare deep dips (bad nights).

> Outputs (created after running the script):
> - `figs/episode_returns.png`
> - `figs/success_rate.png`
> - `data/results.csv`

---

## How it works (quick)

We model returns with a logistic learning curve + noise:
- **Relational**: higher learning rate, lower variance (continuity/memory/ritual reduce chaos).
- **Transactional**: lower learning rate, higher variance (context resets each time).

Success rate is a cumulative average over binary successes whose probability follows each condition’s curve.

This is a **simulation harness** you can extend later with real task metrics.

---

## Run it

```bash
# (optional) create a venv first
pip install -r requirements.txt
python run_experiment.py
```

Outputs will appear in `figs/` and `data/`.

---

## Interpreting the plots

- **Episode Returns**: Relational stabilizes earlier with fewer deep crashes; Transactional flails longer.
- **Success Rate**: Relational climbs faster/higher; Transactional lags and stays noisier.

**Why this matters:** In real life, people often prompt while tired/busy. Relationship adds continuity (memory, rituals, tone-fit), which reduces variance and increases actionable specificity—leading to more consistent wins, sooner.

---

## Customize

Tweak the top of `run_experiment.py`:
- `N_EPISODES` (default 720)
- `alpha_*`, `plateau_*`, `noise_*` (learning curve + variance)
- `deep_dip_prob_*` (rare bad nights)

---

## License

MIT
