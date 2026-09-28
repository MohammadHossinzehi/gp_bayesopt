"""Render a step by step picture of Bayesian optimisation on a 1D function.

    pip install matplotlib
    python examples/visualize_bo.py            # writes bo_steps.png

Each panel shows the GP posterior mean, its 95% band, the evaluated points,
and (below) the expected improvement curve whose argmax is sampled next.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gpbo import BayesianOptimizer, expected_improvement  # noqa: E402
from gpbo.benchmarks import BENCHMARKS  # noqa: E402

bench = BENCHMARKS["forrester"]
opt = BayesianOptimizer(bench.bounds, n_initial=3, seed=1)
grid = np.linspace(0, 1, 400)[:, None]
truth = np.array([bench.func(x) for x in grid])

steps = 6
fig, axes = plt.subplots(2, steps, figsize=(3.2 * steps, 5.5), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})
for _ in range(3):
    x = opt.ask()
    opt.tell(x, bench.func(x))

for i in range(steps):
    x_next = opt.ask()  # fits the GP and maximises EI
    mu, sd = opt.gp.predict(grid, return_std=True)
    ei = expected_improvement(mu, sd, best=min(opt._y), xi=0.0)
    res = opt.result()
    top, bot = axes[0, i], axes[1, i]
    top.plot(grid, truth, "k--", lw=1, label="objective")
    top.plot(grid, mu, color="C0", label="GP mean")
    top.fill_between(grid[:, 0], mu - 2 * sd, mu + 2 * sd, color="C0", alpha=0.2)
    top.scatter(res.X[:, 0], res.y, color="C3", zorder=3, label="evaluations")
    top.axvline(x_next[0], color="C2", lw=1)
    top.set_title(f"{res.n_evals} evals, best {res.fun:.3f}")
    bot.plot(grid, ei, color="C2")
    bot.axvline(x_next[0], color="C2", lw=1)
    bot.set_yticks([])
    opt.tell(x_next, bench.func(x_next))

axes[0, 0].legend(loc="upper left", fontsize=8)
axes[1, 0].set_ylabel("EI")
fig.tight_layout()
out = Path("bo_steps.png")
fig.savefig(out, dpi=110)
print(f"wrote {out.resolve()}  (final best {opt.result().fun:.4f}, true min {bench.minimum:.4f})")
