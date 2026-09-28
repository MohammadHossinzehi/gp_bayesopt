"""Command line benchmark: Bayesian optimisation vs random search.

    python -m gpbo --benchmark branin --iters 30 --seeds 5 --acq ei
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .benchmarks import BENCHMARKS
from .optimizer import BayesianOptimizer, random_search


def main(argv=None):
    p = argparse.ArgumentParser(prog="gpbo", description=__doc__.splitlines()[0])
    p.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="branin")
    p.add_argument("--iters", type=int, default=30, help="function evaluations per run")
    p.add_argument("--seeds", type=int, default=5, help="independent runs to average")
    p.add_argument("--acq", choices=["ei", "pi", "lcb"], default="ei")
    p.add_argument("--verbose", action="store_true", help="print every evaluation of the first run")
    a = p.parse_args(argv)

    bench = BENCHMARKS[a.benchmark]
    print(f"{bench.name}: dim={len(bench.bounds)}  known minimum={bench.minimum:.5f}  "
          f"budget={a.iters} evals  acquisition={a.acq}\n")
    bo_gap, rs_gap = [], []
    t0 = time.perf_counter()
    for s in range(a.seeds):
        cb = None
        if a.verbose and s == 0:
            cb = lambda i, x, y, best: print(f"  eval {i:3d}  f={y:10.5f}  best={best:10.5f}  x={np.round(x, 4)}")
        bo = BayesianOptimizer(bench.bounds, acquisition=a.acq, seed=s).minimize(bench.func, a.iters, callback=cb)
        rs = random_search(bench.func, bench.bounds, a.iters, seed=s)
        bo_gap.append(bo.fun - bench.minimum)
        rs_gap.append(rs.fun - bench.minimum)
        print(f"seed {s}:  BO regret {bo_gap[-1]:.2e}   random search regret {rs_gap[-1]:.2e}")
    print(f"\nmedian regret   BO {np.median(bo_gap):.2e}   random {np.median(rs_gap):.2e}"
          f"   ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
