#!/usr/bin/env python
"""Write a small synthetic dataset in exact ML-25M file format (for smoke tests)."""
import argparse

import _cli  # noqa: F401  (sys.path)
from neptune.synthetic import write_synthetic_ml25m

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", default="artifacts/raw/synthetic")
p.add_argument("--users", type=int, default=300)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
print(write_synthetic_ml25m(a.out, n_users=a.users, seed=a.seed))
