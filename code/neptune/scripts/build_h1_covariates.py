#!/usr/bin/env python
"""I01_COVARIATES -- clusters, lambda_hat, tau^2 (block + multinomial), DEFF, split-half, tail."""
from _cli import parser, setup

from neptune.config import build_run_config
from neptune.pipeline.covariates import build_covariates

args = parser(__doc__).parse_args()
prereg, profile = setup(args)
cov, man = build_covariates(args.data, build_run_config(prereg, profile, for_training=False))
print({k: man[k] for k in ("n_users", "corr_lambda_logn", "n_single_session_users",
                           "n_split_half_usable", "deff_median")})
print(cov[["n_train", "lambda_hat", "tau2_block", "deff", "tail_test"]].describe().round(4))
