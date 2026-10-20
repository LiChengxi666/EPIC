# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Extracted unchanged from the bundled experiment implementation.
import torch


def _solve_exact_prefix_kl_log_bounds(log_budget: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``exp(z) - 1 - z = exp(log_budget)`` on both branches.

    ``z`` is the cumulative prefix log-ratio. The equation has one negative
    and one positive root for every positive budget. A direct ratio-space
    solve is numerically unusable for long prefixes because
    ``t * delta / pi_old(prefix)`` quickly exceeds the floating-point range.
    This routine therefore works from its logarithm and uses:

    - a local series for very small budgets;
    - Newton iterations while the budget is directly representable;
    - asymptotically exact log-space updates for large budgets.

    The returned tensors are detached trust-region boundaries.
    """

    with torch.no_grad():
        work_dtype = torch.float32 if log_budget.dtype in (torch.float16, torch.bfloat16) else log_budget.dtype
        log_c = log_budget.detach().to(work_dtype)

        # The local roots are z+ = sqrt(2c) - c/3 + O(c^(3/2)) and
        # z- = -sqrt(2c) - c/3 + O(c^(3/2)).
        c_for_newton = torch.exp(torch.clamp(log_c, max=12.0))
        sqrt_2c = torch.sqrt(2.0 * c_for_newton)
        use_series = c_for_newton < 1e-4
        upper_series = (sqrt_2c - c_for_newton / 3.0).clamp_min(0.0)
        lower_series = -(sqrt_2c + c_for_newton / 3.0)

        # Positive root. expm1 keeps the Newton residual accurate near zero.
        upper_newton = torch.where(c_for_newton < 1.0, sqrt_2c, torch.log1p(c_for_newton) + 0.5)
        for _ in range(4):
            expm1_upper = torch.expm1(upper_newton)
            residual = expm1_upper - upper_newton - c_for_newton
            upper_newton = (upper_newton - residual / expm1_upper.clamp_min(1e-12)).clamp_min(0.0)

        # For large c, exp(z) = c + 1 + z. Fixed-point updates in log-space
        # avoid ever materializing c and converge rapidly because z << c.
        upper_large = log_c.clamp_min(0.0)
        for _ in range(4):
            upper_large = torch.logaddexp(log_c, torch.log1p(upper_large))
        upper = torch.where(log_c > 12.0, upper_large, torch.where(use_series, upper_series, upper_newton))

        # Negative root with u = -z: exp(-u) - 1 + u = c. For large c,
        # u = c + 1 - exp(-u) is indistinguishable from c + 1. Overflow to
        # -inf is intentional: it denotes an inactive lower ratio boundary.
        lower_u = torch.where(c_for_newton < 1.0, sqrt_2c, c_for_newton + 1.0)
        for _ in range(4):
            residual = torch.exp(-lower_u) - 1.0 + lower_u - c_for_newton
            derivative = (-torch.expm1(-lower_u)).clamp_min(1e-12)
            lower_u = (lower_u - residual / derivative).clamp_min(0.0)
        lower_newton = -lower_u
        lower_large = -(torch.exp(log_c) + 1.0)
        lower = torch.where(log_c > 12.0, lower_large, torch.where(use_series, lower_series, lower_newton))

        return lower.to(log_budget.dtype), upper.to(log_budget.dtype)
