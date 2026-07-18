from __future__ import annotations

import torch


def cox_ph_loss(survival: torch.Tensor, risk: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Negative Cox partial log-likelihood with Efron handling of tied event times."""
    if survival.ndim != 2 or survival.shape[1] != 2:
        raise ValueError(f"survival must have shape [N, 2], received {tuple(survival.shape)}")
    risk = risk.reshape(-1)
    if risk.shape[0] != survival.shape[0]:
        raise ValueError("risk and survival must contain the same number of samples.")

    time = survival[:, 0]
    event = survival[:, 1].bool()
    order = torch.argsort(time, descending=True)
    time, event, risk = time[order], event[order], risk[order]
    if event.sum() == 0:
        return risk.sum() * 0.0

    # Subtracting a constant leaves the Cox likelihood unchanged and prevents overflow.
    centered_risk = risk - risk.max().detach()
    exp_risk = torch.exp(centered_risk)
    cumulative_risk = torch.cumsum(exp_risk, dim=0)
    event_indices = torch.nonzero(event, as_tuple=False).flatten()
    event_times = time[event]

    log_likelihood = risk.new_zeros(())
    event_count = 0
    for event_time in torch.unique(event_times):
        tied_indices = event_indices[event_times == event_time]
        tied_count = int(tied_indices.numel())
        event_count += tied_count
        risk_set_end = torch.nonzero(time >= event_time, as_tuple=False).flatten()[-1]
        risk_sum = cumulative_risk[risk_set_end]
        tied_exp_sum = exp_risk[tied_indices].sum()
        fractions = torch.arange(tied_count, device=risk.device, dtype=risk.dtype) / tied_count
        denominators = torch.clamp(risk_sum - fractions * tied_exp_sum, min=eps)
        log_likelihood += centered_risk[tied_indices].sum() - torch.log(denominators).sum()

    return -log_likelihood / event_count
