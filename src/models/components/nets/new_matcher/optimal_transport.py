import torch


def _log_sinkhorn(
    Z: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    its_count: int
) -> torch.Tensor:
    f, g = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(its_count):
        f = log_a - (Z + g[:, None, :]).logsumexp(2)
        g = log_b - (Z + f[:, :, None]).logsumexp(1)
    return Z + f[:, :, None] + g[:, None, :]


def log_optimal_transport(
    scores: torch.Tensor,
    alpha: torch.Tensor,
    its_count: int
) -> torch.Tensor:
    n, l, s = scores.shape

    Z = torch.cat([scores, alpha.expand(n, l, 1)], dim=2)
    Z = torch.cat([Z, alpha.expand(n, 1, s + 1)], dim=1)

    l_, s_ = scores.new_tensor([l]), scores.new_tensor([s])
    norm = -(l_ + s_).log()
    log_a = torch.cat([norm.expand(l), s_.log() + norm]).expand(n, -1)
    log_b = torch.cat([norm.expand(s), l_.log() + norm]).expand(n, -1)

    out = _log_sinkhorn(Z, log_a, log_b, its_count)
    out -= norm
    return out
