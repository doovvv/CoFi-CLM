"""CoFi-CLM training objective: reconstruction, token cross-entropy, and coarse VQ terms."""

import torch
import torch.nn.functional as F


def log_nmse(target, reconstruction, center=0.5):
    target, reconstruction = target - center, reconstruction - center
    error = ((target - reconstruction) ** 2).flatten(1).sum(1)
    power = (target**2).flatten(1).sum(1)
    return torch.log1p(error / (power + 1e-8)).mean()


def clm_loss(output, weights):
    target = output["target"]
    losses = {}
    for key, prefix in [
        ("reconstruction", ""),
        ("coarse_reconstruction", "coarse_"),
        ("lm_reconstruction", "lm_"),
    ]:
        name = "reconstruction" if prefix == "" else prefix + "mse"
        losses[name] = F.mse_loss(output[key], target)
        losses[prefix + "log_nmse"] = log_nmse(target, output[key])
    losses["token_ce"] = F.cross_entropy(
        output["token_logits"].flatten(0, 1), output["fine_indices"].flatten()
    )
    for key in ("vq", "diversity", "variance"):
        losses[key] = output[key]
    losses["total"] = (
        losses["reconstruction"]
        + weights["nmse"] * losses["log_nmse"]
        + weights["coarse"] * (losses["coarse_mse"] + weights["nmse"] * losses["coarse_log_nmse"])
        + weights["lm"] * (losses["lm_mse"] + weights["nmse"] * losses["lm_log_nmse"])
        + weights["token_ce"] * losses["token_ce"]
        + weights["vq"] * losses["vq"]
        + weights["diversity"] * losses["diversity"]
        + weights["variance"] * losses["variance"]
    )
    return losses
