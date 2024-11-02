import torch


class LARS(torch.optim.Optimizer):
    """Based on the VICReg implementation of LARS, found in https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

    # Copyright (c) Meta Platforms, Inc. and affiliates.

    """

    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay"] != 0 and (
                    g["weight_decay_filter"] is None or not g["weight_decay_filter"](p)
                ):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0,
                            (g["eta"] * param_norm / update_norm),
                            one,
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def get_optimizer(optim, params, lr, wd=0, momentum=0):
    match optim:
        case "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
        case "lars":
            return LARS(params, lr=lr, weight_decay=wd, momentum=momentum)
        case _:
            raise ValueError("Invalid optimizer choice")
