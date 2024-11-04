from typing import Any, Callable, Dict, List, Tuple, Literal

import torch
import torch.distributed
import torch.utils
from utils.training_fn import print_and_save_stat

LOWER_INPUTS_MAX = 5


class EmptyLrSched:
    def __init__(self): ...

    def step(self, return_lr=False): ...


def get_par_req_grad(model: torch.nn.Module):
    return iter([par for par in model.parameters() if par.requires_grad is True])


class IGGradCalc:
    def __init__(
        self,
        solver: Callable,
        lam_dampening: float = 0.0,
        solver_kwargs: Dict[Any, Any] | None = None,
    ):
        self.lam_dampening = lam_dampening

        # The solver algorithm to approximate the inverse hessian vector product
        # The current implementation only supports the Conjugate Gradient method (see the cg_solver function)
        self.solver = solver

        if solver_kwargs is not None:
            self.solver_kwargs_dict = dict(solver_kwargs)
        else:
            self.solver_kwargs_dict = {}

        self.solver_type = "cg"

    def __call__(
        self,
        lower_inputs: Tuple[torch.Tensor],
        lower_criteria: Callable,
        lower_backbone: torch.nn.Module,
        vecs: Tuple[torch.Tensor],
        lambd: float | Tuple[float] | List[float],
        lam_dampening: None | float = None,
    ) -> Tuple[torch.Tensor]:

        if lam_dampening is not None:
            self.lam_dampening = lam_dampening

        # Function which calculates hessian vector product, and returns this product along
        # with the other term, such that the output is the inverse IG vector product
        def hpid_v_p(vs: torch.Tensor) -> torch.Tensor:
            hvps = tuple([torch.zeros_like(vec) for vec in vecs])
            # Calculates first order gradients
            for lower_input in lower_inputs:
                # loss = lower_model(lower_input)
                grads: Tuple[torch.Tensor] = iter(
                    torch.autograd.grad(
                        lower_criteria(lower_input),
                        lower_backbone.parameters(),
                        create_graph=True,
                    )
                )

                # prod is a tuple of scalar tensors
                prods: Tuple[torch.Tensor] = iter(
                    [grad.mul(v).sum() for grad, v in zip(grads, vs)]
                )

                hvp_add: Tuple[torch.Tensor] = iter(
                    torch.autograd.grad(prods, lower_backbone.parameters())
                )
                hvps = tuple(
                    hvp.add(hvp_add, alpha=1 / len(lower_inputs))
                    for hvp, hvp_add in zip(hvps, hvp_add)
                )

            return tuple(
                [
                    vmul + hvp.div_(lambd + self.lam_dampening)
                    for vmul, hvp in zip(vs, hvps)
                ]
            )

        return self.solver(hpid_v_p, vecs, **self.solver_kwargs_dict)


def cg_solver(
    mvp_fn: Callable,
    vecs: Tuple[torch.Tensor],
    iter_num: int = 10,
    x_init: Tuple[torch.Tensor] | None = None,
    verbose: bool = False,
    layerwise_solve=True,
) -> Tuple[torch.Tensor]:
    """
    Conjugate gradient solver, based on the iMaml implementation:
    https://github.com/aravindr93/imaml_dev/blob/master/implicit_maml/utils.py
    """
    if x_init is not None:
        xs = x_init
        rs = tuple([vec - Hx0 for vec, Hx0 in zip(vecs, mvp_fn(x_init))])
    else:
        xs = tuple([torch.zeros_like(vec) for vec in vecs])
        rs = tuple([vec for vec in vecs])

    for v in vecs:
        if v.dtype == torch.float16:
            xs = tuple([x.half() for x in xs])
    del vecs

    ds = tuple([r.clone() for r in rs])

    if layerwise_solve:
        # Initialising norm(r), avoiding the neccesity for calculating this value twice in a loop
        new_rdotrs = tuple([torch.sum(r**2) for r in rs])

        for _ in range(iter_num):
            Hds = mvp_fn(ds)

            rdotrs = tuple([r_copy for r_copy in new_rdotrs])

            alphas = tuple(
                [
                    rdotr / (torch.sum(d * Hd) + 1e-12)
                    for rdotr, d, Hd in zip(rdotrs, ds, Hds)
                ]
            )

            xs = tuple([x + alpha * d for x, d, alpha in zip(xs, ds, alphas)])
            rs = tuple([r - alpha * Hd for r, Hd, alpha in zip(rs, Hds, alphas)])

            new_rdotrs = tuple([torch.sum(r**2) for r in rs])
            betas = tuple(
                [
                    new_rdotr / (rdotr + 1e-12)
                    for new_rdotr, rdotr in zip(new_rdotrs, rdotrs)
                ]
            )

            ds = tuple([r + beta * d for r, d, beta in zip(rs, ds, betas)])

            if verbose:
                print(new_rdotrs)
                print("")

        del mvp_fn, rs, ds, rdotrs, Hds, alphas, new_rdotrs, betas
    else:
        # Initialising norm(r), avoiding the neccesity for calculating this value twice in a loop
        new_rdotr = sum([torch.sum(r**2) for r in rs])

        for _ in range(iter_num):
            rdotr = new_rdotr

            Hds = mvp_fn(ds)

            alpha = sum([rdotr / (torch.sum(d * Hd) + 1e-12) for d, Hd in zip(ds, Hds)])

            xs = tuple([x + alpha * d for x, d in zip(xs, ds)])
            rs = tuple([r - alpha * Hd for r, Hd in zip(rs, Hds)])

            new_rdotr = sum([torch.sum(r**2) for r in rs])
            beta = new_rdotr / (rdotr + 1e-12)

            ds = tuple([r + beta * d for r, d in zip(rs, ds)])

            if verbose:
                print(new_rdotr)
                print("")

        del mvp_fn, rs, ds, rdotr, Hds, alpha, new_rdotr, beta

    return tuple([torch.nan_to_num(x) for x in xs])


class BiSSL_Trainer:
    def __init__(
        self,
        optimizers: Tuple[torch.optim.Optimizer],
        models: Tuple[torch.nn.Module],
        loss_fn_d: torch.optim.Optimizer,
        device: torch.device,
        ig_grad_calc: IGGradCalc,
        rank: int | None = None,
        lr_scheduler_p: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_d: torch.optim.lr_scheduler.LRScheduler | None = None,
        lower_num_iter: int = 20,
        upper_num_iter: int = 8,
        lower_iter_type: Literal["steps", "epochs"] = "steps",
        upper_iter_type: Literal["steps", "epochs"] = "steps",
    ):
        self.optimizer_d, self.optimizer_p = optimizers
        self.model_d, self.model_p = models

        self.loss_fn_d = loss_fn_d

        self.device = device

        self.ig_grad_calc = ig_grad_calc

        if lr_scheduler_p is None:
            self.lr_scheduler_p = EmptyLrSched()
        else:
            self.lr_scheduler_p = lr_scheduler_p

        if lr_scheduler_d is None:
            self.lr_scheduler_d = EmptyLrSched()
        else:
            self.lr_scheduler_d = lr_scheduler_d

        self.lower_num_iter = lower_num_iter
        self.lower_iter_type = lower_iter_type
        self.upper_num_iter = upper_num_iter
        self.upper_iter_type = upper_iter_type

        if rank is not None:
            self.rank = rank
        else:
            self.rank = 0

        self.dl_iter_u, self.dl_iter_l = None, None
        self.step_u, self.step_l = None, None

        self.sampler_it = 0

    def _update_upper_grad(
        self,
        grads: Tuple[torch.Tensor],
        model: torch.nn.Module,
    ):
        pars_requires_grad = get_par_req_grad(model)

        for p, grad in zip(pars_requires_grad, grads):
            if p.grad is None:
                p.grad = grad

            p.grad.copy_(grad)

    def regularization_loss(
        self,
        theta_d: Tuple[torch.nn.Parameter],
        model_p_pars: Tuple[torch.nn.Parameter],
        lambd: float = 0.0,
    ):
        """
        Calculates the regularization loss for the lower-level loss calculation.
        """

        deltas = tuple(
            [
                par_p - par_d.clone().detach()
                for par_p, par_d in zip(model_p_pars, theta_d)
            ]
        )

        return 0.5 * lambd * sum([torch.sum(delta**2) for delta in deltas])

    def _get_upper_grads(
        self,
        upper_input: Tuple[torch.Tensor, torch.Tensor],
        lower_inputs: List[Tuple[torch.Tensor, torch.Tensor]],
        lambd: float = 1.0,
        cg_lam_dampening: float | None = None,
    ) -> Tuple[Tuple[torch.Tensor]]:

        loss_uhead_lbackbone = self.loss_fn_d(
            self.model_d.module.head(self.model_p.module.backbone(upper_input[0])),
            upper_input[1],
        )

        grads_head = torch.autograd.grad(
            loss_uhead_lbackbone,
            self.model_d.module.head.parameters(),
            retain_graph=True,
        )

        grads_backbone = torch.autograd.grad(
            loss_uhead_lbackbone, self.model_p.module.backbone.parameters()
        )

        if self.ig_grad_calc.solver_type == "cg":
            ig_grads_backbone = self.ig_grad_calc(
                lower_inputs=lower_inputs,
                lower_criteria=self.model_p,
                lower_backbone=self.model_p.module.backbone,
                vecs=grads_backbone,
                lambd=lambd,
                lam_dampening=cg_lam_dampening,
            )

        else:
            raise KeyError

        del loss_uhead_lbackbone

        return iter(ig_grads_backbone), iter(grads_head)

    def _upper_train_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        lambd: float,
        upper_classic_grad_scale: float,
        epoch: int,
        dl_len: int,
        lower_inputs: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        loss_d_avg = 0
        self.model_d.train()
        self.model_p.train()

        for orig, y in dataloader:
            orig, y = (
                orig.to(self.device, non_blocking=True),  # .double(),
                y.to(self.device, non_blocking=True),
            )

            self.optimizer_d.zero_grad(set_to_none=True)
            self.optimizer_p.zero_grad(set_to_none=True)

            grads_d_backbone, grads_d_head = self._get_upper_grads(
                upper_input=(orig, y),
                lower_inputs=lower_inputs,
                lambd=lambd,
            )

            # Calculates the "classic" downstream gradient if gamma (called upper_classic_grad_scale here) != 0
            if upper_classic_grad_scale != 0:
                loss_d_classic = self.loss_fn_d(self.model_d(orig), y)
                grads_d_classic_bb = iter(
                    torch.autograd.grad(
                        loss_d_classic,
                        self.model_d.module.backbone.parameters(),
                        retain_graph=True,
                    )
                )

                grads_d_classic_head = iter(
                    torch.autograd.grad(
                        loss_d_classic,
                        self.model_d.module.head.parameters(),
                    )
                )
                del loss_d_classic

                grads_d_backbone = iter(
                    [
                        grad_d.add(grad_d_cl, alpha=upper_classic_grad_scale)
                        for grad_d, grad_d_cl in zip(
                            grads_d_backbone, grads_d_classic_bb
                        )
                    ]
                )

                grads_d_head = iter(
                    [
                        grad_ig.add(grad_classic, alpha=upper_classic_grad_scale)
                        for grad_ig, grad_classic in zip(
                            grads_d_head, grads_d_classic_head
                        )
                    ]
                )

            self._update_upper_grad(
                grads=grads_d_backbone,
                model=self.model_d.module.backbone,
            )
            self._update_upper_grad(
                grads=grads_d_head,
                model=self.model_d.module.head,
            )

            torch.nn.utils.clip_grad_norm_(self.model_d.parameters(), 10.0)

            self.optimizer_d.step()

            self.model_d.eval()
            loss_d_avg += self.loss_fn_d(self.model_d(orig), y).item()
            self.model_d.train()

            self.step_u += 1

            self.lr_scheduler_d.step()

            if self.step_u % self.upper_num_iter == 0:
                # loss_d_avg *= epoch + 1
                string = (
                    f"Avg loss over batch no."
                    + f" {self.step_u + 1 - self.upper_num_iter}-{self.step_u}"
                    + f" / {dl_len}: {loss_d_avg/self.upper_num_iter:>5f}"
                )
                if self.rank == 0:
                    print_and_save_stat(string, rank=self.rank)

                self.optimizer_d.zero_grad(set_to_none=True)

                loss_d_avg = 0

                break

        print_and_save_stat("", rank=self.rank)

    def _lower_train_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        dl_len: int,
        epoch: int,
        lambd: float = 0.0,
    ):
        loss_p_avg = 0

        self.model_p.train()

        if self.ig_grad_calc.solver_type == "cg":
            lower_input_return = []
        else:
            lower_input_return = None

        for (x1, x2), _ in dataloader:
            x1, x2 = (
                x1.to(self.device, non_blocking=True),
                x2.to(self.device, non_blocking=True),
            )

            if self.ig_grad_calc.solver_type == "cg":
                # Stores the LOWER_INPUT_MAX (=5 in the paper) first lower level inputs to use for approximating the lower-level hessian in the upper-level gradient approximation.
                if len(lower_input_return) < LOWER_INPUTS_MAX:
                    lower_input_return.append((x1, x2))

            loss_p = self.model_p((x1, x2))
            loss_p_avg += loss_p.item()
            if lambd != 0:
                loss_p += self.regularization_loss(
                    theta_d=self.model_d.module.backbone.parameters(),
                    lambd=lambd,
                    model_p_pars=self.model_p.module.backbone.parameters(),
                ).to(self.device)

            self.optimizer_p.zero_grad(set_to_none=True)

            loss_p.backward()

            torch.nn.utils.clip_grad_norm_(self.model_p.parameters(), 10.0)

            self.optimizer_p.step()

            del loss_p

            self.step_l += 1

            self.lr_scheduler_p.step()

            if self.step_l % self.lower_num_iter == 0:

                string = f"Avg loss over batch no. {self.step_l + 1 - self.lower_num_iter}-{self.step_l} / {dl_len}: {loss_p_avg/self.lower_num_iter:>5f}"

                if self.rank == 0:
                    print_and_save_stat(string, rank=self.rank)

                loss_p_avg = 0

                break

            self.optimizer_p.zero_grad(set_to_none=True)

        return lower_input_return

    def __call__(
        self,
        dataloaders: Tuple[torch.utils.data.DataLoader],
        samplers: Tuple[torch.utils.data.Sampler],
        epoch: int,
        lambd: float = 1.0,
        upper_classic_grad_scale: float = 0.0,
    ):
        dataloader_u, dataloader_l = dataloaders
        sampler_u, sampler_l = samplers

        # Resets dataloaders if remaining number of samples are smaller than the number of
        # gradient steps to be made in the respective levels.
        if self.step_l is None or len(dataloader_l) - self.step_l < self.lower_num_iter:
            if sampler_l is not None:
                sampler_l.set_epoch(epoch)
                self.sampler_it += 1

            self.dl_iter_l = iter(dataloader_l)
            self.step_l = 0

        if self.step_u is None or len(dataloader_u) - self.step_u < self.upper_num_iter:
            if sampler_u is not None:
                sampler_u.set_epoch(epoch)
                self.sampler_it += 1

            self.dl_iter_u = iter(dataloader_u)
            self.step_u = 0

        self.model_d.train()
        self.model_p.train()

        #########################
        ###### Lower Level ######
        #########################
        if self.rank == 0:
            print(
                "Lower Level...",
            )

        if self.lower_num_iter > 0:
            lower_input = self._lower_train_loop(
                dataloader=self.dl_iter_l,
                dl_len=len(dataloader_l),
                epoch=epoch,
                lambd=lambd,
            )

        print_and_save_stat("", rank=self.rank)

        ##########################
        ####### Upper Level ######
        ##########################
        print_and_save_stat("Upper Level...", rank=self.rank)

        for inp, _ in dataloader_l:
            lower_input = [
                (
                    inp[0].to(self.device, non_blocking=True),
                    inp[1].to(self.device, non_blocking=True),
                )
            ]
            break

        self._upper_train_loop(
            dataloader=self.dl_iter_u,
            lambd=lambd,
            upper_classic_grad_scale=upper_classic_grad_scale,
            epoch=epoch,
            dl_len=len(dataloader_u),
            lower_inputs=lower_input,
        )
