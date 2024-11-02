from typing import Callable, List, Optional
import torch
import numpy as np

import math
import torch.utils


def print_and_save_stat(
    text: str, rank: int, stats_file=None, print_in_console: bool = False
):
    if print_in_console and stats_file is not None:
        print(text)
        print(text, file=stats_file)
    elif stats_file is not None:
        print(text, file=stats_file)
    elif rank == 0:
        print(text)


class CosineLRSchedulerWithWarmup:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        len_loader: int,
        warmup_epochs: int,
        end_lrs: List[int] | None = None,
        start_epoch: int = 0,
        batch_size: int | None = None,
    ):
        self.optimizer = optimizer
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        if end_lrs is None:
            self.end_lrs = [base_lr * 0.001 for base_lr in self.base_lrs]
        else:
            self.end_lrs = end_lrs

        self.lrs = self.base_lrs
        if batch_size is not None:
            self.lrs = [(batch_size / 256) * base_lr for base_lr in self.base_lrs]

        self.total_steps = n_epochs * len_loader
        self.warmup_steps = warmup_epochs * len_loader

        self.tr_step = 1 + start_epoch * len_loader

    def step(self, return_lr=False):
        if self.tr_step < self.warmup_steps:
            lrs = [lr * self.tr_step / self.warmup_steps for lr in self.lrs]
        else:
            q = 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (self.tr_step - self.warmup_steps)
                    / (self.total_steps - self.warmup_steps)
                )
            )
            lrs = [
                lr * q + end_lr * (1 - q) for lr, end_lr in zip(self.lrs, self.end_lrs)
            ]
        for idx, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lrs[idx]

        self.tr_step += 1

        if return_lr:
            return lrs


class PretextTrainerCuda:
    """
    Trainer class used for conventional pretext training / pretext warmup of BiSSL.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

    def __call__(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        rank: int,
    ):
        loss_avg = 0

        log_dicts = []

        self.model.train()

        for step, ((x1, x2), _) in enumerate(dataloader, 1):
            x1, x2 = (
                x1.to(self.device, non_blocking=True),
                x2.to(self.device, non_blocking=True),
            )

            loss = self.model((x1, x2))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            loss_avg += loss.item()
            del loss
            self.optimizer.step()

            if rank == 0 and step % (len(dataloader) // 10) == 0:
                print(
                    f"Avg loss over batch no. {step + 1 - (len(dataloader) // 10)}-{step} / {len(dataloader)}: {loss_avg/(len(dataloader) // 10):>5f}"
                )
                pg = self.optimizer.param_groups
                log_dict = {
                    "train_pretext/loss_avg": loss_avg / (len(dataloader) // 10),
                    "train_pretext/lr_backbone": pg[0]["lr"],
                    "train_pretext/epoch": epoch,
                }
                log_dicts.append(log_dict)

                loss_avg = 0

            self.lr_scheduler.step()

        return log_dicts


def train_d_classic(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    rank: int,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
):
    loss_avg = 0
    model.train()

    if len(dataloader) > 1:
        batchnum_lossavg = len(dataloader) // 2
    else:
        batchnum_lossavg = 1

    for batch, (orig, y) in enumerate(dataloader):
        orig, y = (
            orig.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
        )

        loss = loss_fn(model(orig), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg += loss_fn(model(orig), y).item()

        if rank == 0 and (batch + 1) % batchnum_lossavg == 0:
            print(
                f"Avg loss over batch no. {batch + 2 - batchnum_lossavg}-{batch+1} / {len(dataloader)}: {loss_avg/batchnum_lossavg:>5f}"
            )

            loss_avg = 0
        if lr_scheduler is not None:
            lr_scheduler.step()


def test_d(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
    rank: int,
    label: str = "Test Error",
    non_blocking: bool = True,
):
    model.eval()
    test_loss = 0

    num_batches = 0

    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=non_blocking), y.to(
                device, non_blocking=non_blocking
            )

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            acc1, acc5 = accuracy(pred, y, topk=(1, 5))
            top1.update(acc1[0].item(), X.size(0))
            top5.update(acc5[0].item(), X.size(0))

            num_batches += 1
    test_loss /= num_batches
    if rank == 0:
        print(
            label
            + f": \n Top1 Acc: {(top1.avg):>0.2f}%, Top5 Acc: {(top5.avg):>0.2f}%, Avg loss: {test_loss:>5f} \n"
        )
    return top1.avg, top5.avg, test_loss


def test_d_mAP(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: str,
    label: str = "Test Error",
    rank: Optional[int] = None,
    non_blocking: bool = True,
):
    """Calculates mAP loss as used for training and evaluating on the VOC07 dataset."""
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=non_blocking), y.to(
                device, non_blocking=non_blocking
            )

            # Forward pass to get logits
            logits = model(X)

            all_logits.append(logits)
            all_targets.append(y)

    # Concatenate all results from all batches (ensure that these are all Tensors)
    all_logits = torch.cat(all_logits)  # Now a single Tensor
    all_targets = torch.cat(all_targets)  # Now a single Tensor

    # Convert Tensors to NumPy arrays for easier processing
    all_logits = all_logits.cpu().numpy()
    all_targets = all_targets.cpu().numpy()

    # Compute mAP for each class
    num_classes = all_logits.shape[1]
    aps = []

    for cls in range(num_classes):
        # Get ground truth and logits for this class
        gt = all_targets[:, cls]
        scores = all_logits[:, cls]

        # Check if there are any positive samples for this class
        if np.sum(gt) == 0:
            # No positive samples in ground truth, skip this class or handle as needed
            print(
                f"Warning: No positive samples for class {cls}. Skipping mAP calculation for this class."
            )
            continue

        # Sort by logits directly (higher logits indicate higher confidence in that class)
        sorted_indices = np.argsort(-scores)
        sorted_gt = gt[sorted_indices]

        # Compute true positives and false positives
        tp = np.cumsum(sorted_gt)
        fp = np.cumsum(1 - sorted_gt)

        # Compute precision and recall
        recalls = tp / np.sum(gt)  # Safe now, since we've checked np.sum(gt) > 0
        precisions = tp / (tp + fp)

        # 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            precisions_at_recall = precisions[recalls >= t]
            p = np.max(precisions_at_recall) if precisions_at_recall.size > 0 else 0
            ap += p / 11
        aps.append(ap)

    # Compute the mean Average Precision (mAP)
    mAP = np.mean(aps) if aps else 0  # Handle case where aps list might be empty

    if rank == 0:
        print(f"{label}: \n mAP: {mAP:.4f} \n")

    return 100 * mAP


class AverageMeter(object):
    """Computes and stores the average and current value

    # This class is reused from the VICReg implementation
    # found at https://github.com/facebookresearch/vicreg/

    # Copyright (c) Meta Platforms, Inc. and affiliates.

    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class MAPMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
