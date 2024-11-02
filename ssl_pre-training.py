import json
import os
import sys
from argparse import ArgumentParser, Namespace

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

import utils.augmentations as aug

from utils.datasets import GetData
from utils.models import SimCLR
from utils.optimizers import get_optimizer
from utils.parsers import get_args_pretext
from utils.training_fn import (
    CosineLRSchedulerWithWarmup,
    PretextTrainerCuda,
    print_and_save_stat,
)
from utils.distributed import init_distributed_mode

if __name__ == "__main__":
    parser: ArgumentParser = get_args_pretext()

    args: Namespace = parser.parse_args()

    ### Distributed Setup ###
    torch.backends.cudnn.benchmark = True
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["OMP_NUM_THREADS"] = args.omp_num_threads
    args.world_size = torch.cuda.device_count()
    init_distributed_mode(args)

    device = torch.device(args.device)

    print_and_save_stat(text="(Classic Pretext) " + " ".join(sys.argv), rank=args.rank)

    print_and_save_stat(
        text=json.dumps(dict(train_type="Console Args", data=" ".join(sys.argv))),
        print_in_console=False,
        rank=args.rank,
    )

    print("")
    print(f"Device = {device}")

    #### DATA IMPORT ####
    get_data = GetData(args.data_dir, args.download_dataset)
    interpolation = InterpolationMode.BICUBIC

    data = get_data(
        dset_name=args.dset,
        split="unlabeled" if args.dset == "stl10" else "train",
        transform=aug.TrainTransform(
            img_size=args.img_size,
            interpolation_mode=interpolation,
            min_ratio=args.img_crop_min_ratio,
        ),
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        data, shuffle=True, drop_last=True
    )

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    dataloader = DataLoader(
        data,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
    )

    ### MODEL IMPORT ###

    model = SimCLR(
        args.mlp,
        args.arch,
        temp=args.temperature,
        distributed=True,
        proj_use_bn=args.proj_use_bn,
    )

    print(
        f"Backbone: {sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1_000_000} M Parameters"
    )
    print(
        f"Pretext Head: {sum(p.numel() for p in model.projector.parameters() if p.requires_grad)/1_000_000} M Parameters"
    )

    mod_local_path_prefix = (
        args.model_dir + "pretext_" + "arch-" + args.arch + f"_epochs{args.epochs}"
    )

    mod_local_path_bb = mod_local_path_prefix + "_bb.pth"
    mod_local_path_proj = mod_local_path_prefix + "_proj.pth"

    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model)

    ### TRAINING SETUP ###

    optimizer = get_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        wd=args.wd,
        momentum=args.momentum,
    )

    lr_scheduler = CosineLRSchedulerWithWarmup(
        optimizer=optimizer,
        n_epochs=args.epochs,
        len_loader=len(dataloader),
        warmup_epochs=10,
    )

    train_p = PretextTrainerCuda(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
    )

    for epoch in range(args.epochs):
        print_and_save_stat("", rank=args.rank)
        print_and_save_stat(
            f"Epoch {epoch+1} / {args.epochs}\n-------------------------------",
            rank=args.rank,
        )

        train_p(dataloader, epoch, rank=args.rank)

        if args.rank == 0 and args.save_model:
            model.eval()
            sd_bb = model.module.backbone.state_dict()
            sd_proj = model.module.projector.state_dict()

            torch.save(
                sd_bb,
                mod_local_path_bb,
            )
            torch.save(
                sd_proj,
                mod_local_path_proj,
            )
            print_and_save_stat("Saved Model.", rank=args.rank)

        print_and_save_stat("", rank=args.rank)
