import os
import sys
import json
import argparse
from functools import partial
import random

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from ray import tune, init
from ray.tune import TuneConfig
from ray.air import session

from utils.models import DSClassifier

import utils.augmentations as aug
from utils.distributed import setup_for_distributed

from utils.parsers import get_args_ft
from utils.training_fn import (
    train_d_classic,
    test_d,
    test_d_mAP,
    print_and_save_stat,
    CosineLRSchedulerWithWarmup,
)
from utils.datasets import GetData


def train(config, args, pretrained_model_bb):
    device = torch.device(args.device)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()

    # Disables all printing, as the Ray Tune HPO module will print plenty :)
    setup_for_distributed(False)

    # HPO Config Import
    for hpar, val in config.items():
        vars(args)[hpar] = val

    if "seed" in config.keys():
        random.seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])

    match args.backbone_origin:
        case "pretext":
            print_and_save_stat(
                "(Post Classic SSL Pre-Traing Fine-Tuning) " + " ".join(sys.argv),
                rank=args.rank,
            )
        case "bissl":
            print_and_save_stat(
                "(Post BiSSL Fine-Tuning) " + " ".join(sys.argv), rank=args.rank
            )
        case _:
            raise ValueError("Invalid backbone origin")

    print_and_save_stat("(Post BiSSL) " + " ".join(sys.argv), rank=args.rank)
    print_and_save_stat(
        json.dumps(dict(train_type="Console Args", data=" ".join(sys.argv))),
        print_in_console=False,
        rank=args.rank,
    )

    print("")
    print("Device = " + args.device)

    #### DATA IMPORT ####
    get_data = GetData(
        root=args.data_dir,
        download=args.download_dataset,
    )
    interpolation = InterpolationMode.BICUBIC

    data_train = get_data(
        dset_name=args.dset,
        transform=aug.BLODownstreamTrainTransform(
            img_size=args.pretrain_img_size,
            interpolation_mode=interpolation,
            min_ratio=args.pretrain_img_crop_min_ratio,
        ),
        split="train",
    )

    data_test = get_data(
        dset_name=args.dset,
        split="val",
        transform=aug.BLODownstreamTestTransform(
            img_size=args.pretrain_img_size, interpolation_mode=interpolation
        ),
    )

    # Create data loaders.
    # For training
    dataloader_train = DataLoader(
        data_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )

    # For testing
    dataloader_test = DataLoader(
        data_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ### MODEL IMPORT ###
    model = DSClassifier(
        args.pretrain_arch,
        n_classes=len(data_test.classes),
    )

    model.backbone.load_state_dict(torch.load(pretrained_model_bb, map_location=device))

    print(
        f"Backbone: {sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)/1_000_000} M Parameters"
    )
    print(
        f"Downstream Head: {sum(p.numel() for p in model.head.parameters() if p.requires_grad)/1_000_000} M Parameters"
    )

    ### TRAINING SETUP ###
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        weight_decay=args.wd,
        momentum=args.optimizer_momentum,
    )

    lr_scheduler = CosineLRSchedulerWithWarmup(
        optimizer=optimizer,
        n_epochs=args.epochs,
        len_loader=len(dataloader_train),
        warmup_epochs=0,
    )

    mod_local_path_prefix = (
        args.model_dir
        + f"post-{args.backbone_origin}_downstream_arch-"
        + args.pretrain_arch
    )

    mod_local_path_bb = (
        mod_local_path_prefix + f"_dset-{args.dset}_bb_seed{args.seed}.pth"
    )
    mod_local_path_h = (
        mod_local_path_prefix + f"_dset-{args.dset}_h_seed{args.seed}.pth"
    )

    best_acc_te = argparse.Namespace(top1=0, top5=0)

    model.cuda(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model)

    ### TRAINING ###
    for epoch in range(int(args.epochs)):
        print_and_save_stat("", rank=args.rank)
        print_and_save_stat(
            f"Epoch {epoch+1} / {args.epochs}\n-------------------------------",
            rank=args.rank,
        )

        # Adapt backbone from pretext into downstream model
        train_d_classic(
            model=model,
            loss_fn=loss_fn,
            dataloader=dataloader_train,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            rank=args.rank,
        )

        print_and_save_stat("", rank=args.rank)

        if args.dset == "voc2007":
            top1_te = test_d_mAP(
                dataloader_test,
                model,
                device,
                "Test Error (mAP)",
                args.rank,
            )

            loss_te = 0
            top5_te = 0
        else:
            top1_te, top5_te, loss_te = test_d(
                dataloader_test,
                model,
                loss_fn,
                device,
                label="Test Error",
                rank=args.rank,
            )

        best_acc_top1_prev = best_acc_te.top1
        best_acc_te.top1 = max(best_acc_te.top1, top1_te)
        best_acc_te.top5 = max(best_acc_te.top5, top5_te)

        session.report({"loss": loss_te, "accuracy": top1_te})

        if (
            args.rank == 0
            and args.save_model
            and (best_acc_te.top1 - best_acc_top1_prev) > 0
        ):
            model.eval()
            torch.save(
                model.module.backbone.state_dict(),
                mod_local_path_bb,
            )
            torch.save(
                model.module.head.state_dict(),
                mod_local_path_h,
            )

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = get_args_ft()
    args = parser.parse_args()

    # Distributed Setup
    torch.backends.cudnn.benchmark = True
    os.environ["NCCL_P2P_DISABLE"] = "1"

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ["RANK"] = str(args.rank)
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["WORLD_SIZE"] = str(args.world_size)
    else:
        print("Not using distributed mode")

    # Disable ray tune from logging data
    # Will raise a warning after training has ended, but it is safe to ignore.
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    if args.rank == 0:
        # HPO Setup
        # Either conduct hyperparamter search, or conduct runs all with the same hyperparameters and varying seeds.
        if args.use_hpo:
            config = {
                "lr": tune.qloguniform(
                    args.hpo_lr_min, args.hpo_lr_max, args.hpo_lr_min
                ),
                "wd": tune.qloguniform(
                    args.hpo_wd_min, args.hpo_wd_max, args.hpo_wd_min
                ),
                "seed": tune.uniform(0, 1),
            }
        else:
            config = {
                "lr": tune.choice([args.lr]),
                "wd": tune.choice([args.wd]),
                "seed": tune.uniform(0, 1),
            }

        # We assign 1 gpu and 10 cpus pr task. This can be adjusted if needed.
        ngpus_pr_task = 1
        ncpus_pr_task = 10

        # Only confirmed to work for ngpus_pr_task = 1
        args.world_size = ngpus_pr_task
        init(
            num_cpus=ncpus_pr_task * torch.cuda.device_count(),
            num_gpus=torch.cuda.device_count(),
            include_dashboard=False,
        )
        trainable_with_resources = tune.with_resources(
            partial(
                train,
                args=args,
                pretrained_model_bb=args.model_dir + args.pretrained_backbone_filename,
            ),
            {"cpu": ncpus_pr_task, "gpu": ngpus_pr_task},
        )

        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=config,
            tune_config=TuneConfig(num_samples=args.num_runs),
        )
        results = tuner.fit()
