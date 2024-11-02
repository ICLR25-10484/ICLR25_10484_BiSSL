import argparse
from config_default import (
    ArgsGeneralDefaults,
    ArgsPretextDefaults,
    ArgsBiSSLDefaults,
    ArgsFineTuningDefaults,
)

binary_choices = (0, 1)
device_choices = ("cuda",)
model_arch_choices = (
    "resnet18",
    "resnet50",
)
opt_choices = (
    "sgd",
    "lars",
)
d_dset_choices = (
    "stl10",
    "food",
    "fashionmnist",
    "cars",
    "dtd",
    "pets",
    "flowers",
    "aircrafts",
    "cifar10",
    "cifar100",
    "caltech101",
    "sun397",
    "voc2007",
    "cub2011",
)
backbone_origin_choices = ("pretext", "bissl")


def add_general_args(parser: argparse.ArgumentParser):
    args_general = ArgsGeneralDefaults()

    # Dirs and Data
    parser.add_argument(
        "--data-dir",
        default=args_general.data_dir,
        help="Dir to datasets.",
    )
    parser.add_argument(
        "--model-dir",
        default=args_general.model_dir,
        help="Dir to stored models.",
    )

    parser.add_argument(
        "--download-dataset",
        default=args_general.download_dataset,
        type=int,
        choices=binary_choices,
        help="Allow download of dataset if not present in storage",
    )

    # Device Setup
    parser.add_argument(
        "--device",
        default=args_general.device,
        choices=device_choices,
        type=str,
        help="Device to use for training / testing. Currently only supports cuda",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=args_general.num_workers,
        help="Number of workers",
    )
    parser.add_argument(
        "--world-size",
        default=args_general.world_size,
        type=int,
        help="number of distributed processes",
    )
    parser.add_argument(
        "--dist-url",
        default=args_general.dist_url,
        help="url used to set up distributed training",
    )

    parser.add_argument(
        "--omp-num-threads",
        default=str(ArgsGeneralDefaults.omp_num_threads),
        help="num of cpu threads",
    )


def get_args_pretext():
    args_pretext = ArgsPretextDefaults()

    parser = argparse.ArgumentParser(
        description="Pretext Pre-Training HyperParameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Adds General Arugments
    add_general_args(parser)

    parser.add_argument(
        "--img-size",
        default=args_pretext.img_size,
        type=int,
        help="Input image size",
    )
    parser.add_argument(
        "--img-crop-min-ratio",
        default=args_pretext.img_crop_min_ratio,
        type=float,
        help="Minimum ratio of the image size for the random crop",
    )

    parser.add_argument(
        "--arch",
        default=args_pretext.arch,
        choices=model_arch_choices,
        help="Architecture of the backbone encoder network",
    )
    parser.add_argument(
        "--mlp",
        default=args_pretext.mlp,
        help="Size and number of layers of the MLP expander / projection head",
    )
    parser.add_argument(
        "--proj-use-bn",
        default=args_pretext.proj_use_bn,
        type=int,
        choices=binary_choices,
        help="Use batch norm in the projection head",
    )

    # Training Args
    parser.add_argument(
        "--epochs", default=args_pretext.epochs, type=int, help="Number of epochs"
    )
    parser.add_argument(
        "--dset",
        default=args_pretext.dset,
        choices=d_dset_choices,
        help="Pre-Training Dataset",
    )

    parser.add_argument(
        "--batch-size",
        default=args_pretext.batch_size,  # Effective batch size (per worker batch size is [batch-size] / world-size)
        type=int,
        help="Batch Size",
    )
    parser.add_argument(
        "--lr",
        default=args_pretext.lr,
        type=float,
        help="Base learning rate, effective learning after warmup is [lr] * [batch-size] / 256",
    )
    parser.add_argument(
        "--wd", default=args_pretext.wd, type=float, help="Weight decay"
    )
    parser.add_argument(
        "--momentum",
        default=args_pretext.momentum,
        type=float,
        help="Momentum",
    )

    parser.add_argument(
        "--optimizer",
        default=args_pretext.optimizer,
        choices=opt_choices,
        help="Type of optimizer",
    )

    parser.add_argument(
        "--temperature",
        default=args_pretext.temperature,
        type=float,
        help="Temperature for the softmax in the contrastive NT-XENT loss used with SimCLR",
    )

    parser.add_argument(
        "--save-model",
        default=args_pretext.save_model,
        type=int,
        choices=binary_choices,
        help="Save model to storage.",
    )

    return parser


def get_args_bissl():
    args_bissl = ArgsBiSSLDefaults()

    parser = argparse.ArgumentParser(
        description="BiSSL HyperParameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Adds General Arugments
    add_general_args(parser)

    parser.add_argument(
        "--arch",
        default=args_bissl.arch,
        choices=model_arch_choices,
        help="Architecture of the backbone encoder network",
    )

    parser.add_argument(
        "--img-size",
        default=args_bissl.img_size,
        type=int,
        help="input image size",
    )
    parser.add_argument(
        "--img-crop-min-ratio",
        default=args_bissl.img_crop_min_ratio,
        type=float,
        help="Minimum ratio of the image size for the random crop",
    )

    parser.add_argument(
        "--epochs", default=args_bissl.epochs, type=int, help="Number of epochs"
    )

    ### Downstream Data
    parser.add_argument(
        "--d-batch-size",
        default=args_bissl.d_batch_size,  # Effective batch size (per worker batch size is [batch-size] / world-size)
        type=int,
        help="Downstream Batch Size",
    )
    parser.add_argument(
        "--d-dset",
        default=args_bissl.d_dset,
        choices=d_dset_choices,
        help="choice of downstream dataset",
    )

    ### Downstream Training
    parser.add_argument(
        "--d-lr", default=args_bissl.d_lr, type=float, help="Upper-level learning rate"
    )
    parser.add_argument(
        "--d-wd",
        default=args_bissl.d_wd,
        type=float,
        help="Upper-level Weight decay",
    )
    parser.add_argument(
        "--d-momentum",
        default=args_bissl.d_momentum,
        type=float,
        help="Downstream Optimizer Momentum",
    )
    parser.add_argument(
        "--d-optimizer",
        default=args_bissl.d_optimizer,
        choices=opt_choices,
        help="Optimizer for downstream training",
    )

    parser.add_argument(
        "--d-adaptation-freq",
        default=args_bissl.d_adaptation_freq,
        type=int,
        help="Adapt downstream backbone every specified epoch. Does not adapt if 0.",
    )

    parser.add_argument(
        "--d-adaptation-strength",
        default=args_bissl.d_adaptation_strength,
        type=float,
        help="Upper level adaptation strength.",
    )

    # Linear Warmup Hyperparameters
    parser.add_argument(
        "--d-linear-warmup-epochs",
        default=args_bissl.d_linear_warmup_epochs,
        type=int,
    )

    ### Pretext Data
    parser.add_argument(
        "--p-batch-size",
        default=args_bissl.p_batch_size,  # Effective batch size (per worker batch size is [batch-size] / world-size)
        type=int,
        help="Pretext Batch Size",
    )
    parser.add_argument(
        "--p-dset",
        default=args_bissl.p_dset,
        help="Pretext dataset",
        choices=d_dset_choices,
    )

    ### Pretext Model
    parser.add_argument(
        "--p-mlp",
        default=args_bissl.p_mlp,
        help="Architecture of the mlp projection head of the pretext model",
    )
    parser.add_argument(
        "--p-proj-use-bn",
        default=args_bissl.p_proj_use_bn,
        type=int,
        choices=binary_choices,
        help="Use batch norm in the (lower-level) pretext projection head",
    )
    parser.add_argument(
        "--p-temperature",
        default=args_bissl.p_temperature,
        type=float,
        help="(SimCLR Specific) Temperature for the softmax in the contrastive loss (of pretext model)",
    )

    ### Pretext Training
    parser.add_argument(
        "--p-lr",
        default=args_bissl.p_lr,
        type=float,
        help="Lower-level learning rate",
    )

    parser.add_argument(
        "--p-wd", default=args_bissl.p_wd, type=float, help="Lower-level weight decay"
    )

    parser.add_argument(
        "--p-momentum",
        default=args_bissl.p_momentum,
        type=float,
        help="Pretext Optimizer Momentum",
    )
    parser.add_argument(
        "--p-optimizer",
        default=args_bissl.p_optimizer,
        choices=opt_choices,
        help="Optimizer choice for training pretext model",
    )

    parser.add_argument(
        "--p-pretrained-backbone-filename",
        default=args_bissl.p_pretrained_backbone_filename,
        help="Filename for pre-pretrained backbone model parameters.",
    )
    parser.add_argument(
        "--p-pretrained-proj-filename",
        default=args_bissl.p_pretrained_proj_filename,
        help="Filename for pre-pretrained projection head model parameters.",
    )

    ### Shared

    parser.add_argument(
        "--save-model",
        default=args_bissl.save_model,
        type=int,
        choices=binary_choices,
        help="Save lower-level backbone after training is finished",
    )

    parser.add_argument(
        "--upper-classic-grad-scale",
        default=args_bissl.upper_classic_grad_scale,
        type=float,
        help="lr on the gradients of the ds model wrt ds parameters",
    )

    parser.add_argument(
        "--lower-num-iter",
        default=args_bissl.lower_num_iter,
        type=int,
        help="Lower level number of iterations. Is either epochs or steps depending on the lower-iter-type",
    )
    parser.add_argument(
        "--lower-iter-type",
        default=args_bissl.lower_iter_type,
        type=str,
        choices=["epochs", "steps"],
    )
    parser.add_argument(
        "--upper-num-iter",
        default=args_bissl.upper_num_iter,
        type=int,
        help="Upper level number of iterations. Is either epochs or steps depending on the upper-iter-type",
    )
    parser.add_argument(
        "--upper-iter-type",
        default=args_bissl.upper_iter_type,
        type=str,
        choices=["epochs", "steps"],
    )

    # Cg Solver Args
    parser.add_argument(
        "--cg-lam",
        default=args_bissl.cg_lam,
        type=float,
        help="reg scaler (downstream)",
    )
    parser.add_argument(
        "--cg-lam-dampening",
        default=args_bissl.cg_lam_dampening,
        type=float,
        help="reg scaler dampening",
    )

    parser.add_argument(
        "--cg-iter-num",
        default=args_bissl.cg_solver_kwargs["iter_num"],
        type=int,
        help="max number of iterations for cg solver",
    )

    parser.add_argument(
        "--cg-verbose",
        default=args_bissl.cg_solver_kwargs["verbose"],
        type=int,
        choices=binary_choices,
        help="make cg solver print residual for each iteration (0/1 = n/y)",
    )

    return parser


def get_args_ft():
    args_ft = ArgsFineTuningDefaults()

    parser = argparse.ArgumentParser(
        description="Fine-Tuning HyperParameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Adds General Args
    add_general_args(parser)

    # Pretrain Args
    parser.add_argument(
        "--backbone-origin",
        default=args_ft.backbone_origin,
        choices=backbone_origin_choices,
        help="Specify if the backbone is from pretext pre-training only or from using BiSSL. Only affects model naming and some console outputs.",
    )
    parser.add_argument(
        "--pretrain-arch",
        default=args_ft.pretrain_arch,
        choices=model_arch_choices,
        help="Architecture of the backbone encoder network",
    )

    parser.add_argument(
        "--pretrain-img-size",
        default=args_ft.pretrain_img_size,
        type=int,
        help="input image size",
    )
    parser.add_argument(
        "--pretrain-img-crop-min-ratio",
        default=args_ft.pretrain_img_crop_min_ratio,
        type=float,
        help="Minimum ratio of the image size for the random crop",
    )

    parser.add_argument(
        "--pretrained-backbone-filename",
        default=args_ft.pretrained_backbone_filename,
        type=str,
        help="Name of pretrained backbone (in .pth file format)",
    )

    # Hyperparameter Optimization Args
    parser.add_argument(
        "--use-hpo",
        default=args_ft.use_hpo,
        type=int,
        choices=binary_choices,
        help="Wether or not to conduct a hyperparameter optimization (HPO) over a grid search of learning rates and weight decays. If false, then num_runs number of runs will be conducted, all with the same specified lr and wd.",
    )
    parser.add_argument(
        "--hpo-lr-min",
        default=args_ft.hpo_lr_min,
        type=float,
        help="Minimum learning rate of HPO grid",
    )
    parser.add_argument(
        "--hpo-lr-max",
        default=args_ft.hpo_lr_max,
        type=float,
        help="Maximum learning rate of HPO grid",
    )
    parser.add_argument(
        "--hpo-wd-min",
        default=args_ft.hpo_wd_min,
        type=float,
        help="Minimum weight decay of HPO grid",
    )
    parser.add_argument(
        "--hpo-wd-max",
        default=args_ft.hpo_wd_max,
        type=float,
        help="Maximum weight decay of HPO grid",
    )

    ### General Training
    parser.add_argument(
        "--num-runs",
        default=args_ft.num_runs,
        type=int,
        help="Total number of training runs to conduct.",
    )
    parser.add_argument(
        "--batch-size",
        default=args_ft.batch_size,  # Effective batch size (per worker batch size is [batch-size] / world-size)
        type=int,
        help="Batch Size",
    )

    ### Main Training
    parser.add_argument(
        "--epochs", default=args_ft.epochs, type=int, help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        default=args_ft.lr,
        type=float,
        help="Learning rate",
    )
    parser.add_argument("--wd", default=args_ft.wd, type=float, help="Weight decay")
    parser.add_argument(
        "--optimizer-momentum",
        default=args_ft.optimizer_momentum,
        type=float,
    )

    ### Dataset
    parser.add_argument(
        "--dset",
        default=args_ft.dset,
        type=str,
    )

    ### Model
    parser.add_argument(
        "--save-model",
        default=args_ft.save_model,
        type=int,
        choices=binary_choices,
        help="Save model after training.",
    )

    return parser
