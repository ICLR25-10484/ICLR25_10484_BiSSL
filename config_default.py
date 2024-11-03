"""NB: Having classes for all hyperparameters like below is not
necessary, as these could be specified in the parsers.py script.
I do however prefer this setup, as this gives a much clearer 
overview over all hyperparameters.
"""

from typing import Literal


class ArgsGeneralDefaults:
    # Load config from previous run
    # load_config_run_id: str = None

    data_dir = "./data/"  # Dir to datasets.
    model_dir = "./models/"  # Dir to models.

    download_dataset: bool = (
        False  # Allow download of dataset if not present in storage
    )

    device: Literal["cuda"] = "cuda"  # The current implementation only supports cuda
    num_workers: int = 10
    world_size: int = 1
    dist_url: str = "env://"

    omp_num_threads: int = 1


class ArgsPretextDefaults(ArgsGeneralDefaults):
    """Args for pretext_classic.py"""

    img_size: int = 96  # Size of the input image (img_size x img_size)
    img_crop_min_ratio: float = (
        0.5  # Minimum ratio of the random image cropping conducted during training
    )

    arch: str = (
        "resnet18"  # Architecture of the backbone encoder network. Verified to work with 'resnet18' and 'resnet50'
    )
    mlp: str = (
        f"{2**8}-{2**8}-{2**8}"  # Size and number of layers of the MLP expander head
    )
    proj_use_bn: bool = (
        True  # Use batch norm in the pretext projection head. (Have been set to True for all documented experiments.)
    )

    epochs: int = 600  # Number of epochs
    dset: str = "stl10"  # Dataset used for pre-training.
    batch_size: int = (
        2**10
    )  # Effective batch size (per worker batch size is [batch-size] / world-size)
    lr: float = (
        4.8  # Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256
    )
    wd: float = 1e-6  # Weight decay
    momentum: float = 0.9  # Momentum

    optimizer: Literal["sgd", "lars"] = "lars"  # Optimizer.

    # For SimCLR
    temperature: float = 0.5  # Temperature for the softmax in the contrastive loss

    save_model: bool = True  # Wether or not to store model parameters


class ArgsBiSSLDefaults(ArgsGeneralDefaults):
    """Args for bissl.py"""

    arch = ArgsPretextDefaults.arch

    img_size: int = ArgsPretextDefaults.img_size
    img_crop_min_ratio: float = ArgsPretextDefaults.img_crop_min_ratio

    epochs: int = (
        500  # Naming is a bit misleading here, as this is the number of alternations between conducting upper-level and lower-level training in BiSSL.
    )

    # Downstream Data
    d_batch_size: int = 256
    d_dset: Literal[
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
    ] = "stl10"

    # Downstream Training
    d_lr: float = 0.015
    d_wd: float = 0.01
    d_momentum: float = 0.9
    d_optimizer: Literal["sgd", "lars"] = "sgd"

    d_adaptation_freq: int = 100
    d_adaptation_strength: float = 0.1

    # Downstream Linear Warmup Hyperparameters
    d_linear_warmup_epochs: int = 20

    # Pretext Data
    p_batch_size = ArgsPretextDefaults.batch_size
    p_dset = ArgsPretextDefaults.dset

    # Pretext Model
    p_mlp = ArgsPretextDefaults.mlp
    p_proj_use_bn = ArgsPretextDefaults.proj_use_bn
    p_temperature: float = ArgsPretextDefaults.temperature

    # Pretext Training
    p_lr: float = 1.0
    p_wd: float = ArgsPretextDefaults.wd
    p_momentum = ArgsPretextDefaults.momentum
    p_optimizer: Literal["sgd", "lars"] = ArgsPretextDefaults.optimizer

    # Filename of model parameters  placed in root + model_dir
    p_pretrained_backbone_filename: str = "pretext_arch-resnet18_epochs500_bb.pth"
    p_pretrained_proj_filename: str = "pretext_arch-resnet18_epochs500_proj.pth"

    save_model: bool = True

    # Scaling of the second term in the upper level loss, assigned the symbol gamma in the paper
    upper_classic_grad_scale: float = 0.01

    # Number of either gradient steps or epochs to conduct for the respective lower and upper levels before alternating.
    # The assignment "steps" have been used throughout all experiments in the paper.
    lower_num_iter: int = 20
    lower_iter_type: Literal["steps", "epochs"] = "steps"
    upper_num_iter: int = 8
    upper_iter_type: Literal["steps", "epochs"] = "steps"

    # Cg Solver Args
    cg_lam: float = 0.001
    cg_lam_dampening: float = 10.0
    cg_solver_kwargs = dict(
        iter_num=5,
        verbose=False,
    )


class ArgsFineTuningDefaults(ArgsGeneralDefaults):
    """Args for fine-tuning.py"""

    # Pretrain Config
    backbone_origin: Literal["pretext", "bissl"] = (
        "pretext"  # Specify if the backbone is from pretext pre-training only or from using BiSSL. Only affects model naming and some console outputs.
    )
    pretrain_arch = ArgsPretextDefaults.arch
    pretrain_img_size = ArgsPretextDefaults.img_size
    pretrain_img_crop_min_ratio = ArgsPretextDefaults.img_crop_min_ratio

    # Filename of pretrained model. Assumes it is placed in root + model_dir
    pretrained_backbone_filename = "pretext_arch-resnet18_epochs500_bb.pth"

    # Hyperparamter Optimization Args

    # Whether or not to conduct a hyperparameter optimization over a random grid search of learning rates and weight decays, which ranges are specified below.
    # If false, then num_runs number of runs will be conducted, all with the same lr and wd, specified under general training args.
    use_hpo: bool = True
    hpo_lr_min: float = 0.0001
    hpo_lr_max: float = 1.0
    hpo_wd_min: float = 0.00001
    hpo_wd_max: float = 0.01

    # General Training Args
    num_runs: int = 200  # Total number of trainings to conduct
    batch_size: int = ArgsBiSSLDefaults.d_batch_size
    epochs: int = 400
    lr: int = 0.05  # Is only used if use_hpo is False
    wd: float = 0.001  # Is only used if use_hpo is False
    optimizer_momentum: float = ArgsBiSSLDefaults.d_momentum

    # Dataset Args
    dset: Literal[
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
    ] = "stl10"

    # Model args
    save_model: bool = False
