import itertools
import argparse
import collections
import warnings

import numpy as np
import torch

import hw_tts.loss as module_loss
import hw_tts.model as module_arch
from hw_tts.trainer import Trainer
from hw_tts.utils import prepare_device
from hw_tts.utils.object_loading import get_dataloaders
from hw_tts.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    generator = config.init_obj(config["arch"]["generator"], module_arch)
    logger.info(generator)
    msd = config.init_obj(config["arch"]["msd"], module_arch)
    logger.info(msd)
    mpd = config.init_obj(config["arch"]["mpd"], module_arch)
    logger.info(mpd)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    generator = generator.to(device)
    msd = msd.to(device)
    mpd = mpd.to(device)
    if len(device_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)
        msd = torch.nn.DataParallel(msd, device_ids=device_ids)
        mpd = torch.nn.DataParallel(mpd, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = {}
    for key, value in config["loss"].items():
        loss_module[key] = config.init_obj(value, module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, generator.parameters())
    optimizer_g = config.init_obj(config["optimizer_g"], torch.optim, trainable_params)
    lr_scheduler_g = config.init_obj(config["lr_scheduler_g"], torch.optim.lr_scheduler, optimizer_g)

    trainable_params = itertools.chain(filter(lambda p: p.requires_grad, msd.parameters()), filter(lambda p: p.requires_grad, mpd.parameters()))
    optimizer_d = config.init_obj(config["optimizer_d"], torch.optim, trainable_params)
    lr_scheduler_d = config.init_obj(config["lr_scheduler_d"], torch.optim.lr_scheduler, optimizer_d)

    trainer = Trainer(
        generator,
        msd,
        mpd,
        loss_module,
        metrics,
        optimizer_g,
        optimizer_d,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
