import math
import os
import datetime
import time
import random
import numpy as np
import argparse
from requests import get
import wandb
from tqdm import tqdm
from loguru import logger

import torch
import torch.utils.data
from torch import mode, nn
import torchvision
from torchvision import transforms
from torchvision import models
import torch.distributed.optim

from mmdet.registry import MODELS
import utils

from spikingjelly.clock_driven import functional


def get_model(args):
    resnet = MODELS.build(
        dict(
            type="SpikeResNet",
            depth=50,
            out_indices=(1, 2, 3),
            spike_cfg=dict(spike_mode="if", spike_backend="cupy", spike_T=args.T),
            train_cls=True,
        ),
    )
    # model = models.resnet50(pretrained=False)
    return resnet


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Classification Training")

    parser.add_argument("--data-path", default="/root/autodl-tmp/imagenet", help="dataset")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("-b", "--batch-size", default=48, type=int)
    parser.add_argument(
        "--epochs", default=320, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=16,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument("--lr", default=0.0025, type=float, help="initial learning rate")
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="Momentum for SGD. Adam will not use momentum",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="weight decay (default: 0)",
        dest="weight_decay",
    )
    parser.add_argument("--output-dir", default=".", help="path where to save")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use AMP training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--T", default=4, type=int, help="simulation steps")
    parser.add_argument(
        "--adam", action="store_true", help="Use Adam. The default optimizer is SGD."
    )

    parser.add_argument("--cos_lr_T", default=320, type=int, help="T_max of CosineAnnealingLR.")

    parser.add_argument(
        "--zero_init_residual", action="store_true", help="zero init all residual blocks"
    )

    args = parser.parse_args()
    return args


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, total_epochs, scaler):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    if utils.is_main_process():
        pbar = tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{total_epochs}")
    else:
        pbar = None
    for image, target in data_loader:
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError("loss is Nan")

        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters["acc1"].update(acc1_s, n=batch_size)
        metric_logger.meters["acc5"].update(acc5_s, n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        if pbar is not None:
            pbar.set_postfix(loss=loss_s, lr=metric_logger.lr.global_avg)
            pbar.update(1)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return (
        metric_logger.loss.global_avg,
        metric_logger.acc1.global_avg,
        metric_logger.acc5.global_avg,
    )


def eval(model, criterion, epoch, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        if utils.is_main_process():
            pbar = tqdm(total=len(data_loader), desc=f"Epoch {epoch+1} Test ")
        else:
            pbar = None
        for batch in data_loader:
            images, target = batch
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            if pbar is not None:
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = (
        metric_logger.loss.global_avg,
        metric_logger.acc1.global_avg,
        metric_logger.acc5.global_avg,
    )
    return loss, acc1, acc5


def train_loops(
    args,
    model,
    model_without_ddp,
    criterion,
    optimizer,
    lr_scheduler,
    scaler,
    train_loader,
    val_loader,
    device,
    output_dir,
):
    max_test_acc1 = 0.0
    test_acc5_at_max_test_acc1 = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        args.start_epoch = checkpoint["epoch"] + 1

        max_test_acc1 = checkpoint["max_test_acc1"]
        test_acc5_at_max_test_acc1 = checkpoint["test_acc5_at_max_test_acc1"]

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, args.epochs, scaler
        )

        lr_scheduler.step()

        test_loss, test_acc1, test_acc5 = eval(model, criterion, epoch, val_loader, device)

        if utils.is_main_process():
            print(
                f"Epoch[{epoch}] : train_acc1: {train_acc1}, train_acc5: {train_acc5}, train_loss: {train_loss}, test_acc1: {test_acc1}, test_acc5: {test_acc5}, test_loss: {test_loss}"
            )
            logger.info(
                f"Epoch[{epoch}] : train_acc1: {train_acc1}, train_acc5: {train_acc5}, train_loss: {train_loss}, test_acc1: {test_acc1}, test_acc5: {test_acc5}, test_loss: {test_loss}"
            )
            wandb.log(
                {
                    "train_acc1": train_acc1,
                    "train_acc5": train_acc5,
                    "train_loss": train_loss,
                    "test_acc1": test_acc1,
                    "test_acc5": test_acc5,
                    "test_loss": test_loss,
                    "lr": np.float16(lr_scheduler.get_last_lr()[0]),
                }
            )

        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True

        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            "max_test_acc1": max_test_acc1,
            "test_acc5_at_max_test_acc1": test_acc5_at_max_test_acc1,
        }
        if utils.is_main_process():
            utils.save_checkpoint(
                args=args,
                checkpoint=checkpoint,
                output_dir=output_dir,
                save_max=save_max,
                epoch=epoch,
            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(output_dir)

        print(
            "Training time {}".format(total_time_str),
            "max_test_acc1",
            max_test_acc1,
            "test_acc5_at_max_test_acc1",
            test_acc5_at_max_test_acc1,
        )


def main():
    args = parse_args()
    args = utils.init_distributed_mode(args)
    output_dir = utils.setup_output(args)

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset_train, dataset_test, train_sampler, test_sampler = utils.load_data(
        train_dir, val_dir, args.cache_dataset, args.distributed
    )

    print(f"dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}")

    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    model = get_model(args)

    device = torch.device(args.device)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            # nesterov=True,
        )
    if args.amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cos_lr_T)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    train_loops(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        train_loader=data_loader,
        val_loader=data_loader_test,
        device=device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
