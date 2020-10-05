#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions that handle saving and loading of checkpoints."""

import os
import pickle
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager


import net.utils.logging_tool as logging

# from slowfast.utils.c2_model_loading import get_name_convert_func

logger = logging.get_logger(__name__)


def make_checkpoint_dir(path_to_job):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    checkpoint_dir = os.path.join(path_to_job, "checkpoints")
    # Create the checkpoint dir from the master process
    if  not PathManager.exists(checkpoint_dir):
        try:
            PathManager.mkdirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir


def get_checkpoint_dir(path_to_job):
    """
    Get path for storing checkpoints.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    return os.path.join(path_to_job, "checkpoints")


def get_path_to_checkpoint(path_to_job, model_name,epoch):
    """
    Get the full path to a checkpoint file.
    Args:
        path_to_job (string): the path to the folder of the current job.
        model_name : G or D
        epoch (int): the number of epoch for the checkpoint.
    """
    name = "checkpoint_epoch_{}_{:05d}.pyth".format(model_name,epoch)
    return os.path.join(get_checkpoint_dir(path_to_job), name)


def get_last_checkpoint(path_to_job):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """

    d = get_checkpoint_dir(path_to_job)
    names = PathManager.ls(d) if PathManager.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    assert len(names), "No checkpoints found in '{}'.".format(d)
    # Sort the checkpoints by epoch.
    name = sorted(names)[-1]
    return os.path.join(d, name)


def has_checkpoint(path_to_job):
    """
    Determines if the given directory contains a checkpoint.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = get_checkpoint_dir(path_to_job)
    files = PathManager.ls(d) if PathManager.exists(d) else []
    return any("checkpoint" in f for f in files)


def is_checkpoint_epoch(cur_epoch, checkpoint_period):
    """
    Determine if a checkpoint should be saved on current epoch.
    Args:
        cur_epoch (int): current number of epoch of the model.
        checkpoint_period (int): the frequency of checkpointing.
    """
    return (cur_epoch + 1) % checkpoint_period == 0


def save_checkpoint(path_to_job, model, model_name,optimizer, epoch, cfg):
    """
    Save a checkpoint.
    Args:
        model (model): model to save the weight to the checkpoint.
        optimizer (optim): optimizer to save the historical state.
        epoch (int): current number of epoch of the model.
        cfg (CfgNode): configs to save.
    """
    # Save checkpoints only from the master process.

    # Ensure that the checkpoint dir exists.
    PathManager.mkdirs(get_checkpoint_dir(path_to_job))
    # Omit the DDP wrapper in the multi-gpu setting.
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint.
    path_to_checkpoint = get_path_to_checkpoint(path_to_job, model_name,epoch + 1)
    # if (epoch+1)%10==0 or(epoch+1)==cfg.SOLVER.MAX_EPOCH :
    with PathManager.open(path_to_checkpoint, "wb") as f:
        torch.save(checkpoint, f)
    return path_to_checkpoint





def load_checkpoint(
    path_to_checkpoint,
    model,
    data_parallel=True,
    optimizer=None,
    inflation=False,
    convert_from_caffe2=False,
):
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
        data_parallel (bool): if true, model is wrapped by
        torch.nn.parallel.DistributedDataParallel.
        optimizer (optim): optimizer to load the historical state.
        inflation (bool): if True, inflate the weights from the checkpoint.
        convert_from_caffe2 (bool): if True, load the model from caffe2 and
            convert it to pytorch.
    Returns:
        (int): the number of training epoch of the checkpoint.
    change: more conv layer  above
    """
    assert PathManager.exists(
        path_to_checkpoint
    ), "Checkpoint '{}' not found".format(path_to_checkpoint)
    # Account for the DDP wrapper in the multi-gpu setting.
    ms = model.module if data_parallel else model
    # print("ms type ",type(ms))
    # print("convert_from_caffe2=",convert_from_caffe2)


    # Load the checkpoint on CPU to avoid GPU mem spike.
    # checkpoint name =path_to_checkpoint.split("/")[-1]
    with PathManager.open(path_to_checkpoint, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu") # load checkpoint
        print("type checkpoint",type(checkpoint))
    # if inflation:
    #     # Try to inflate the model.
    #     model_state_dict_3d = (
    #         model.module.state_dict()
    #         if data_parallel
    #         else model.state_dict()
    #     )
    #     inflated_model_dict = inflate_weight(
    #         checkpoint["model_state"], model_state_dict_3d
    #     )
    #     ms.load_state_dict(inflated_model_dict, strict=False)

    """
    pretrained_dict=torch.load(model_weight)
    model_dict=myNet.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    """

    ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "epoch" in checkpoint.keys():
        epoch = checkpoint["epoch"]
    else:
        epoch = -1
    return epoch
