#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""

import torch
from fvcore.common.registry import Registry
MODEL_REGISTRY = Registry("MODEL")

# MODEL_REGISTRY.__doc__ = """
# Registry for video model.
#
# The registered object will be called with `obj(cfg)`.
# The call should return a `torch.nn.Module` object.
# """

def build_model(cfg, model_name):
    """
    Build the GAN model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in net/config/defaults.py.
    """
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    # if model_name in ["Discriminator","D"]:
    #     name = cfg.Discriminator.MODEL_NAME  # G or D
    #
    #     # print("MODEL_REGISTRY ",MODEL_REGISTRY )
    #
    # elif model_name in ["Generator","G"]:
    #     name = cfg.Generator.MODEL_NAME
    print("MODEL_REGISTRY",MODEL_REGISTRY.__dict__)
    # choose G or D according model name
    model = MODEL_REGISTRY.get(model_name)(cfg)

    # print("model type in build py ",type(model))
    # Determine the GPU used by the current process
    # cur_device = torch.cuda.current_device()
    # Transfer the model to the current GPU device
    # model = model.cuda(device=cur_device)
    model=model.cuda()
    # Use multi-process data parallel model in the multi-gpu setting
    # if cfg.NUM_GPUS > 1:
    #     # Make model replica operate on the current device
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         module=model, device_ids=[cur_device], output_device=cur_device
    #     )
    return model

if __name__=="__main__":
    print("this is model build py ")

