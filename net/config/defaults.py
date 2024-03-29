
"""Configs."""
from fvcore.common.config import CfgNode

from net.config import custom_config

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# BN epsilon.
_C.BN.EPSILON = 1e-5

# BN momentum.
_C.BN.MOMENTUM = 0.1

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SplitBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = ""

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 1

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = "" #r"C:\kaggle\SlowFast-Facebook/SLOWFAST_4x16_R50.pkl"

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"



# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = ""

# Total mini-batch size
_C.TEST.BATCH_SIZE = 1

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = " "

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
# _C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
# _C.TEST.NUM_SPATIAL_CROPS = 3

# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"


# # -----------------------------------------------------------------------------
# # ResNet options
# # -----------------------------------------------------------------------------
# _C.RESNET = CfgNode()
#
# # Transformation function.
# _C.RESNET.TRANS_FUNC = "bottleneck_transform"
#
# # Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
# _C.RESNET.NUM_GROUPS = 1
#
# # Width of each group (64 -> ResNet; 4 -> ResNeXt).
# _C.RESNET.WIDTH_PER_GROUP = 64
#
# # Apply relu in a inplace manner.
# _C.RESNET.INPLACE_RELU = True
#
# # Apply stride to 1x1 conv.
# _C.RESNET.STRIDE_1X1 = False
#
# #  If true, initialize the gamma of the final BN of each block to zero.
# _C.RESNET.ZERO_INIT_FINAL_BN = False
#
# # Number of weight layers.
# _C.RESNET.DEPTH = 50
#
# # If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# # kernel of 1 for the rest of the blocks.
# _C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]
#
# # Size of stride on different res stages.
# _C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]
#
# # Size of dilation on different res stages.
# _C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]




# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = ""

# Model name
_C.MODEL.MODEL_NAME = ""

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 2

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01


# -----------------------------------------------------------------------------
# Generator options
# -----------------------------------------------------------------------------
_C.Generator = CfgNode()

_C.Generator.MODEL_NAME="Generator"

_C.Generator.LOSS_FUNC="total_G_loss"

_C.Generator.INIT_LR=0.0002

_C.Generator.beta1=0.5

_C.Generator.beta2=0.9

_C.Generator.DROPOUT_RATE=0.3
# -----------------------------------------------------------------------------
#  Discriminator options
# -----------------------------------------------------------------------------
_C.Discriminator = CfgNode()

# model name for discriminator
_C.Discriminator.MODEL_NAME="Discriminator"

#loss func
_C.Discriminator.LOSS_FUNC="D_loss"

#INIT LR
_C.Discriminator.INIT_LR=0.00002


_C.Discriminator.beta1=0.5

_C.Discriminator.beta2=0.9

_C.Discriminator.DROPOUT_RATE=0.3
# action func for FC layer   sigmod in paper
_C.Discriminator.HEAD_ACT="sigmod"
# num_classes=2  real or fake
_C.Discriminator.NUM_CLASSES=2

# -----------------------------------------------------------------------------
# tensorboard
# -----------------------------------------------------------------------------

_C.TENSORBOARD = CfgNode()

_C.TENSORBOARD.ROOT=r"F:/ICCV2019_AVENUE/vis_result"



_C.NPY_SAVE = CfgNode()
_C.NPY_SAVE.IMG=r"F:\avenue_save_npy\imgs"
_C.NPY_SAVE.FLOW=r"F:\avenue_save_npy\flows"

# -----------------------------------------------------------------------------
# Avenue DATA
# -----------------------------------------------------------------------------


_C.AVENUE = CfgNode()

_C.AVENUE.GROUND_MAT=r"F:\avenue\pixel ground truth\ground_truth_demo\testing_label_mask/"

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""


# The spatial crop size of the input clip.
_C.DATA.CROP_SIZE = 224

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 32



# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3,3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.001

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "" # "cosine"

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 5

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "adam"


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1



# Output basedir.
_C.OUTPUT_DIR =  "" #"E:/SlowFastCheckpoints/HMDB51_split3-LR1e-3_SF8X8"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
# _C.RNG_SEED = 1 set seed in  train

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl" #  do not used in this time

# ---------------------------------------------------------------------------- #
# Benchmark options
# ---------------------------------------------------------------------------- #
# test data loader time
_C.BENCHMARK = CfgNode()

# Number of epochs for data loading benchmark.
_C.BENCHMARK.NUM_EPOCHS = 5

# Log period in iters for data loading benchmark.
_C.BENCHMARK.LOG_PERIOD = 100

# If True, shuffle dataloader for epoch during benchmark.
_C.BENCHMARK.SHUFFLE = True


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 1

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False




# Add custom config with default values.
custom_config.add_custom_config(_C)


def _assert_and_infer_cfg(cfg):
    # # BN assertions.
    # if cfg.BN.USE_PRECISE_STATS:
    #     assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    # assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    # assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())



if __name__ =="__main__":
    print("defaults cfg py ")
