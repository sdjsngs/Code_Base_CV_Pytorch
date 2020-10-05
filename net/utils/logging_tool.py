#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""

import builtins
import decimal
import functools
import logging
import logging.handlers
import os
import sys
import simplejson

from fvcore.common.file_io import PathManager



def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")


def setup_logging(output_dir,log_name):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)

    _FORMAT=logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m%d %H:%M:%S",
    )
    # logging.basicConfig(
    #     level=logging.INFO, format=_FORMAT, stream=sys.stdout
    # )

    sh=logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(_FORMAT)
    logger.addHandler(sh)

    out_log=os.path.join(output_dir,log_name)
    fh=logging.FileHandler(out_log) #_open_log_file(out_log)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FORMAT)
    logger.addHandler(fh)



def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def log_json_stats(stats):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.6f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("json_stats: {:s}".format(json_stats))
