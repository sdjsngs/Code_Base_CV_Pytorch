#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
import json
from fvcore.common.timer import Timer

import slowfast.datasets.ava_helper as ava_helper
import net.utils.logging_tool as logging
import net.utils.metrics as metrics
import net.utils.misc as misc


from sklearn.metrics import average_precision_score

logger = logging.get_logger(__name__)


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video. 30
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method # sum or max
        # Initialize tensors.

        self.video_preds = torch.zeros((num_videos, num_cls))
        if multi_label:
            self.video_preds -= 1e10

        self.video_labels = (
            torch.zeros((num_videos, num_cls))
            if multi_label
            else torch.zeros((num_videos)).long()
        )
        self.clip_count = torch.zeros((num_videos)).long()
        # Reset metric.
        self.reset()
        self.finall_label={}

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        if self.multi_label:
            self.video_preds -= 1e10
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]): #  batch ge
            vid_id = int(clip_ids[ind]) // self.num_clips
            # print("vid_id",vid_id)
            # print("shape of self video labels",self.video_labels.shape)
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                # print("preds[ind] shape",preds[ind].shape,preds[ind].sum())
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(
                    self.video_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter,top1_acc,top5_acc):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
            "top-1 acc":top1_acc,
            "top-5 acc": top5_acc,
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "test_final"}
        if self.multi_label:
            map = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            stats["map"] = map
        else:
            # num_topks_correct = metrics.topks_correct(
            #     self.video_preds, self.video_labels, ks
            # )

            # check self.video_preds
            # print("self.video_preds shape",self.video_preds.shape)
            num_topks_correct ,self.finall_label= metrics.topks_correct_each_class(
                self.video_preds, self.video_labels, ks,self.finall_label
            )

            # do each label
            topks = [
                (x / self.video_preds.size(0)) * 100.0
                for x in num_topks_correct
            ]
            # dict {label:top-1}
            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )
        # 保存dict 到json
        # print(self.finall_label)
        jsObj = json.dumps(self.finall_label)

        # fileObject = open('HMDB51_temporal_Kmeans_split1_8X8_FIX.json', 'w')
        fileObject = open('HMDB51_MAXindex_split2_SF8X8_Test_MUltiGPU.json', 'w')
        fileObject.write(jsObj)
        fileObject.close()


        logging.log_json_stats(stats)

    def each_label_test(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """


        stats = {"split": "test_each_class"}

        num_topks_correct = metrics.topks_correct(
            self.video_preds, self.video_labels, ks
        )
        topks = [
            (x / self.video_preds.size(0)) * 100.0
            for x in num_topks_correct
        ]
        # dict {label:top-1}
        assert len({len(ks), len(topks)}) == 1
        for k, topk in zip(ks, topks):
            stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                topk, prec=2
            )
        logging.log_json_stats(stats)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()

        self.loss_D = ScalarMeter(cfg.LOG_PERIOD)

        self.loss_G = ScalarMeter(cfg.LOG_PERIOD)
        self.appe_loss=ScalarMeter(cfg.LOG_PERIOD)
        self.flow_loss=ScalarMeter(cfg.LOG_PERIOD)
        self.loss_G_three_part= ScalarMeter(cfg.LOG_PERIOD)

        self.loss_D_total = 0.0
        # loss_G,appe_loss,flow_loss,loss_G_total
        self.loss_G_total = 0.0
        self.appe_loss_total=0.0
        self.flow_loss_total=0.0
        self.loss_G_three_part_total=0.0

        self.lr_G = None
        self.lr_D = None
        # Current minibatch errors (smoothed over a window).
        # self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        # self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.num_samples_G = 0
        self.num_samples_D = 0

    def reset(self):
        """
        Reset the Meter.
        """
        # self.loss.reset()
        # self.loss_total = 0.0
        # self.lr = None
        self.loss_D.reset()

        self.loss_G.reset()
        self.appe_loss.reset()
        self.flow_loss.reset()
        self.loss_G_three_part.reset()

        self.loss_D_total = 0.0

        self.loss_G_total = 0.0
        self.appe_loss_total=0.0
        self.flow_loss_total=0.0
        self.loss_G_three_part_total=0.0


        self.lr_G = None
        self.lr_D = None
        # self.mb_top1_err.reset()
        # self.mb_top5_err.reset()
        # self.num_top1_mis = 0
        # self.num_top5_mis = 0
        self.num_samples = 0
        self.num_samples_D=0
        self.num_samples_G=0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        # if not self._cfg.DATA.MULTI_LABEL:
        #     # Current minibatch stats
        #     self.mb_top1_err.add_value(top1_err)
        #     self.mb_top5_err.add_value(top5_err)
        #     # Aggregate stats
        #     self.num_top1_mis += top1_err * mb_size
        #     self.num_top5_mis += top5_err * mb_size

    def update_stats_G(self,loss_G, appe_loss, flow_loss, loss_G_three_part, lr, mb_size):
        """
        Update the current stats.
        Args:

            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss_G.add_value(loss_G)
        self.appe_loss.add_value(appe_loss)
        self.flow_loss.add_value(flow_loss)
        self.loss_G_three_part.add_value(loss_G_three_part)
        #
        self.lr_G = lr
        # self.loss_total_G+= loss * mb_size
        self.loss_G_total+=loss_G*mb_size
        self.appe_loss_total=appe_loss*mb_size
        self.flow_loss_total=flow_loss*mb_size

        self.loss_G_three_part_total=loss_G_three_part*mb_size

        self.num_samples_G += mb_size


    def update_stats_D(self,loss_D, lr, mb_size):
        """
        Update the current stats of D .
        Args:

            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss_D.add_value(loss_D)
        self.lr_D = lr
        self.loss_D_total += loss_D * mb_size
        self.num_samples_D += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter,mode):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        # stats in D or G
        if mode in ["D","Discriminator"]:
            stats = {
                "_type": "train_iter",
                "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
                "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
                "time_diff": self.iter_timer.seconds(),
                "eta": eta,

                "loss_D": self.loss_D.get_win_median(),
                "lr_D": self.lr_D,
                "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            }
        elif mode in ["G","Generator"]:
            stats = {
                "_type": "train_iter",
                "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
                "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
                "time_diff": self.iter_timer.seconds(),
                "eta": eta,

                "loss_G": self.loss_G.get_win_median(),
                "appe_loss": self.appe_loss.get_win_median(),
                "flow_loss":self.flow_loss.get_win_median(),
                "three_part_loss_G":self.loss_G_three_part.get_win_median(),
                "lr_G": self.lr_G,
                "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            }

        else:
            raise NotImplementedError(
                "Does not support  state"
            )
        logging.log_json_stats(stats)

        # stats = {
        #     "_type": "train_iter",
        #     "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
        #     "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
        #     "time_diff": self.iter_timer.seconds(),
        #     "eta": eta,
        #
        #     "loss": self.loss.get_win_median(),
        #     "lr": self.lr,
        #     "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        # }
        # if not self._cfg.DATA.MULTI_LABEL:
        #     stats["top1_err"] = self.mb_top1_err.get_win_median()
        #     stats["top5_err"] = self.mb_top5_err.get_win_median()


    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        # stats in G or D
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "lr_D": self.lr_D,
            "loss_D": self.loss_D_total / self.num_samples_D,
            "lr_G":self.lr_G,
            "loss_G": self.loss_G_total / self.num_samples_G,
            "appe_loss":self.appe_loss_total/self.num_samples_G,
            "flow_loss":self.flow_loss_total/self.num_samples_G,
            "total_G_loss":self.loss_G_three_part_total/self.num_samples_G,
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
        }

        # avg_loss = self.loss_total_D / self.num_samples_D
        # stats["loss_D"] = avg_loss

        # stats = {
        #     "_type": "train_epoch",
        #     "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
        #     "time_diff": self.iter_timer.seconds(),
        #     "eta": eta,
        #     "lr": self.lr,
        #     "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        #     "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
        # }
        # if not self._cfg.DATA.MULTI_LABEL:
        #     top1_err = self.num_top1_mis / self.num_samples
        #     top5_err = self.num_top5_mis / self.num_samples
        #     avg_loss = self.loss_total / self.num_samples
        #     stats["top1_err"] = top1_err
        #     stats["top5_err"] = top5_err
        #     stats["loss"] = avg_loss
        logging.log_json_stats(stats)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f} GB".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f} GB".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            stats["map"] = get_map(
                torch.cat(self.all_preds).cpu().numpy(),
                torch.cat(self.all_labels).cpu().numpy(),
            )
        else:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            self.min_top1_err = min(self.min_top1_err, top1_err)
            self.min_top5_err = min(self.min_top5_err, top5_err)

            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["min_top1_err"] = self.min_top1_err
            stats["min_top5_err"] = self.min_top5_err

        logging.log_json_stats(stats)


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~np.all(labels == 0, axis=0)]
    labels = labels[:, ~np.all(labels == 0, axis=0)]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap
