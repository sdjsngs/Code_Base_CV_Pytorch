#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"

    # Find the top max_k predictions for each sample
    # print("labels",labels,type(labels[0]))
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    # print("labels=",labels.shape,labels)

    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # top_max_k_correct shape [5,batch_size]
    # Compute the number of topk correct predictions for each k.
    topks_correct = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    # print("len(topks_correct):",len(topks_correct))
    return topks_correct
def topks_correct_each_class(preds, labels, ks, finally_label):
    """
    Given the predictions, labels, and a list of top-k values,
    return  label:correct
    compute the number of correct predictions for each top-k value.
    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.
    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    # create label:correct
    # assert (type(finally_label)=="dict") ,[type(finally_label)," is not a dict  "]
    for label in labels:
        if not finally_label.get(label.cpu()):
            finally_label[str(label.detach().cpu().numpy()) + "_top-1"] = 0.0
            finally_label[str(label.detach().cpu().numpy()) + "_top-5"] = 0.0
            finally_label[str(label.detach().cpu().numpy()) + "_count"] = 0.0
            finally_label[str(label.detach().cpu().numpy()) +"_wrong_video_name_in_top1"] = []
            finally_label[str(label.detach().cpu().numpy()) +"_wrong_video_name_in_top5"] = []



    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    for step,label in enumerate(labels):
        top_1=top_max_k_correct[:1, int(step)].view(-1).float().sum().cpu().numpy()
        top_5=top_max_k_correct[:5, int(step)].view(-1).float().sum().cpu().numpy()
        finally_label[str(label.detach().cpu().numpy()) + "_top-1"] += top_1
        finally_label[str(label.detach().cpu().numpy()) + "_top-5"] += top_5
        finally_label[str(label.detach().cpu().numpy()) + "_count"] +=1
        if top_1==0:
            finally_label[str(label.detach().cpu().numpy()) +"_wrong_video_name_in_top1"].append([step+1,top_max_k_inds[0,step].detach().cpu().numpy().tolist()])
        if top_5 == 0:
            finally_label[str(label.detach().cpu().numpy()) +"_wrong_video_name_in_top5"].append([step + 1, top_max_k_inds[:, step].detach().cpu().numpy().tolist()])


    topks_correct = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    # print("len(topks_correct):",len(topks_correct))
    # print ("finally label",finally_label)
    return topks_correct ,finally_label


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]



if __name__=="__main__":
    print("do topk correct")