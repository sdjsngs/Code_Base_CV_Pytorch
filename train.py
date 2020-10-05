"""
train py
train 15 epoch  with batch size =16    #  do better  img size [192,128] in paper
Optical flow from FlowNet2
optical flow  scale in flownet2   (384,512 ,3)
avenue scale (360,640,3) /  (384,640,3)
scale in paper  128,192,3 (h,w,3)
rescale to paper scale?
"""
import torch
import  torch.nn as nn
import  torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import random
import numpy as np
from net.dataset import loader
from net.models.build import build_model
from net.utils.parser import load_config, parse_args
import net.models.optimizer as optim
from net.models import loss as losses
import net.utils.logging_tool as logging
import net.utils.misc as misc
import net.utils.checkpoint as cu
from net.utils.meters import  TrainMeter, ValMeter
from net.utils.tensorboard_vis import init_summary_writer,show_img_and_flow,loss_add
from net.utils.AverageMeter import AverageMeter
logger = logging.get_logger(__name__)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(train_loader, model_G,model_D, optimizer_G,optimizer_D, train_meter, cur_epoch, writer,cfg):
    """
    train G get pred_imgs and pred_flows
    train D get real feature real label fake feature fake label
    loss {
    D_loss:
    G_loss:
    appe_loss:
    flow_loss:
    total_G_loss:
    }
    :param train_loader:
    :param model_G:
    :param model_D:
    :param optimizer_G:
    :param optimizer_D:
    :param train_meter:
    :param cur_epoch:
    :param cfg:
    :return:
    """
    print("cur_epoch,", cur_epoch)
    model_G.train()
    model_D.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    loss_D_meter=AverageMeter()
    loss_G_meter=AverageMeter()
    appe_loss_meter=AverageMeter()
    flow_loss_meter=AverageMeter()
    loss_G_total_meter=AverageMeter()


    for cur_iter, (imgs, flows) in enumerate(train_loader):
        imgs=imgs.float().cuda()
        flows=flows.float().cuda()
        # keep LR in  0.00002 in D and 0.0002 in G
        lr_D=0.00002
        lr_G=0.0002

        #  frozen G  to get first pred flow
        # G input rgb-frame generate frame and flow
        # D take [rgb-frame,flow ] do
        # do G get pred flow
        # print("next(model_G.parameters()).is_cuda", next(model_G.parameters()).is_cuda)
        # logger.info("next(model_G.parameters()).is_cuda in train ", next(model_G.parameters()).is_cuda)
        # logger.info(imgs.shape)
        pred_imgs, pred_flows = model_G(imgs)

        D_real_logits, D_real = model_D(imgs, flows)
        D_fake_logits, D_fake = model_D(imgs, pred_flows)

        loss_fun_D = losses.get_loss_func(cfg.Discriminator.LOSS_FUNC)
        loss_D = loss_fun_D(D_real_logits, torch.ones_like(D_real), D_fake_logits, torch.zeros_like(D_fake))

        # check Nan Loss.
        misc.check_nan_losses(loss_D)
        optimizer_D.zero_grad()
        loss_D.backward(retain_graph=True)
        # Update the parameters.
        optimizer_D.step()
        loss_D = loss_D.item()
        loss_D_meter.update(loss_D,imgs.size(0))
        train_meter.iter_toc()
        # Update and log stats
        # recoding  G_lr  int_loss ,
        train_meter.update_stats_D(
            loss_D, lr_D, imgs[0].size(0) * cfg.NUM_GPUS
        )
        train_meter.log_iter_stats(cur_epoch, cur_iter, mode="D")
        train_meter.iter_tic()

        # loss G
        loss_fun_G = losses.get_loss_func(cfg.Generator.LOSS_FUNC)
        loss_G, appe_loss,flow_loss,loss_G_total = loss_fun_G(D_fake_logits, torch.ones_like(D_fake), pred_imgs, imgs, pred_flows, flows)


        # check Nan Loss. 3 + 1  loss in G
        misc.check_nan_losses(loss_G)
        misc.check_nan_losses(appe_loss)
        misc.check_nan_losses(flow_loss)
        misc.check_nan_losses(loss_G_total)
        optimizer_G.zero_grad()
        loss_G_total.backward()
        # Update the parameters.
        optimizer_G.step()
        # loss.item()
        loss_G=loss_G.item()
        appe_loss=appe_loss.item()
        flow_loss=flow_loss.item()
        loss_G_total=loss_G_total.item()

        loss_G_meter.update(loss_G, imgs.size(0))
        appe_loss_meter.update(appe_loss, imgs.size(0))
        flow_loss_meter.update(flow_loss,imgs.size(0))
        loss_G_total_meter.update(loss_G_total,imgs.size(0))
        train_meter.iter_toc()
        # Update and log stats
        # recoding  G_lr  int_loss ,
        train_meter.update_stats_G(
            loss_G,appe_loss,flow_loss,loss_G_total,lr_G, imgs[0].size(0) * cfg.NUM_GPUS
        )
        train_meter.log_iter_stats(cur_epoch, cur_iter,mode="G")
        train_meter.iter_tic()

    loss_add(writer,"loss_D",loss_D_meter.avg,cur_epoch=cur_epoch)
    loss_add(writer, "loss_G", loss_G_meter.avg,cur_epoch=cur_epoch)
    loss_add(writer, "loss_appe", appe_loss_meter.avg,cur_epoch=cur_epoch)
    loss_add(writer, "loss_flow", flow_loss_meter.avg,cur_epoch=cur_epoch)
    loss_add(writer, "loss_G_total", loss_G_total_meter.avg,cur_epoch=cur_epoch)

    train_meter.log_epoch_stats(cur_epoch)

    train_meter.reset()

    return


def train(cfg):

    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Train with config:")
    logger.info("init tensorboard :")
    writer=init_summary_writer(cfg.TENSORBOARD.ROOT)
    print("build GAN model")
    # model = build_model(cfg)
    model_D = build_model(cfg, model_name="Discriminator")
    model_G = build_model(cfg, model_name="Generator")

    # print("next(model_G.parameters()).is_cuda",model_G.device)
    # print("next(model_D.parameters()).is_cuda",next(model_G.parameters()).is_cuda)

    # optimizer = optim.construct_optimizer(model, cfg)
    optimizer_G = optim.construct_optimizer(model_G, cfg,model_name="Generator")
    optimizer_D = optim.construct_optimizer(model_D, cfg,model_name="Discriminator")
    print(" build G optimizers   and D optimizers")
    # logger.info("next(model_G.parameters()).is_cuda", next(model_G.parameters()).is_cuda)
    # logger.info("next(model_D.parameters()).is_cuda",next(model_D.parameters()).is_cuda)
    # load model or not
    start_epoch=0
    train_loader=loader.construct_loader(cfg,"train") # for this work return [img,flow]

    train_meter = TrainMeter(len(train_loader), cfg)
    # val_meter = ValMeter(len(val_loader), cfg)

    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        train_epoch(train_loader, model_G,model_D, optimizer_G,optimizer_D, train_meter, cur_epoch, writer,cfg)

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):

            cu.save_checkpoint(cfg.OUTPUT_DIR, model_G, "Generator",optimizer_G, cur_epoch, cfg)
            cu.save_checkpoint(cfg.OUTPUT_DIR, model_D, "Discriminator",optimizer_D, cur_epoch, cfg)
        # Evaluate the model on validation set.
        # if misc.is_eval_epoch(cfg, cur_epoch):
        #     eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
    writer.close()
if __name__=="__main__":
    """

       :param cfg: 
       :return:
       1 load model
       2 load data
       3 train
       4 save checkpoints 
       """
    torch.backends.cudnn.benchmark = True
    setup_seed(10)
    args = parse_args()
    cfg = load_config(args)
    train(cfg)







