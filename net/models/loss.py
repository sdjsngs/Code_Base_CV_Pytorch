"""Loss functions."""
import torch
import  torch.nn as nn
import  torch.nn.functional as F

def gradient(x):
    # tf.image.image_gradients(image)
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    l = x
    r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    t = x
    b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = torch.abs(r - l), torch.abs(b - t)
    # dx will always have zeros in the last column, r-l
    # dy will always have zeros in the last row,    b-t
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy

def Gradient_loss(gen_frames,gt_frames):
    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    #
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)
    # condense into one tensor and avg
    return torch.mean(grad_diff_x  + grad_diff_y)

def MSE_loss(img_pred,img):
    mse_loss=nn.MSELoss()
    loss=mse_loss(img_pred,img)
    return loss
def L1_loss(img_pred,img):
    l1_loss=nn.L1Loss()
    loss=l1_loss(img_pred,img)
    return loss
def BCE_loss(img_pred,img_label):
    bce_loss=nn.BCELoss()
    loss=bce_loss(img_pred,img_label)
    return loss
def int_loss(img_pred,img):
    return MSE_loss(img_pred,img)
def flow_loss(flow_pred,flow):
    return L1_loss(flow_pred,flow)
def appe_loss(img_pred,img):
    return int_loss(img_pred,img)+Gradient_loss(img_pred,img)


def G_loss(D_fake_logits,D_fake):

    return BCE_loss(D_fake_logits,D_fake)
def total_G_loss(D_fake_logits,D_fake,img_pred,img,flow_pred,flow):

    totalGloss=0.25*G_loss(D_fake_logits,D_fake)+appe_loss(img_pred,img)+2*flow_loss(flow_pred,flow)

    return  G_loss(D_fake_logits,D_fake),appe_loss(img_pred,img),flow_loss(flow_pred,flow),totalGloss


def D_loss(D_real_logits,D_real,D_fake_logits,D_fake,):

    return 0.5*BCE_loss(D_real_logits,D_real)+0.5* BCE_loss(D_fake_logits,D_fake)

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "G_loss": G_loss,
    "D_loss":D_loss,
    "total_G_loss":total_G_loss,
}





def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]



if __name__=="__main__":
    print("this is loss")