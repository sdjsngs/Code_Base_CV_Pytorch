"""
tensorboard  visualize
"""
import torch
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# summary_root=r""
# #
# writer = SummaryWriter(summary_root)

def init_summary_writer(summary_root):
    writer = SummaryWriter(summary_root)
    return writer


def show_img_and_flow(writer,img,pred_img,flow,pred_flow):

    writer.add_image("raw img" ,img)
    writer.add_image("pred img", pred_img)
    writer.add_image("raw flow",flow)
    writer.add_image("pred flow",pred_flow)

def loss_add(writer,loss_name,loss_item,cur_epoch):
    writer.add_scalar(loss_name, loss_item, cur_epoch)

if __name__=="__main__":
    print("tensorbaord visition part ")





