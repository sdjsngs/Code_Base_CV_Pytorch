"""
cal anomly score
"""
import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt


def calc_anomaly_score_one_frame_one_patch(in_appe, out_appe, in_flow, out_flow, thresh_cut_off=[0, 0, 0], operation=np.mean):
    """
    cal S_I and S_F in one frame  of one patch in size of 16X16X3
    :param in_appe: in img patch
    :param out_appe:  pred img patch
    :param in_flow:  in flow  patch
    :param out_flow: pred flwo patch
    :param thresh_cut_off:
    :param operation:
    :return:
    """

    assert in_appe.shape == out_appe.shape
    assert in_flow.shape == out_flow.shape

    loss_appe = (in_appe - out_appe) ** 2
    loss_flow = (in_flow - out_flow) ** 2

    # cut-off low scores to check only high scores
    if thresh_cut_off is not None:
        assert len(thresh_cut_off) == 3
        loss_appe = np.clip(loss_appe, thresh_cut_off[0], None)
        loss_flow = np.clip(loss_flow, thresh_cut_off[1], None)

    # appe_score=
    # return score map for pixel-wise assessment

    return operation(loss_appe,), operation(loss_flow,)


def calc_max_anomaly_score_one_frame(in_appe, out_appe, in_flow, out_flow,slide=16):
    """

    :param in_appe:
    :param out_appe:
    :param in_flow:
    :param out_flow:
    :param slide:  slide window 16X16
    :return:  max appe score and flow score in one frame
    """
    assert in_appe.shape == out_appe.shape
    assert in_flow.shape == out_flow.shape
    H=in_appe.shape[0]
    W=in_appe.shape[1]
    assert H % slide == 0
    assert W % slide == 0
    # slid to get patch
    appe_scores=[]
    flow_scores=[]
    for h in range(H//slide):
        for w in range(W//slide):
            appe_score,flow_score=calc_anomaly_score_one_frame_one_patch(
                in_appe[slide*h:slide*(h+1),slide*w:slide*(w+1),:],
                out_appe[slide*h:slide*(h+1),slide*w:slide*(w+1),:],
                in_flow[slide*h:slide*(h+1),slide*w:slide*(w+1),:],
                out_flow[slide*h:slide*(h+1),slide*w:slide*(w+1),:],
            )
            appe_scores.append(appe_score)
            flow_scores.append(flow_score)

    return max(appe_scores),max(flow_scores)

def get_weights_one_clip(in_appe, out_appe, in_flow, out_flow):
    assert in_appe.shape == out_appe.shape
    assert in_flow.shape == out_flow.shape
    #appe_scores,flow_scores=#
    score=np.array([calc_max_anomaly_score_one_frame(in_appe[i], out_appe[i], in_flow[i], out_flow[i])
                     for i in range((in_appe.shape[0]))])
    appe_score=score[:,0]
    flow_score=score[:,1]
    W_I=1.0/(np.mean(appe_score))
    W_F=1.0/(np.mean(flow_score))
    return W_I,W_F

def calc_anomaly_score_one_clip(in_appe, out_appe, in_flow, out_flow,l_s=0.2):

    """
    W_F,W_I
    :param in_appe:
    :param out_appe:
    :param in_flow:
    :param out_flow:
    :param l_s:
    :return:
    """
    W_I,W_F=get_weights_one_clip(in_appe,out_appe,in_flow,out_flow)
    score = np.array([calc_max_anomaly_score_one_frame(in_appe[i], out_appe[i], in_flow[i], out_flow[i])
                      for i in range((in_appe.shape[0]))])
    appe_score = score[:, 0]
    flow_score = score[:, 1]
    one_clip_score=math.log(W_I*appe_score)+l_s*math.log(W_F*flow_score)

    return one_clip_score


def calc_anomaly_score_full_clip(in_appe, out_appe, in_flow, out_flow, ):

   # load file firt





    return

    # return #[calc_anomaly_score_one_clip(in_appe[i], out_appe[i], in_flow[i], out_flow[i])
    #                   #for i in range((in_appe.shape[0]))]


def load_npy_show_cv2(img_path):
    img=np.load(img_path)
    img=img*255.0
    img=np.clip(img,a_min=0,a_max=None).astype(int)
    # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    plt.imshow(img)
    plt.show()
    # cv2.namedWindow("img", 0);
    # cv2.resizeWindow("img", 640, 480);
    # cv2.imshow("img",img)
    # cv2.waitKey()


def get_img_flow_path(img_root,flow_root,pred_img_root,pred_flow_root ,num_clips=21):
    """

    :param img_root: F:\avenue\testing\frames
    :param flow_root: F:\avenue_optical\testing\optical_flow_visualize
    :param pred_img_root: F:\avenue_save_npy\imgs
    :param pred_flow_root: F:\avenue_save_npy\flows
    :return:
    """
    img_paths = []
    flow_paths = []
    pred_img_paths = []
    pred_flow_paths = []



    for num_clip in os.listdir(img_root):
        img_path = []
        flow_path = []
        pred_img_path = []
        pred_flow_path = []
        img_path=[os.join(img_root,num_clip,(img for img in os.listdir(os.path.join(img_root,num_clip))))]



    return



if __name__=="__main__":
    print("anomaly score cal")
    x=np.random.random(size=[10,128,192,1])
    print(x.shape[0])
    # print(np.mean(x,axis=-1).shape)
    # single_img_path=r"F:\avenue_save_npy\imgs\15/0031_img_.npy"
    # load_npy_show_cv2(single_img_path)
    score=get_weights_one_clip(x,x,x,x)
