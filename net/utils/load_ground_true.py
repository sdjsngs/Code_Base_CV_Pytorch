"""
load  ground true
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# avenue testing label mask
from scipy.io import loadmat
import os
# root F:\avenue\pixel ground truth\ground_truth_demo\testing_label_mask
def load_single_mat(mat_file_floder,n_clip=1,dataset="Avenue",vis=True):
    """
    :param mat_file:  mat file path
    :return:  anomaly boundary  [num ,2]
    """
    filename = '%s/%d_label.mat' % (mat_file_floder, n_clip)
    data=sio.loadmat(filename)
    volLabel=data["volLabel"]
    n_bin = np.array([np.sum(volLabel[0, i]) for i in range(len(volLabel[0]))])
    abnormal_frames_index = np.where(n_bin > 0)[0]
    ret=get_segments(abnormal_frames_index)
    if vis:
        show_single_ground_true(n_bin.shape[0],ret)
    return ret

def find_boundary(seq):
    tmp = np.insert(seq, 0, -10)
    diff = tmp[1:] - tmp[:-1]
    peaks = np.where(diff != 1)[0]
    #
    ret = np.empty((len(peaks), 2), dtype=int)
    for i in range(len(ret)):
        ret[i] = [peaks[i], (peaks[i+1]-1) if i < len(ret)-1 else (len(seq)-1)]
    return ret

def get_segments(seq):

    #
    ends = find_boundary(seq)
    # segment=np.array([[seq[curr_end[0]], seq[curr_end[1]]] for curr_end in ends]).reshape(-1) + 1  # +1 for 1-based index (same as UCSD data)
    segment = np.array([[seq[curr_end[0]], seq[curr_end[1]]] for curr_end in ends]) # .reshape(-1)
    return segment
def load_ground_truth_Avenue(folder, n_clip):
    ret = []
    for i in range(n_clip):
        filename = '%s/%d_label.mat' % (folder, i+1)
        # print(filename)
        data = loadmat(filename)['volLabel']
        n_bin = np.array([np.sum(data[0, i]) for i in range(len(data[0]))])
        abnormal_frames = np.where(n_bin > 0)[0]
        ret.append(get_segments(abnormal_frames))
    return ret



def show_single_ground_true(time_druation,anomaly_boundry,cfg=None):
    y=np.zeros(time_druation)
    for boundry in anomaly_boundry:
        y[boundry[0]:boundry[1]]=1
    x=np.arange(time_druation)
    plt.stackplot(x,y,colors='red')
    plt.show()
    return


if __name__=="__main__":
    print("ground true ")
    root=r"F:\avenue\pixel ground truth\ground_truth_demo\testing_label_mask/"
    # singel_mat=root+"1_label.mat"
    # vol=load_single_mat(root)
    # print(vol.shape)
    # load_ground_truth_Avenue(folder=root,n_clip=1)

    ret=load_ground_truth_Avenue(root,len(os.listdir(root)))


