import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
import random
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from fvcore.common.file_io import PathManager
import time
import net.utils.logging_tool as logging
import concurrent.futures
# import matplotlib.image as mpimg

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from net.dataset.build import DATASET_REGISTRY

logger = logging.get_logger(__name__)

#构造器 调用
# print("DATASET_REGISTRY in chuk ",DATASET_REGISTRY.__dict__)

@DATASET_REGISTRY.register()
class Avenue(Dataset):
    """
    Avenue Image loader   return   img (jpg) and  optical flow png
    image size in Avenue is [360,640,3]  and resize to [192,168,3]
    Avenue image size [360,640,3]
    optical flow size [384,640,3]
    1 resize optical flow to [360,640,3]
    avenue     img
        -training
            -frames
                -01
                    -0000.jpg
                        ....
        -testing
            -frames
                    -01
                        -0000.jpg
                            ....
        testing_label_mask
            -1_label.mat
    avenur flow
    -training
            -optical_flow_visualize
                -01_visualize
                    -0000-vis.png
                        ....
        -testing
            -optical_flow_visualize
                -01_visualize
                    -0000-vis.png

    """

    def __init__(self ,cfg,mode):
        """
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            img shape = 640 320 3
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Avenue".format(mode)
        self.mode = mode
        # self.cfg = cfg
        self.data_root=r"F:\avenue"
        self.flow_root=r"F:\avenue_optical"
        self.test_mat=r"avenue.mat"
        self.img_data_paths=[]
        self.optical_flow_paths=[]
        # self.

        logger.info("Constructing Avenue {}...".format(mode))
        self._construct_loader()


    def _construct_loader(self):
        """
        Construct the avenue loader.
        """
        if self.mode in ["train", "val"]:
            # load  img and flow.png  path
            self.train_root=r"F:/avenue/training/frames"
            self.train_optical_root=r"F:/avenue_optical/training/optical_flow_visualize"
        elif self.mode in ["test"]:
            self.train_root = r"F:/avenue/testing/frames"
            self.train_optical_root = r"F:/avenue_optical/testing/optical_flow_visualize"
        for num in os.listdir(self.train_root):
            img_data_path=[]
            for single_jpg in os.listdir(os.path.join(self.train_root,num)):
                img_data_path.append(
                    os.path.join(self.train_root,num,single_jpg).replace("\\","/")
                )

            #   pop last frame which without optical flow
            self.img_data_paths+=img_data_path[:-1]

        for num in os.listdir(self.train_optical_root):
            for single_flow in os.listdir(os.path.join(self.train_optical_root, num)):
                self.optical_flow_paths.append(
                    os.path.join(self.train_optical_root, num, single_flow).replace("\\","/")
                )

        assert (len(self.img_data_paths)==len(self.optical_flow_paths)) ,"match fail in total num of data"
    def __getitem__(self, index):
        """
        Given the img index, return the list of frames, label, and video
        index
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.

        """

        frame_root=self.img_data_paths[index]
        optical_root=self.optical_flow_paths[index]
        n_clip=frame_root.split("/")[-2]
        n_img=frame_root.split("/")[-1].split(".")[0]

        # load frame and corresponding optical flow  the last frame has no optical flow
        # frame

        frame=self.load_img(frame_root)
        optical_flow=self.load_flow(optical_root)

        frame=self._normalize(frame)
        optical_flow=self._normalize(optical_flow)
        if self.mode in ["test","val"]:
            return n_clip,n_img,frame,optical_flow
        return frame,optical_flow


    def load_img(self,img_path):
        # img = mpimg.imread(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (192, 128))
        img=torch.from_numpy(img)
        img=img.permute(2,0,1)
        return img

    def load_flow(self,flow_path):
        optical_flow=cv2.imread(flow_path)
        optical_flow=cv2.cvtColor(optical_flow,cv2.COLOR_BGR2RGB)
        # resize to img size
        optical_flow = cv2.resize(optical_flow, (192, 128))
        # optical_flow=cv2.resize(optical_flow,(640,360))# opencv resize (W,H) (192,128)
        optical_flow=torch.from_numpy(optical_flow)
        optical_flow=optical_flow.permute(2,0,1)
        return  optical_flow

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.img_data_paths)

    def _normalize(self,frame):
        """

        :param frame:
        :return:
        """
        frame=frame/255.0
        return frame



if __name__=="__main__":


    # data_root=r"F:\avenue\training\frames\01/0000.jpg"

    train_data=DataLoader(Avenue(cfg=None,mode="test"),batch_size=1,shuffle=False)
    for step,(n_clip,n_img,img,optical) in enumerate(train_data):
        print("n_clip",n_clip)
        print("n_img",n_img)
        print("img shape=",img.shape)
        print("optical shape= ",optical.shape)

