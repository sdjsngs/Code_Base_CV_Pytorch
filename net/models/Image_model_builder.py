"""
Conv AE  in pytorch

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from net.models.build import MODEL_REGISTRY
from .build import MODEL_REGISTRY
class conv2d_Inception(nn.Module):
    def __init__(self,inplace,place):
        super(conv2d_Inception,self).__init__()
        self.n_branch=4
        place=place//4
        #self.s1_11 = nn.Conv2d(in_channels=inplace,out_channels=place,kernel_size=(1,1),stride=1,padding=0)
        self.s_11 = nn.Conv2d(in_channels=inplace,out_channels=place,kernel_size=(1,1),stride=1,padding=0)
        self.s_1n = nn.Conv2d(in_channels=place, out_channels=place, kernel_size=(1, 3), stride=1, padding=(0,1))
        self.s_n1 = nn.Conv2d(in_channels=place, out_channels=place, kernel_size=(3,1), stride=1, padding=(1,0))
        # self.s5_11 = nn.Conv2d(in_channels=inplace, out_channels=place, kernel_size=(1, 1), stride=1, padding=0)
        # self.s5_1n = nn.Conv2d(in_channels=place, out_channels=place, kernel_size=(1, 3), stride=1, padding=(0, 1))
        # self.s5_n1 = nn.Conv2d(in_channels=place, out_channels=place, kernel_size=(3, 1), stride=1, padding=(1, 0))
        # self.s7_11 = nn.Conv2d(in_channels=inplace, out_channels=place, kernel_size=(1, 1), stride=1, padding=0)
        # self.s7_1n = nn.Conv2d(in_channels=place, out_channels=place, kernel_size=(1, 3), stride=1, padding=(0, 1))
        # self.s7_n1 = nn.Conv2d(in_channels=place, out_channels=place, kernel_size=(3, 1), stride=1, padding=(1, 0))

    def forward(self, x):
        # do 1x1 3x3 5x5 7x7
        out1=self.s_11(x)

        out3=self.s_11(x)
        out3=self.s_1n(out3)
        out3=self.s_n1(out3)

        out5 = self.s_11(x)
        out5 = self.s_1n(out5)
        out5 = self.s_n1(out5)
        out5 = self.s_1n(out5)
        out5 = self.s_n1(out5)

        out7 = self.s_11(x)
        out7 = self.s_1n(out7)
        out7 = self.s_n1(out7)
        out7 = self.s_1n(out7)
        out7 = self.s_n1(out7)
        out7 = self.s_1n(out7)
        out7 = self.s_n1(out7)

        # concat in channel dim
        out=torch.cat([out1,out3,out5,out7],dim=1)
        return  out

class Encoder(nn.Module):
    def __init__(self):
        """
        input  192,128,3 or bigger than it
        """
        super(Encoder,self).__init__()
        self.place=64
        self.Inception=conv2d_Inception(inplace=3,place=self.place)
        self.conv1=self._make_Sequential(in_channel=64,out_channel=64,bn=False)
        self.conv2 = self._make_Sequential(in_channel=64,out_channel=128,stride=2)
        self.conv3 = self._make_Sequential(in_channel=128, out_channel=256, stride=2)
        self.conv4 = self._make_Sequential(in_channel=256, out_channel=512, stride=2)
        self.conv5 = self._make_Sequential(in_channel=512, out_channel=512, stride=2)

    def forward(self, x):
        """

        :param x:  shape=[128,192,3]
        :return:
        """
        x=self.Inception(x)
        x1=self.conv1(x)
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        x4=self.conv4(x3)
        x5=self.conv5(x4)

        # return [x1,x2,x3,x4,x5]
        return x5,[x4,x3,x2,x1]
    def _make_Sequential(self,in_channel,out_channel,kernel_size=3,stride=1,padding=1,bn=True):
        if bn:
            return  nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding),
                nn.ReLU(),
            )


class Decoder_imgs(nn.Module):
    def __init__(self):
        super(Decoder_imgs,self).__init__()
        self.deconv_1 = self._construct_deconv_layer(in_channel=512,out_channel=512,)
        self.deconv_2 = self._construct_deconv_layer(in_channel=512,out_channel=256)
        self.deconv_3 = self._construct_deconv_layer(in_channel=256, out_channel=128)
        self.deconv_4 = self._construct_deconv_layer(in_channel=128, out_channel=64)
        self.conv_5 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        d1=self.deconv_1(x)
        d2=self.deconv_2(d1)
        d3=self.deconv_3(d2)
        d4=self.deconv_4(d3)
        pred_imgs=self.conv_5(d4)

        return pred_imgs

    def _construct_deconv_layer(self,in_channel,out_channel,kernel_size=3,stride=2,padding=1,output_padding=(1,1),drop_prob=0.7):
        """
        deconv
        BN
        dropout
        action
        :return:
        """
        decoder_layer=nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=output_padding),
            nn.Dropout2d(drop_prob),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        return decoder_layer
class Decoder_flows(nn.Module):
    def __init__(self):
        super(Decoder_flows,self).__init__()
        self.deconv_1 = self._construct_deconv_layer(in_channel=512,out_channel=512,)
        self.deconv_2 = self._construct_deconv_layer(in_channel=512*2,out_channel=256)
        self.deconv_3 = self._construct_deconv_layer(in_channel=256*2, out_channel=128)
        self.deconv_4 = self._construct_deconv_layer(in_channel=128*2, out_channel=64)
        self.conv_5 = nn.Conv2d(in_channels=64*2,out_channels=3,kernel_size=3,stride=1,padding=1)

    def forward(self, x,encoder_list):
        d1=self.deconv_1(x)
        d1_cat=self.short_concat(d1,encoder_list[0])
        d2=self.deconv_2(d1_cat)
        d2_cat=self.short_concat(d2,encoder_list[1])
        d3=self.deconv_3(d2_cat)
        d3_cat=self.short_concat(d3,encoder_list[2])
        d4=self.deconv_4(d3_cat)
        d4_cat=self.short_concat(d4,encoder_list[3])
        pred_flows=self.conv_5(d4_cat)

        return pred_flows

    def _construct_deconv_layer(self, in_channel, out_channel, kernel_size=3, stride=2, padding=1, output_padding=(1,1),
                                drop_prob=0.7):
        """
        deconv
        BN
        dropout
        action
        :return:
        """
        decoder_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding),
            nn.Dropout2d(drop_prob),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        return decoder_layer

    def short_concat(self,deconder_feature,encoder_feature):
        return torch.cat((deconder_feature,encoder_feature),dim=1)



# print("MODEL_REGISTRY.__dict__ in image model builder py ",MODEL_REGISTRY.__dict__)
@MODEL_REGISTRY.register()
class Generator(nn.Module):
    def  __init__(self,cfg):
        super(Generator,self).__init__()
        #   encoder channel [3,64,128,256,512,512]
        #   decoder channel [512,256,128,64,3]
        self.cfg = cfg
        self.encoder=Encoder()
        self.decoder_imgs=Decoder_imgs()
        self.decoder_flows=Decoder_flows()

    def forward(self,x):
        x5,encoder_list=self.encoder(x)
        pred_imgs=self.decoder_imgs(x5)
        pred_flows=self.decoder_flows(x5,encoder_list)
        return pred_imgs,pred_flows



@MODEL_REGISTRY.register()
class Discriminator(nn.Module):
    def __init__(self,cfg):
        super(Discriminator, self).__init__()
        self.cfg=cfg
        self.conv1=self._make_layer(6,64,kernel_size=3,stride=1,padding=1,bn=False)
        self.conv2=self._make_layer(64,128,stride=2)
        self.conv3 = self._make_layer(128, 256, stride=2)
        self.conv4 = self._make_layer(256, 512, stride=2)
        self.Sigmoid=nn.Sigmoid()
    def forward(self, input_img,input_flow):
        """
        判别器
        :param input:  shape=[B,6,128,192]
        :return: [B,512,12,24]
        """
        input=torch.cat([input_img,input_flow],dim=1)
        x1=self.conv1(input)
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        x4=self.conv4(x3)
        score=self.Sigmoid(x4)

        return score,x4

    def _make_layer(self,in_channel,out_channel,kernel_size=3,stride=1,padding=1,bn=True):
        if bn:
            return  nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.1),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding),
                nn.LeakyReLU(0.1),
            )



if __name__=="__main__":
    print("this is Conv AE ")

    imgs = torch.randn(size=(4, 3, 192, 128)).cuda()
    flows=torch.randn(size=(4,3,192,128)).cuda()
    # decoder=Discriminator().cuda()
    # score,x4=decoder(imgs,flows)
    cfg=None
    encoder=Generator(cfg).cuda()
    print(encoder(imgs)[0].shape,encoder(imgs)[1].shape)
    #