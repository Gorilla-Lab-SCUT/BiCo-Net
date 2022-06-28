import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pspnet import PSPNet

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

class ModifiedResnet(nn.Module):
    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()
        self.model = psp_models['resnet18'.lower()]()

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.e_conv1 = nn.Conv1d(32, 64, 1)
        self.e_conv2 = nn.Conv1d(64, 128, 1)    
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.ap1 = nn.AvgPool1d(num_points)
        self.conv5 = nn.Conv1d(1408, 512, 1)
        self.num_points = num_points

    def forward(self, x, emb):     
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1) # (B, 128, N)
        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1) # (B, 256, N)   
        
        x = F.relu(self.conv3(pointfeat_2))
        x = F.relu(self.conv4(x))
        ap_x = self.ap1(x)
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points) # (B, 1024, N)
        feat1 = torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) # 128 + 256 + 1024
        feat1 = F.relu(self.conv5(feat1)) # (B, 512, N)
        return feat1 

class ModelFeat(nn.Module):
    def __init__(self, num_points):
        super(ModelFeat, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.e_conv1 = nn.Conv1d(3, 64, 1)
        self.e_conv2 = nn.Conv1d(64, 128, 1)    
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.ap1 = nn.AvgPool1d(num_points)
        self.conv5 = nn.Conv1d(1408, 512, 1)
        self.num_points = num_points

    def forward(self, cad):
        x = cad[:, 0:6, :]# (B, 6, N)
        emb = cad[:, 6:9, :]# (B, 3, N)

        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1) # (B, 128, N)
        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1) # (B, 256, N)
        
        x = F.relu(self.conv3(pointfeat_2))
        x = F.relu(self.conv4(x))
        ap_x = self.ap1(x)
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points) # (B, 1024, N)
        feat1 = torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) # 128 + 256 + 1024
        feat1 = F.relu(self.conv5(feat1)) # (B, 512, N)
        return feat1

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)
        self.modelfeat = ModelFeat(num_points)

        self.conv1_avg = nn.Conv1d(512, 512, 1)
        self.conv2_avg = nn.Conv1d(512, 1024, 1)
        self.conv3_avg = nn.Conv1d(512, 512, 1)
        self.conv4_avg = nn.Conv1d(512, 1024, 1)
        self.conv5_avg = nn.Conv1d(512, 512, 1)
        self.conv6_avg = nn.Conv1d(512, 512, 1)
        
        '---------------------------------------------------------------------------------'         

        self.conv1_m = nn.Conv1d(1536, 1024, 1)
        self.conv2_m = nn.Conv1d(1024, 512, 1)
        self.conv3_m = nn.Conv1d(512, 128, 1)
        self.conv4_m = nn.Conv1d(128, num_obj*3, 1) 

        self.conv1_n = nn.Conv1d(1536, 1024, 1)
        self.conv2_n = nn.Conv1d(1024, 512, 1)
        self.conv3_n = nn.Conv1d(512, 128, 1)
        self.conv4_n = nn.Conv1d(128, num_obj*3, 1)
                  
        self.conv1_r = nn.Conv1d(1536+512, 1024, 1)
        self.conv2_r = nn.Conv1d(1024, 512, 1)
        self.conv3_r = nn.Conv1d(512, 128, 1)
        self.conv4_r = nn.Conv1d(128, num_obj*4, 1)
   
        self.conv1_t = nn.Conv1d(1536+512, 1024, 1)
        self.conv2_t = nn.Conv1d(1024, 512, 1)
        self.conv3_t = nn.Conv1d(512, 128, 1)
        self.conv4_t = nn.Conv1d(128, num_obj*3, 1) 

        self.conv1_a = nn.Conv1d(1536, 1024, 1)
        self.conv2_a = nn.Conv1d(1024, 512, 1)
        self.conv3_a = nn.Conv1d(512, 128, 1)
        self.conv4_a = nn.Conv1d(128, num_obj*3, 1)
   
        self.conv1_b = nn.Conv1d(1536, 1024, 1)
        self.conv2_b = nn.Conv1d(1024, 512, 1)
        self.conv3_b = nn.Conv1d(512, 128, 1)
        self.conv4_b = nn.Conv1d(128, num_obj*3, 1) 
        
        '---------------------------------------------------------------------------------'

        self.maxpool = nn.MaxPool1d(num_points)
        self.avgpool = nn.AvgPool1d(num_points)

    def forward(self, img, x, n, choose, cls, cad):
        out_img = self.cnn(img)# (B, 32, h, w)
        bs, di, _, _ = out_img.size()
        emb = out_img.view(bs, di, -1)# (B, 32, h*w)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()# (B, 32, N)

        x = torch.cat([x, n], dim=2)# (B, N, 6)
        obsfeat = self.feat(x.transpose(2, 1).contiguous(), emb)# (B, 512, N)

        avg1 = F.relu(self.conv1_avg(obsfeat))
        avg2 = F.relu(self.conv2_avg(avg1))
        globalfeat1 = self.avgpool(avg2).repeat(1, 1, self.num_points)# (B, 1024, N)

        avg3 = F.relu(self.conv3_avg(obsfeat))
        avg4 = F.relu(self.conv4_avg(avg3))
        posefeat = self.avgpool(avg4).repeat(1, 1, self.num_points)# (B, 1024, N)

        '---------------------------------------------------------------------------------'  
        
        cadfeat = self.modelfeat(cad.transpose(2, 1).contiguous())# (B, 512, N)
        avg5 = F.relu(self.conv5_avg(cadfeat))
        avg6 = F.relu(self.conv6_avg(avg5))        
        globalfeat2 = self.avgpool(avg6).repeat(1, 1, self.num_points)# (B, 512, N)

        '---------------------------------------------------------------------------------'  

        pointfeat1 = torch.cat([obsfeat, globalfeat1], 1)# 512+1024
        pointfeat2 = torch.cat([obsfeat, globalfeat2, posefeat], 1)# 512+512+1024
        pointfeat3 = torch.cat([cadfeat, posefeat], 1)# 512+1024

        '---------------------------------------------------------------------------------'  

        mx = F.relu(self.conv1_m(pointfeat1))
        mx = F.relu(self.conv2_m(mx))
        mx = F.relu(self.conv3_m(mx))      
        out_mx = self.conv4_m(mx).view(bs, self.num_obj, 3, -1)# (B, num_obj, 3, N)
        
        nx = F.relu(self.conv1_n(pointfeat1))
        nx = F.relu(self.conv2_n(nx))
        nx = F.relu(self.conv3_n(nx))
        out_nx = self.conv4_n(nx).view(bs, self.num_obj, 3, -1)# (B, num_obj, 3, N)
        out_nx = out_nx / torch.norm(out_nx, dim=2, keepdim=True)

        rx = F.relu(self.conv1_r(pointfeat2))
        rx = F.relu(self.conv2_r(rx))
        rx = F.relu(self.conv3_r(rx))
        out_rx = self.conv4_r(rx).view(bs, self.num_obj, 4, -1)# (B, num_obj, 4, N)
        out_rx = out_rx / torch.norm(out_rx, dim=2, keepdim=True)

        tx = F.relu(self.conv1_t(pointfeat2))
        tx = F.relu(self.conv2_t(tx))
        tx = F.relu(self.conv3_t(tx))
        out_tx = self.conv4_t(tx).view(bs, self.num_obj, 3, -1)# (B, num_obj, 3, N)

        ax = F.relu(self.conv1_a(pointfeat3))
        ax = F.relu(self.conv2_a(ax))
        ax = F.relu(self.conv3_a(ax))
        out_ax = self.conv4_a(ax).view(bs, self.num_obj, 3, -1)# (B, num_obj, 3, N)
        
        bx = F.relu(self.conv1_b(pointfeat3))
        bx = F.relu(self.conv2_b(bx))
        bx = F.relu(self.conv3_b(bx))
        out_bx = self.conv4_b(bx).view(bs, self.num_obj, 3, -1)# (B, num_obj, 3, N)
        out_bx = out_bx / torch.norm(out_bx, dim=2, keepdim=True)        

        out_rx = out_rx[:, cls[0], ...].transpose(2, 1).contiguous()# (B, N, 4)
        out_tx = out_tx[:, cls[0], ...].transpose(2, 1).contiguous()# (B, N, 3)
        out_mx = out_mx[:, cls[0], ...].transpose(2, 1).contiguous()# (B, N, 3)
        out_nx = out_nx[:, cls[0], ...].transpose(2, 1).contiguous()# (B, N, 3)
        out_ax = out_ax[:, cls[0], ...].transpose(2, 1).contiguous()# (B, N, 3)
        out_bx = out_bx[:, cls[0], ...].transpose(2, 1).contiguous()# (B, N, 3)
        return out_rx, out_tx, out_mx, out_nx, out_ax, out_bx
    