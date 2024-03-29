import argparse
import time

import numpy as np
import numpy.ma as ma
import open3d as o3d
import scipy.io as scio
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from lib.network import PoseNet
from lib.pointpair_matching import ppf_filtering
from lib.rotation import quaternion_to_matrix
from lib.ops import ADDS_Dis, L2_Dis, fps

model = './local_data/ycbv_pose_model_36_0.008871091098015229.pth'
dataset_root = './local_data/YCB_Video_Dataset'
pred_mask = './local_data/results_PVN3D'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default=dataset_root, help='dataset root dir')
parser.add_argument('--pred_mask', type=str, default=pred_mask, help='pred mask dir')
parser.add_argument('--model', type=str, default=model, help='resume PoseNet model')
opt = parser.parse_args()
num_obj = 21
num_points = 1000
num_fps = 100


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, num_points, dataset_root):
        self.dataset_root = dataset_root
        self.num_points = num_points

        self.testlist = []
        frameid = 0
        input_file = open('datasets/ycb/dataset_config/test_data_list.txt')
        print('reading test dataset list...')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            lst = scio.loadmat('{0}/{1}-meta.mat'.format(dataset_root, input_line))['cls_indexes'].flatten()
            for ii in range(len(lst)):
                self.testlist.append({'frameid':frameid, 'frame_name':input_line, 'itemid':lst[ii]})
            frameid += 1
        input_file.close()  
        self.length = len(self.testlist) 
        print(self.length)

        class_file = open('datasets/ycb/dataset_config/classes.txt')
        class_id = 1
        self.cld = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break        
            points = np.loadtxt('{0}/models/{1}/points.xyz'.format(dataset_root, class_input[:-1]))
            self.cld[class_id] = points[0:2620, :]
            class_id += 1
        class_file.close()

        '----------------------------------------------------------------------------'

        class_file = open('datasets/ycb/dataset_config/classes.txt')
        self.cad = []
        cad_id = 1
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            print('loading cad: obj_%06d.ply'%cad_id)
            pnc = o3d.io.read_point_cloud(f'./local_data/models_rgb/obj_{cad_id:06d}.ply')
            pp = np.array(pnc.points)/1000.0# (N, 3)
            pp = pp - pp.mean(0)
            nn = np.array(pnc.normals)# (N, 3)
            cc = np.array(pnc.colors)# (N, 3) normalized
            cad = np.concatenate([pp, nn, cc], axis=1)# (N, 9)
            self.cad.append(cad)
            cad_id += 1
        class_file.close()

        '----------------------------------------------------------------------------'

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.cam_cx = 312.9869
        self.cam_cy = 241.3109
        self.cam_fx = 1066.778
        self.cam_fy = 1067.487
        self.cam_scale = 10000.0
        self.num_pt_mesh = 1000


    def __getitem__(self, index):
        frameid = self.testlist[index]['frameid']
        img = Image.open('{0}/{1}-color.png'.format(self.dataset_root, self.testlist[index]['frame_name']))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.dataset_root, self.testlist[index]['frame_name'])))
        gt_meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.dataset_root, self.testlist[index]['frame_name']))
        lst = gt_meta['cls_indexes'].flatten()        

        pred_meta = scio.loadmat('{0}/{1}.mat'.format(opt.pred_mask, '%06d'%frameid))
        label = np.array(pred_meta['labels'])
        pred_rois = np.array(pred_meta['rois'])

        itemid = int(self.testlist[index]['itemid'])    
        print('processing frame {} instance {}'.format(frameid, itemid))
        try:
            if np.where(pred_rois[:, 1:2].flatten() == itemid)[0].shape[0] != 0:
                rmin, rmax, cmin, cmax = self.get_bbox(pred_rois, np.where(pred_rois[:, 1:2].flatten() == itemid)[0][0])
            else:
                raise ZeroDivisionError

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
            mask = mask_label * mask_depth

            img_masked = np.array(img)[:, :, :3]# (H, W, 3)
            img_masked = img_masked.transpose(2, 0, 1)# (3, H, W)
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]# (3, H, W) -> (3, h, w)
 
            target_r = np.resize(gt_meta['poses'][:, :, np.where(lst==itemid)[0][0]][0:3, 0:3], (3, 3))# (3, 3)
            target_t = np.resize(gt_meta['poses'][:, :, np.where(lst==itemid)[0][0]][0:3, 3], (1, 3))# (1, 3) 

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > self.num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                if len(choose) == 0: 
                    raise ZeroDivisionError
                choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            pt2 = depth_masked / self.cam_scale 
            pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
            pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
            cloud = np.concatenate([pt0, pt1, pt2], axis=1)

            #centralization
            centroid = np.mean(cloud, axis=0)# 1*3
            cloud = cloud - centroid
            target_t = target_t - centroid            

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            normal = np.array(pcd.normals)# n*3

            model_points = self.cld[itemid]

            cls = itemid - 1
            cad = self.cad[cls][np.random.choice(len(self.cad[cls]), self.num_pt_mesh, replace=False)]
            _, sidx_r = fps(cloud, npoint=num_fps)
            _, sidx_i = fps(cloud, npoint=num_fps)
            _, midx_r = fps(cad, npoint=num_fps)
            _, midx_i = fps(cad, npoint=num_fps)
            return torch.from_numpy(cloud.astype(np.float32)),\
                   torch.from_numpy(normal.astype(np.float32)),\
                   torch.LongTensor(choose.astype(np.int32)),\
                   self.norm(torch.from_numpy(img_masked.astype(np.float32))),\
                   torch.from_numpy(model_points.astype(np.float32)), \
                   torch.from_numpy(target_r.astype(np.float32)),\
                   torch.from_numpy(target_t.astype(np.float32)),\
                   cls, \
                   torch.from_numpy(cad.astype(np.float32)),\
                   torch.LongTensor(sidx_r.astype(np.int32)),\
                   torch.LongTensor(sidx_i.astype(np.int32)),\
                   torch.LongTensor(midx_r.astype(np.int32)),\
                   torch.LongTensor(midx_i.astype(np.int32)),\
                   True
        
        except ZeroDivisionError:
            print("Detector Lost {0} at frame {1}".format(itemid, frameid)) 
            return torch.from_numpy(np.zeros((1000, 3)).astype(np.float32)),\
                   torch.from_numpy(np.zeros((1000, 3)).astype(np.float32)),\
                   torch.LongTensor(np.zeros((1, 1000)).astype(np.int32)),\
                   torch.from_numpy(np.zeros((3, 480, 640)).astype(np.float32)),\
                   torch.from_numpy(np.zeros((2620, 3)).astype(np.float32)),\
                   torch.from_numpy(np.zeros((3, 3)).astype(np.float32)),\
                   torch.from_numpy(np.zeros((1, 3)).astype(np.float32)),\
                   itemid - 1, \
                   torch.from_numpy(np.zeros((1000, 9)).astype(np.float32)),\
                   torch.LongTensor(np.arange(num_fps).astype(np.int32)),\
                   torch.LongTensor(np.arange(num_fps).astype(np.int32)),\
                   torch.LongTensor(np.arange(num_fps).astype(np.int32)),\
                   torch.LongTensor(np.arange(num_fps).astype(np.int32)),\
                   False


    def __len__(self):
        return self.length


    def get_bbox(self, pred_rois, idx):
        border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        img_width, img_length = 480, 640
        rmin = np.max([int(pred_rois[idx][3])+1, 0])
        rmax = np.min([int(pred_rois[idx][5])-1, img_width])
        cmin = np.max([int(pred_rois[idx][2])+1, 0])
        cmax = np.min([int(pred_rois[idx][4])-1, img_length])
        
        r_b = rmax - rmin
        for tt in range(len(border_list)):
            if r_b > border_list[tt] and r_b < border_list[tt + 1]:
                r_b = border_list[tt + 1]
                break
        c_b = cmax - cmin
        for tt in range(len(border_list)):
            if c_b > border_list[tt] and c_b < border_list[tt + 1]:
                c_b = border_list[tt + 1]
                break
        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
        rmin = center[0] - int(r_b / 2)
        rmax = center[0] + int(r_b / 2)
        cmin = center[1] - int(c_b / 2)
        cmax = center[1] + int(c_b / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > img_width:
            delt = rmax - img_width
            rmax = img_width
            rmin -= delt
        if cmax > img_length:
            delt = cmax - img_length
            cmax = img_length
            cmin -= delt
        return rmin, rmax, cmin, cmax


def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap


def cal_auc(dis_list, max_dis=0.1):
    D = np.array(dis_list)
    D[np.where(D > max_dis)] = np.inf
    D = np.sort(D)
    n = len(dis_list)
    acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
    aps = VOCap(D, acc)
    return aps * 100


def cal_metric(ADDS_list, ADD_S_list, idx_list):
    ADDS_list = np.array(ADDS_list)
    ADD_S_list = np.array(ADD_S_list)
    idx_list = np.array(idx_list)
    ADDS_auc_list = []
    ADDS_2cm_list = []
    for idx in range(21):
        ADDS_list_item = ADDS_list[np.where(idx_list==idx)]
        ADDS_auc_item = cal_auc(ADDS_list_item)
        ADDS_auc_list.append(ADDS_auc_item)
        ADDS_2cm_item = round((len(ADDS_list_item[ADDS_list_item <= 0.02]) / len(ADDS_list_item))*100, 2)
        ADDS_2cm_list.append(ADDS_2cm_item)
        print('NO.{0} | ADDS_AUC:{1} | ADDS_2cm:{2}'.format('%02d'%(idx+1), '%3.2f'%ADDS_auc_item, '%3.2f'%ADDS_2cm_item))
    ADDS_auc = cal_auc(ADDS_list)
    ADDS_2cm = round((len(ADDS_list[ADDS_list <= 0.02]) / len(ADDS_list))*100, 2)
    print('ALL   | ADDS_AUC:{0} | ADDS_2cm:{1}'.format('%3.2f'%ADDS_auc, '%3.2f'%ADDS_2cm))
    return  ADDS_auc, ADDS_2cm



if __name__ == '__main__':

    sym_list = [12, 15, 18, 19, 20]

    estimator = PoseNet(num_points = num_points, num_obj = num_obj).cuda()
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()

    testset = TestDataset(num_points, opt.dataset_root)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    t1 = time.time()
    with torch.no_grad():
        ADDS_list = []
        ADD_S_list = []
        cls_list = []
        for data in testloader:
            points, normal, choose, img, model_points, target_r, target_t, cls, cad, sidx_r, sidx_i, midx_r, midx_i, flag = data
            points, normal, choose, img, model_points, target_r, target_t, cad, sidx_r, sidx_i, midx_r, midx_i = \
                                                              Variable(points).cuda(),\
                                                              Variable(normal).cuda(),\
                                                              Variable(choose).cuda(),\
                                                              Variable(img).cuda(),\
                                                              Variable(model_points).cuda(),\
                                                              Variable(target_r).cuda(),\
                                                              Variable(target_t).cuda(),\
                                                              Variable(cad).cuda(),\
                                                              Variable(sidx_r).cuda(),\
                                                              Variable(sidx_i).cuda(),\
                                                              Variable(midx_r).cuda(),\
                                                              Variable(midx_i).cuda()  
            out_rx, out_tx, out_mx, out_nx, out_ax, out_bx = estimator(img, points, normal, choose, cls, cad)

            bs, num_p, _ = out_rx.size()
            out_Rx1 = quaternion_to_matrix(out_rx).squeeze().contiguous()
            out_tx1 = (points + out_tx).view(bs*num_p, 1, 3).contiguous()

            '----------------------------------------------------------------------------------------------'
            out_Rx2, out_tx2 = ppf_filtering(points[0], normal[0], out_mx[0], out_nx[0], num_fps, sidx_r[0], sidx_i[0])
            '----------------------------------------------------------------------------------------------'
            out_Rx3, out_tx3 = ppf_filtering(out_ax[0], out_bx[0], cad[0][:, 0:3], cad[0][:, 3:6], num_fps, midx_r[0], midx_i[0])
            '----------------------------------------------------------------------------------------------'

            out_R = torch.mean(torch.cat([out_Rx1, out_Rx2, out_Rx3], dim=0), dim=0, keepdim=True)# (1, 3, 3)
            out_t = torch.mean(torch.cat([out_tx1, out_tx2, out_tx3], dim=0), dim=0, keepdim=True)# (1, 1, 3)         

            target = torch.bmm(model_points, target_r.transpose(2, 1)) + target_t # (1, 2620, 3)
            pred = torch.bmm(model_points, out_R.transpose(2, 1)) + out_t # (1, 2620, 3)

            if flag: 
                unsym_dis = L2_Dis(pred, target)# (1, 2620, 3) -> 1
                sym_dis = ADDS_Dis(pred, target)# (1, 2620, 3) -> 1
            else: 
                unsym_dis = torch.tensor([np.Inf])
                sym_dis = torch.tensor([np.Inf])
            
            ADDS_list.append(sym_dis.item())
            ADD_S_list.append(sym_dis.item() if cls in sym_list else unsym_dis.item())
            cls_list.append(cls.item())

        t2 = time.time()        
        print('load time {0}'.format(t2-t1))  
        ADDS_auc, ADDS_2cm = cal_metric(ADDS_list, ADD_S_list, cls_list)
