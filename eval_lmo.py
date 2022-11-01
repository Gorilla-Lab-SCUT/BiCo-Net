import argparse
import time

import cv2
import numpy as np
import numpy.ma as ma
import open3d as o3d
import torch
import torch.utils.data
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.autograd import Variable

from lib.knn.__init__ import KNearestNeighbor
from lib.network import PoseNet
from lib.pointpair_matching import ppf_filtering
from lib.rotation import quaternion_to_matrix

model = './local_data/lmo_pose_model_27_0.02377143515188488.pth'
dataset_root = './local_data/Linemod_preprocessed'
pred_mask = './local_data/mask_hybridpose'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default=dataset_root, help='dataset root dir')
parser.add_argument('--model', type=str, default=model, help='resume PoseNet model')
parser.add_argument('--pred_mask', type=str, default=pred_mask, help='pred mask dir')
opt = parser.parse_args()
num_obj = 8
num_points = 1000
num_fps = 100

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, num_pt, root, pred_mask):
        self.num_pt = num_pt
        self.root = root
        self.pred_mask = pred_mask
        self.objlist = [1, 5, 6, 8, 9, 10, 11, 12]
        self.seglist = [21, 106, 128, 170, 191, 213, 234, 255]
        meta_file = yaml.load(open('{0}/data/02/gt.yml'.format(root), 'r'), Loader=yaml.FullLoader)

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.meta = []
        for index in range(1214):
            img_meta = meta_file[index]
            for item_meta in img_meta:
                obj = item_meta['obj_id']
                if obj == 2: 
                    continue
                self.list_rgb.append('{0}/data/02/rgb/{1}.png'.format(root, '%04d'%index))
                self.list_depth.append('{0}/data/02/depth/{1}.png'.format(root, '%04d'%index))
                self.list_label.append('{0}/{1}/{2}.png'.format(self.pred_mask, obj, '%04d'%index))
                self.list_obj.append(obj)
                self.meta.append(item_meta)

        self.cad = {}
        for obj in self.objlist:
            pnc = o3d.io.read_point_cloud('{0}/models/obj_{1}.ply'.format(root, '%02d'%obj))
            pp = np.array(pnc.points)/1000.0
            nn = np.array(pnc.normals)
            cc = np.array(pnc.colors)
            self.cad[obj] = np.concatenate([pp, nn, cc], axis=1)
    
        self.length = len(self.list_rgb)

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.symmetry_obj_idx = [5, 6]    
        self.num_pt_mesh = 500

    def mask_to_bbox(self, mask, padding):
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = 0, 0, 0, 0
        for contour in contours:
            tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
            if tmp_w * tmp_h > w * h:
                x = tmp_x - int(padding/2)
                y = tmp_y - int(padding/2)
                w = tmp_w + padding
                h = tmp_h + padding
        return [x, y, w, h]


    def __getitem__(self, index):
        img = np.array(Image.open(self.list_rgb[index]))[:, :, :3]
        depth = np.array(Image.open(self.list_depth[index]))

        obj = self.list_obj[index]        
        try:
            mask = np.array(Image.open(self.list_label[index]))
        except:
            return self.objlist.index(obj)
        meta = self.meta[index]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(mask, 1))
        mask = mask_label * mask_depth

        img_masked = img
        img_masked = np.transpose(img_masked, (2, 0, 1))# H*W*3 -> 3*H*W
        rmin, rmax, cmin, cmax = get_bbox(self.mask_to_bbox(mask_label, padding=0))

        img_masked = img_masked[:, rmin:rmax, cmin:cmax]# 3*H*W -> 3*h*w
    
        target_r = np.array(meta['cam_R_m2c']).reshape((3, 3))
        target_t = np.array(meta['cam_t_m2c']).reshape((1, 3))/1000.0

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            return self.objlist.index(obj)

        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)/1000.0        

        #centralization
        centroid = np.mean(cloud, axis=0).reshape((1, 3))
        cloud = cloud - centroid
        target_t = target_t - centroid

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        normal = np.array(pcd.normals)

        model_points = self.cad[obj][:, 0:3]
        if self.num_pt == self.num_pt_mesh: cad = model_points
        else: cad = self.cad[obj][np.random.choice(len(self.cad[obj]), self.num_pt, replace=False)]

        cls = self.objlist.index(obj)
        _, sidx_r = fps(cloud, npoint=100)
        _, sidx_i = fps(cloud, npoint=100)
        _, midx_r = fps(cad, npoint=100)
        _, midx_i = fps(cad, npoint=100)
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.from_numpy(normal.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               torch.from_numpy(img_masked.astype(np.float32)), \
               torch.from_numpy(cad.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.from_numpy(target_r.astype(np.float32)), \
               torch.from_numpy(target_t.astype(np.float32)), \
               torch.LongTensor(sidx_r.astype(np.int32)),\
               torch.LongTensor(sidx_i.astype(np.int32)),\
               torch.LongTensor(midx_r.astype(np.int32)),\
               torch.LongTensor(midx_i.astype(np.int32)),\
               cls

    def __len__(self):
        return self.length

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width, img_length = 480, 640
def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
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
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax

def fps(point, npoint):
    N, D = point.shape
    if N < npoint:
        idx = np.random.choice(np.arange(N), npoint-N)
        return np.concatenate([point, point[idx]], axis=0), \
            np.concatenate([np.arange(N), idx], axis=0)
    xyz = point[:, :3]
    centroids = np.zeros((npoint,)).astype(np.int32)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids]
    return point, centroids

def L2_Dis(pred, target):
    return torch.norm(pred - target, dim=2).mean(1)
    
def ADDS_Dis(pred, target):
    knn = KNearestNeighbor(1)
    pred = pred.permute(0, 2, 1).contiguous()
    target = target.permute(0, 2, 1).contiguous()
    inds = knn.forward(target, pred)
    target = torch.gather(target, 2, inds.repeat(1, 3, 1) - 1)
    pred = pred.permute(0, 2, 1).contiguous()
    target = target.permute(0, 2, 1).contiguous()
    del knn
    return torch.norm(pred - target, dim=2).mean(1)

if __name__ == '__main__':

    sym_list = [5, 6]

    estimator = PoseNet(num_points = num_points, num_obj = num_obj).cuda()
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()    

    testset = TestDataset(num_points, opt.dataset_root, opt.pred_mask)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    diameter = []
    meta_file = open('%s/models/models_info.yml'%opt.dataset_root, 'r')
    meta = yaml.load(meta_file, Loader=yaml.FullLoader)
    for obj in [1, 5, 6, 8, 9, 10, 11, 12]:
        diameter.append(meta[obj]['diameter'] / 1000.0)  

    st_time = time.time()
    with torch.no_grad():
        dis_list = []
        cls_list = []
        test_dis, test_count = 0, 0
        for j, data in enumerate(testloader, 0):
            if len(data) == 1: 
                dis_list.append(np.inf)       
                cls_list.append(data[0])
                continue
            points, normal, choose, img, cad, model_points, target_r, target_t, sidx_r, sidx_i, midx_r, midx_i, cls = data
            points, normal, choose, img, cad, model_points, target_r, target_t, sidx_r, sidx_i, midx_r, midx_i = \
                                                                Variable(points).cuda(), \
                                                                Variable(normal).cuda(), \
                                                                Variable(choose).cuda(), \
                                                                Variable(img).cuda(), \
                                                                Variable(cad).cuda(), \
                                                                Variable(model_points).cuda(), \
                                                                Variable(target_r).cuda(), \
                                                                Variable(target_t).cuda(), \
                                                                Variable(sidx_r).cuda(),\
                                                                Variable(sidx_i).cuda(),\
                                                                Variable(midx_r).cuda(),\
                                                                Variable(midx_i).cuda() 
            out_rx, out_tx, out_mx, out_nx, out_ax, out_bx = estimator(img, points, normal, choose, cls, cad)

            bs, num_p, _ = out_rx.size()
            out_Rx1 = quaternion_to_matrix(out_rx).squeeze().contiguous()
            out_tx1 = (points + out_tx).view(bs*num_p, 1, 3).contiguous()
            '----------------------------------------------------------------------------------------------'
            out_Rx2, out_tx2 = ppf_filtering(points[0], normal[0], out_mx[0], out_nx[0], num_fps, sidx_r[0], sidx_i[0])  # 0.0335s
            '----------------------------------------------------------------------------------------------'
            out_Rx3, out_tx3 = ppf_filtering(out_ax[0], out_bx[0], cad[0][:, 0:3], cad[0][:, 3:6], num_fps, midx_r[0], midx_i[0])
            '----------------------------------------------------------------------------------------------'

            out_R = torch.mean(torch.cat([out_Rx1, out_Rx2, out_Rx3], dim=0), dim=0, keepdim=True)# (1, 3, 3)
            out_t = torch.mean(torch.cat([out_tx1, out_tx2, out_tx3], dim=0), dim=0, keepdim=True)# (1, 1, 3)         

            target = torch.bmm(model_points, target_r.transpose(2, 1)) + target_t # (1, 2620, 3)
            pred = torch.bmm(model_points, out_R.transpose(2, 1)) + out_t # (1, 2620, 3)

            unsym_dis = L2_Dis(pred, target)# (1, 2620, 3) -> 1
            sym_dis = ADDS_Dis(pred, target)# (1, 2620, 3) -> 1
            dis = sym_dis if cls[0] in sym_list else unsym_dis

            dis_list.append(dis.item())       
            cls_list.append(cls.item())

            test_dis += dis.item()        
            test_count += 1
            print('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis.item()))

    test_dis = test_dis / test_count     
    print('Test time {0} TEST FINISH Avg dis: {1}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_dis))     

    #calcuate 0.1ADD
    dis_list = np.array(dis_list)
    cls_list = np.array(cls_list)
    n = len(cls_list)        
    add_flag = np.zeros(n)
    for i in range(0, n):
        obj_id = cls_list[i]
        if dis_list[i] <= 0.1*diameter[obj_id]:
            add_flag[i] = 1
    acc = round((np.sum(add_flag)/n)*100, 2)
    print('ADD(S) Result:', acc)
