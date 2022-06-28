import argparse
import numpy as np
from PIL import Image
import scipy.io as scio
import numpy.ma as ma
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
from lib.network import PoseNet
from lib.knn.__init__ import KNearestNeighbor
from lib.pointpair_matching import compute_pose
import open3d as o3d

model = './experiments/ycbv/pretrained_models/pose_model_36_0.008871091098015229.pth'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='', help='dataset root dir')
parser.add_argument('--pred_mask', type=str, default='', help='pred mask dir')
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
            pnc = o3d.io.read_point_cloud('%s/models_rgb/obj_%06d.ply'%(dataset_root, cad_id))
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
                rmin, rmax, cmin, cmax = get_bbox(pred_rois, np.where(pred_rois[:, 1:2].flatten() == itemid)[0][0])
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

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width, img_length = 480, 640
def get_bbox(pred_rois, idx):
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
    print('ALL   | ADDS_AUC:{0} |  ADDS_2cm:{1}'.format('%3.2f'%ADDS_auc, '%3.2f'%ADDS_2cm))
    return  ADDS_auc, ADDS_2cm

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
            out_Rx1 = torch.cat(((1.0 - 2.0 * (out_rx[:, :, 2] ** 2 + out_rx[:, :, 3] ** 2)).view(bs, num_p, 1), \
                    (2.0 * out_rx[:, :, 1] * out_rx[:, :, 2] - 2.0 * out_rx[:, :, 0] * out_rx[:, :, 3]).view(bs, num_p, 1), \
                    (2.0 * out_rx[:, :, 0] * out_rx[:, :, 2] + 2.0 * out_rx[:, :, 1] * out_rx[:, :, 3]).view(bs, num_p, 1), \
                    (2.0 * out_rx[:, :, 1] * out_rx[:, :, 2] + 2.0 * out_rx[:, :, 3] * out_rx[:, :, 0]).view(bs, num_p, 1), \
                    (1.0 - 2.0 * (out_rx[:, :, 1] ** 2 + out_rx[:, :, 3] ** 2)).view(bs, num_p, 1), \
                    (-2.0 * out_rx[:, :, 0] * out_rx[:, :, 1] + 2.0 * out_rx[:, :, 2] * out_rx[:, :, 3]).view(bs, num_p, 1), \
                    (-2.0 * out_rx[:, :, 0] * out_rx[:, :, 2] + 2.0 * out_rx[:, :, 1] * out_rx[:, :, 3]).view(bs, num_p, 1), \
                    (2.0 * out_rx[:, :, 0] * out_rx[:, :, 1] + 2.0 * out_rx[:, :, 2] * out_rx[:, :, 3]).view(bs, num_p, 1), \
                    (1.0 - 2.0 * (out_rx[:, :, 1] ** 2 + out_rx[:, :, 2] ** 2)).view(bs, num_p, 1)), dim=2).view(bs*num_p, 3, 3).contiguous()
            out_tx1 = (points + out_tx).view(bs*num_p, 1, 3).contiguous()

            '----------------------------------------------------------------------------------------------'
            
            scene_coord = points[0]# (1000, 3)
            scene_normal = normal[0]# (1000, 3)
            scene_point = torch.cat([scene_coord, scene_normal], dim=1)# (1000, 6)
            model_coord_pd = out_mx[0]# (1000, 3)
            model_normal_pd = out_nx[0]# (1000, 3)
            model_point_pd = torch.cat([model_coord_pd, model_normal_pd], dim=1)# (1000, 6)

            sidx_r = sidx_r[0][:, None].repeat(1, num_fps).flatten()
            sidx_i = sidx_i[0][None, :].repeat(num_fps, 1).flatten()
            valid_idx = (sidx_r != sidx_i)
            sr = scene_point[sidx_r]
            si = scene_point[sidx_i]
            mr_pd = model_point_pd[sidx_r]
            mi_pd = model_point_pd[sidx_i]
            Tpose_pd = compute_pose(sr, si, mr_pd, mi_pd)
            pd_R = Tpose_pd[:, 0:3, 0:3][valid_idx].contiguous()# (9900, 3, 3)
            pd_t = Tpose_pd[:, 0:3, 3][valid_idx, None, :].contiguous()# (9900, 1, 3)

            N = pd_R.shape[0]
            matching = torch.bmm(points.repeat(N, 1, 1) - pd_t, pd_R)# (9900, 1000, 3)
            dis_mx = L2_Dis(matching, out_mx)# (9900, )
            dis_mnx = dis_mx
            idx_mnx = torch.sort(dis_mnx)[1]
            out_Rx2 = pd_R[idx_mnx][0:int(0.1*N)]
            out_tx2 = pd_t[idx_mnx][0:int(0.1*N)]
            '----------------------------------------------------------------------------------------------'

            scene_coord = out_ax[0]# (1000, 3)
            scene_normal = out_bx[0]# (1000, 3)
            scene_point = torch.cat([scene_coord, scene_normal], dim=1)# (1000, 6)
            model_coord_pd = cad[0][:, 0:3]# (1000, 3)
            model_normal_pd = cad[0][:, 3:6]# (1000, 3)
            model_point_pd = torch.cat([model_coord_pd, model_normal_pd], dim=1)# (1000, 6)

            midx_r = midx_r[0][:, None].repeat(1, num_fps).flatten()
            midx_i = midx_i[0][None, :].repeat(num_fps, 1).flatten()
            valid_idx = (midx_r != midx_i)
            sr = scene_point[midx_r]
            si = scene_point[midx_i]
            mr_pd = model_point_pd[midx_r]
            mi_pd = model_point_pd[midx_i]
            Tpose_pd = compute_pose(sr, si, mr_pd, mi_pd)
            pd_R = Tpose_pd[:, 0:3, 0:3][valid_idx].contiguous()# (9900, 3, 3)
            pd_t = Tpose_pd[:, 0:3, 3][valid_idx, None, :].contiguous()# (9900, 1, 3)

            N = pd_R.shape[0]
            matching = torch.bmm(out_ax.repeat(N, 1, 1) - pd_t, pd_R)# (9900, 1000, 3)
            dis_mx = L2_Dis(matching, cad[:, :, 0:3])# (9900, )
            dis_mnx = dis_mx
            idx_mnx = torch.sort(dis_mnx)[1]
            out_Rx3 = pd_R[idx_mnx][0:int(0.1*N)]
            out_tx3 = pd_t[idx_mnx][0:int(0.1*N)]
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