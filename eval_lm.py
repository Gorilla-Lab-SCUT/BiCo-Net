import argparse
import numpy as np
from PIL import Image
import numpy.ma as ma
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
from lib.network import PoseNet
import yaml
from lib.knn.__init__ import KNearestNeighbor
from lib.pointpair_matching import compute_pose
import open3d as o3d
from tqdm import tqdm


model = './experiments/linemod/pretrained_models/pose_model_44_0.005126086624524529.pth'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='', help='dataset root dir')
parser.add_argument('--model', type=str, default=model, help='resume PoseNet model')
opt = parser.parse_args()
num_obj = 13 
num_points = 1000
num_fps = 100

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, num_pt, root):
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]                                          
        self.num_pt = num_pt

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.cad = {}
        obj_loader = tqdm(self.objlist)
        for obj in obj_loader:
            obj_loader.set_postfix_str(f'loading path and meta of object {obj}')
            input_file = open('{0}/data/{1}/test.txt'.format(root, '%02d'%obj))
            while 1:
                input_line = input_file.readline()
                if not input_line: break
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(root, '%02d'%obj, input_line[:-1]))
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(root, '%02d'%obj, input_line[:-1]))
                self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(root, '%02d'%obj, input_line[:-1]))                
                self.list_obj.append(obj)
                self.list_rank.append(int(input_line[:-1]))

            meta_file = open('{0}/data/{1}/gt.yml'.format(root, '%02d'%obj), 'r')
            self.meta[obj] = yaml.load(meta_file,Loader=yaml.FullLoader)# 
            pnc = o3d.io.read_point_cloud('{0}/models/obj_{1}.ply'.format(root, '%02d'%obj))
            pp = np.array(pnc.points)/1000.0# (N, 3)
            nn = np.array(pnc.normals)# (N, 3)
            cc = np.array(pnc.colors)# (N, 3) normalized
            self.cad[obj] = np.concatenate([pp, nn, cc], axis=1)# (N, 9)

        self.length = len(self.list_rgb)

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.symmetry_obj_idx = [7, 8]        
        self.num_pt_mesh = 500

    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])
        img = np.array(img)[:, :, :3]        
        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))        
        obj = self.list_obj[index]
        rank = self.list_rank[index]

        if obj == 2:
            for i in range(0, len(self.meta[obj][rank])):
                if self.meta[obj][rank][i]['obj_id'] == 2:
                    meta = self.meta[obj][rank][i]
                    break
        else: meta = self.meta[obj][rank][0]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))    
        mask = mask_label * mask_depth

        img_masked = np.transpose(img, (2, 0, 1))
        rmin, rmax, cmin, cmax = get_bbox(np.array(meta['obj_bb']))
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]# (3, h, w)

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

        cad = self.cad[obj][np.random.choice(len(self.cad[obj]), self.num_pt, replace=False)]
        model_points = self.cad[obj][:, 0:3]

        cls = self.objlist.index(obj)
        _, sidx_r = fps(cloud, npoint=num_fps)
        _, sidx_i = fps(cloud, npoint=num_fps)
        _, midx_r = fps(cad, npoint=num_fps)
        _, midx_i = fps(cad, npoint=num_fps)
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.from_numpy(normal.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
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

    sym_list = [7, 8]

    estimator = PoseNet(num_points = num_points, num_obj = num_obj).cuda()
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()    

    testset = TestDataset(num_points, opt.dataset_root)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    diameter = []
    meta_file = open('%s/models/models_info.yml'%opt.dataset_root, 'r')
    meta = yaml.load(meta_file, Loader=yaml.FullLoader)
    for obj in [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]:
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