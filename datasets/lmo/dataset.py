import random

import numpy as np
import numpy.ma as ma
import open3d as o3d
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, bg_img):
        self.objlist = [1, 5, 6, 8, 9, 10, 11, 12]
        self.seglist = [21, 106, 128, 170, 191, 213, 234, 255]
        self.bg_aug = ImageFolder(bg_img)
        self.mode = mode                                            
        self.num_pt = num_pt
        self.add_noise = add_noise

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        if mode == 'train':
            self.list_rank = []
            self.meta = {}
            self.rgb_per_obj = {1:[], 5:[], 6:[], 8:[], 9:[], 10:[], 11:[], 12:[]}
            self.label_per_obj = {1:[], 5:[], 6:[], 8:[], 9:[], 10:[], 11:[], 12:[]}
            obj_loader = tqdm(self.objlist)
            for obj in obj_loader:
                obj_loader.set_postfix_str(f'loading path and meta of object {obj}')
                input_file_train = open('{0}/data/{1}/train.txt'.format(root, '%02d'%obj)).readlines()
                input_file_test = open('{0}/data/{1}/test.txt'.format(root, '%02d'%obj)).readlines()
                input_file = input_file_train + input_file_test
                input_file.sort()
                for input_line in input_file:
                    if not input_line: break
                    self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(root, '%02d'%obj, input_line[:-1]))
                    self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(root, '%02d'%obj, input_line[:-1]))
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(root, '%02d'%obj, input_line[:-1]))           
                    self.list_obj.append(obj)
                    self.list_rank.append(int(input_line[:-1]))
                    self.rgb_per_obj[obj].append('{0}/data/{1}/rgb/{2}.png'.format(root, '%02d'%obj, input_line[:-1]))
                    self.label_per_obj[obj].append('{0}/data/{1}/mask/{2}.png'.format(root, '%02d'%obj, input_line[:-1]))
            
                meta_file = open('{0}/data/{1}/gt.yml'.format(root, '%02d'%obj), 'r')
                self.meta[obj] = yaml.load(meta_file, Loader=yaml.FullLoader)

        elif mode == 'test':
            self.meta = []
            meta_file = yaml.load(open('{0}/data/02/gt.yml'.format(root), 'r'), Loader=yaml.FullLoader) 
            for index in range(1214):
                img_meta = meta_file[index]
                for item_meta in img_meta:
                    obj = item_meta['obj_id']
                    if obj == 2: continue
                    self.list_rgb.append('{0}/data/02/rgb/{1}.png'.format(root, '%04d'%index))
                    self.list_depth.append('{0}/data/02/depth/{1}.png'.format(root, '%04d'%index))
                    self.list_label.append('{0}/data/02/mask_all/{1}.png'.format(root, '%04d'%index))
                    self.list_obj.append(obj)
                    self.meta.append(item_meta)

        self.cad = {}
        for obj in self.objlist:
            pnc = o3d.io.read_point_cloud('{0}/models/obj_{1}.ply'.format(root, '%02d'%obj))
            pp = np.array(pnc.points)/1000.0# N*3
            nn = np.array(pnc.normals)# N*3
            cc = np.array(pnc.colors)# N*3 normalized
            self.cad[obj] = np.concatenate([pp, nn, cc], axis=1)# N*9
    

        self.length = len(self.list_rgb)

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.symmetry_obj_idx = [5, 6] 
        self.num_pt_mesh = 500

    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])
        if self.add_noise: img = self.trancolor(img)
        img = np.array(img)[:, :, :3]        
        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))
        obj = self.list_obj[index]

        if self.mode == 'train': 
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
            meta = self.meta[obj][self.list_rank[index]][0]
        elif self.mode == 'test': 
            mask_label = ma.getmaskarray(ma.masked_equal(label, self.seglist[self.objlist.index(obj)]))
            meta = self.meta[index]

        # background augmentation
        if self.add_noise:
            seed = random.choice(np.arange(len(self.bg_aug)))
            bg_img = np.array(self.trancolor(self.bg_aug[seed][0].resize((640, 480))))
            img_masked = img * mask_label[:, :, None] + bg_img * (~mask_label[:, :, None])
        else:
            img_masked = img

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask = mask_label * mask_depth

        if self.add_noise:
            tmplist = self.objlist.copy()
            tmplist.remove(obj)
            # random add occlusion
            for item in random.sample(tmplist, 3):
                seed = random.choice(range(len(self.rgb_per_obj[item])))
                front_rgb = np.array(self.trancolor(Image.open(self.rgb_per_obj[item][seed])))
                front_label = np.array(Image.open(self.label_per_obj[item][seed]))
                mask_front = ma.getmaskarray(ma.masked_equal(front_label, np.array([255, 255, 255])))
                img_masked = (~mask_front) * img_masked + mask_front * front_rgb
                mask = (~mask_front[:, :, 0]) * mask

        img_masked = np.transpose(img_masked, (2, 0, 1))
        if self.add_noise:
            box_noi = np.random.randint(-10, 10, 4)
        else: 
            box_noi = np.zeros(4).astype(np.int)
        rmin, rmax, cmin, cmax = get_bbox(np.array(meta['obj_bb']) + box_noi)
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
        if self.num_pt_mesh == self.num_pt: model_points = cad[:, 0:3]
        else: model_points = self.cad[obj][np.random.choice(len(self.cad[obj]), self.num_pt_mesh, replace=False)][:, 0:3]

        cls = self.objlist.index(obj)
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.from_numpy(normal.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(cad.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.from_numpy(target_r.astype(np.float32)), \
               torch.from_numpy(target_t.astype(np.float32)), \
               cls

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh


def get_bbox(bbox):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    img_width, img_length = 480, 640
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
