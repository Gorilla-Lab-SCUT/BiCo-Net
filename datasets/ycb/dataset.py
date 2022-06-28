import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import random
import numpy.ma as ma
import scipy.io as scio
import open3d as o3d

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root):
        if mode == 'train':
            self.path = 'datasets/ycb/dataset_config/train_data_list.txt'
        elif mode == 'test':
            self.path = 'datasets/ycb/dataset_config/test_data_list.txt'
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise

        self.list = []
        self.real = []
        self.syn = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(input_line)
            else:
                self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        class_file = open('datasets/ycb/dataset_config/classes.txt')
        self.cld = []
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break        
            points = np.loadtxt('{0}/models/{1}/points.xyz'.format(self.root, class_input[:-1]))
            self.cld.append(points[0:2620, :])
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
            pnc = o3d.io.read_point_cloud('%s/models_rgb/obj_%06d.ply'%(self.root, cad_id))
            pp = np.array(pnc.points)/1000.0# (N, 3)
            pp = pp - pp.mean(0)
            nn = np.array(pnc.normals)# (N, 3)
            cc = np.array(pnc.colors)# (N, 3) normalized
            cad = np.concatenate([pp, nn, cc], axis=1)# (N, 9)
            self.cad.append(cad)
            cad_id += 1
        class_file.close()

        '----------------------------------------------------------------------------'

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]
        self.num_pt_mesh = 500
        self.front_num = 2

        print(len(self.list))

    def __getitem__(self, index):
        img = Image.open('{0}/{1}-color.png'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))

        if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        add_front = False
        if self.add_noise:
            for k in range(5):
                seed = random.choice(self.syn)
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, seed)))
                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                   continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk
                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        while 1:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > self.minimum_num_pt:
                break

        if self.add_noise:
            img = self.trancolor(img)

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        if self.list[index][:8] == 'data_syn':
            seed = random.choice(self.real)
            back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
            back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]# (3, h, w)
            img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img
        else:
            img_masked = img

        if self.add_noise and add_front:
            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])

        if self.list[index][:8] == 'data_syn':
            img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
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

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate([pt0, pt1, pt2], axis=1)

        #centralization
        centroid = np.mean(cloud, axis=0)# (1, 3)
        cloud = cloud - centroid
        target_t = target_t - centroid

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        normal = np.array(pcd.normals)# (N, 3)

        cls = int(obj[idx]) - 1
        model_points = self.cld[cls][np.random.choice(len(self.cld[cls]), self.num_pt_mesh, replace=False)]
        cad = self.cad[cls][np.random.choice(len(self.cad[cls]), self.num_pt, replace=False)]

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



border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
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
