import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet
from lib.loss import Loss
from lib.utils import setup_logger, load_json
from os.path import join
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = '')
parser.add_argument('--dataset_root', type=str, default = '')
parser.add_argument('--batch_size', type=int, default = 8)
parser.add_argument('--workers', type=int, default = 8)
parser.add_argument('--lr', default = 0.0001)
parser.add_argument('--lr_decay', default = [1, 0.3, 0.1, 0.03, 0.01])
parser.add_argument('--lr_steps', default = [10, 20, 30, 40])
parser.add_argument('--nepoch', type=int, default = 50)
parser.add_argument('--resume_posenet', type=str, default = '')
opt = parser.parse_args()

def get_learning_rate(epoch):
    assert len(opt.lr_decay) == len(opt.lr_steps) + 1
    for step, decay in zip(opt.lr_steps, opt.lr_decay):
        if epoch < step:
            return decay * opt.lr
    return opt.lr_decay[-1] * opt.lr

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycbv':
        opt.num_objects = 21 
        opt.num_points = 1000 
        outdir = './experiments/ycbv/trained_models'
        rundir = './experiments/ycbv/runs'
        logdir = './experiments/ycbv/logs'
        opt.repeat_epoch = 1 
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        outdir = './experiments/linemod/trained_models'
        rundir = './experiments/linemod/runs'
        logdir = './experiments/linemod/logs'
        opt.repeat_epoch = 20

    if not os.path.exists(outdir):
        os.makedirs(outdir)
        os.makedirs(rundir)
        os.makedirs(logdir)

    if opt.dataset == 'ycbv': dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root)
    elif opt.dataset == 'linemod': dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers, pin_memory=True)

    if opt.dataset == 'ycbv': test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root)
    elif opt.dataset == 'linemod': test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers, pin_memory=True)

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<')
    print('length of the training set: {}'.format(len(dataset)))
    print('length of the testing set: {}'.format(len(test_dataset)))
    print('number of sample points on mesh: {}'.format(dataset.get_num_points_mesh()))
    print('symmetry object list: {}'.format(dataset.get_sym_list()))

    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects).cuda()
    if opt.resume_posenet != '': estimator.load_state_dict(torch.load('{0}/{1}'.format(outdir, opt.resume_posenet)))
    criterion = Loss(dataset.get_sym_list())
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
    writer = SummaryWriter(logdir=rundir)    

    best_test = np.Inf
    st_time = time.time()
    iteration_count = 0
    for epoch in range(0, opt.nepoch):
        lr = get_learning_rate(epoch)
        for p in optimizer.param_groups: p['lr'] = lr

        logger = setup_logger('epoch%d' % epoch, os.path.join(logdir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0       
        estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                points, normal, choose, img, cad, model_points, target_r, target_t, cls = data
                points, normal, choose, img, cad, model_points, target_r, target_t = \
                                                                 Variable(points).cuda(), \
                                                                 Variable(normal).cuda(), \
                                                                 Variable(choose).cuda(), \
                                                                 Variable(img).cuda(), \
                                                                 Variable(cad).cuda(), \
                                                                 Variable(model_points).cuda(), \
                                                                 Variable(target_r).cuda(), \
                                                                 Variable(target_t).cuda()                                                                                                                
                out_rx, out_tx, out_mx, out_nx, out_ax, out_bx = estimator(img, points, normal, choose, cls, cad)
                loss, loss_pose, loss_aux1, loss_aux2, loss_aux3, loss_aux4 = \
                    criterion(out_rx, out_tx, out_mx, out_nx, out_ax, out_bx, cad, model_points, points, normal, target_r, target_t, cls, epoch)
                loss.backward()                
                
                train_dis_avg += loss_pose.item()
                train_count += 1
                iteration_count += 1                

                if train_count % opt.batch_size == 0:
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", \
                        time.gmtime(time.time()-st_time)), epoch, int(train_count/opt.batch_size), train_count, train_dis_avg/opt.batch_size))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0

                if iteration_count % 100 == 0:
                    writer.add_scalar('train/loss_pose', loss_pose.item(), iteration_count)
                    writer.add_scalar('train/loss_aux1', loss_aux1.item(), iteration_count)
                    writer.add_scalar('train/loss_aux2', loss_aux2.item(), iteration_count)
                    writer.add_scalar('train/loss_aux3', loss_aux3.item(), iteration_count)
                    writer.add_scalar('train/loss_aux4', loss_aux4.item(), iteration_count)

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        logger = setup_logger('epoch%d_test' % epoch, os.path.join(logdir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis, aux1_dis, aux2_dis, aux3_dis, aux4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
        test_count = 0
        estimator.eval()

        pts_dis = []
        result_obj_id = []
        for j, data in enumerate(testdataloader, 0):
            points, normal, choose, img, cad, model_points, target_r, target_t, cls = data
            points, normal, choose, img, cad, model_points, target_r, target_t = \
                                                                Variable(points).cuda(), \
                                                                Variable(normal).cuda(), \
                                                                Variable(choose).cuda(), \
                                                                Variable(img).cuda(), \
                                                                Variable(cad).cuda(), \
                                                                Variable(model_points).cuda(), \
                                                                Variable(target_r).cuda(), \
                                                                Variable(target_t).cuda()                                                                                                                
            out_rx, out_tx, out_mx, out_nx, out_ax, out_bx = estimator(img, points, normal, choose, cls, cad)
            loss, loss_pose, loss_aux1, loss_aux2, loss_aux3, loss_aux4 = \
                criterion(out_rx, out_tx, out_mx, out_nx, out_ax, out_bx, cad, model_points, points, normal, target_r, target_t, cls, epoch)
            pts_dis.append(loss_pose.item())       
            result_obj_id.append(cls.item())

            test_dis += loss_pose.item()
            aux1_dis += loss_aux1.item()
            aux2_dis += loss_aux2.item()
            aux3_dis += loss_aux3.item()
            aux4_dis += loss_aux4.item()

            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, loss_pose.item()))
            test_count += 1

        test_dis = test_dis / test_count
        aux1_dis = aux1_dis / test_count
        aux2_dis = aux2_dis / test_count
        aux3_dis = aux3_dis / test_count
        aux4_dis = aux4_dis / test_count        
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        writer.add_scalar('test/loss_pose', test_dis, epoch)
        writer.add_scalar('test/loss_aux1', aux1_dis, epoch)
        writer.add_scalar('test/loss_aux2', aux2_dis, epoch)
        writer.add_scalar('test/loss_aux3', aux3_dis, epoch)
        writer.add_scalar('test/loss_aux4', aux4_dis, epoch)
        
        if test_dis <= best_test:
            best_test = test_dis
            torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(outdir, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

if __name__ == '__main__':
    main()
