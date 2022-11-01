import numpy as np
import torch

from lib.knn.__init__ import KNearestNeighbor


def fps(point, npoint):
    '''farthest point sampling'''
    N, D = point.shape
    if N < npoint:  # random choose existing point to fill npoints
        idx = np.random.choice(np.arange(N), npoint-N)
        return np.concatenate([point, point[idx]], axis=0), \
            np.concatenate([np.arange(N), idx], axis=0)
    xyz = point[:, :3]
    sampled_idx = np.zeros((npoint,)).astype(np.int32)  # farthest points idx
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)  # inital random select a point idx
    for i in range(npoint):
        sampled_idx[i] = farthest  # an farthest point indx
        centroid = xyz[farthest, :]  # the farthest point coord
        dist = np.sum((xyz - centroid) ** 2, -1)  # cal dis of the farthest point to total point set
        mask = dist < distance       # cal the selected point set distance to total point set
        distance[mask] = dist[mask]  # update distance array, take nearest dis in selected point set as dis
        farthest = np.argmax(distance, -1)
    point = point[sampled_idx]
    return point, sampled_idx


def L2_Dis(pred, target):
    return torch.norm(pred - target, dim=2).mean(1)


knn = KNearestNeighbor(1)

def ADDS_Dis(pred, target):
    pred = pred.permute(0, 2, 1).contiguous()
    target = target.permute(0, 2, 1).contiguous()
    inds = knn.forward(target, pred)
    target = torch.gather(target, 2, inds.repeat(1, 3, 1) - 1)
    pred = pred.permute(0, 2, 1).contiguous()
    target = target.permute(0, 2, 1).contiguous()
    return torch.norm(pred - target, dim=2).mean(1)
