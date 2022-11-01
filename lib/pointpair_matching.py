import numpy as np
import torch

from lib.rotation import axis_angle_to_matrix, matrix_x


def L2_Dis(pred, target):
    return torch.norm(pred - target, dim=2).mean(1)


def Rt2matrix(R, t, inverse=False):
    '''rotation matrix R and translation t to
    homogeneous transfomration matrix'''
    T = torch.zeros((R.size(0), 4, 4)).cuda()
    T[:, 3, 3] = 1.
    if inverse:
        T[:, 0:3, 0:3] = R.permute(0, 2, 1)
        T[:, 0:3, 3] = (-1) * torch.bmm(R.permute(0, 2, 1), t.unsqueeze(2)).squeeze(2)
    else:
        T[:, 0:3, 0:3] = R
        T[:, 0:3, 3] = t
    return T


def transformRT(p):
    '''compute the pose of each oriented point to [0, 0, 0, 1, 0, 0]
    by calculating axis-angle.
    arg:
    p: (N, [x, y, z, nx, ny, nz]), points
    returns:
    R: (N, 3, 3), rotation matrix
    t: (N, 3), translation xyz
    '''
    N = p.shape[0]
    Angle = torch.acos(p[:, 3])  # (N, )  angle of normal and x axis
    Axis = torch.zeros((N, 3)).cuda()
    Axis[:, 1] = p[:, 5]
    Axis[:, 2] = -p[:, 4]   # [0, z, -y]: Axis is orthogonal to normal in yz plane
    idx = (Axis == torch.tensor([0., 0., 0.]).cuda()).all(1).nonzero(as_tuple=False).flatten()
    if len(idx): Axis[idx] = torch.tensor([0., 1., 0.]).cuda()  # filter out full zero
    Axis = Axis / torch.norm(Axis, dim=1, keepdim=True)
    R = axis_angle_to_matrix(Axis, Angle)  # right hand coord system func thus R are inversed
    p_temp = p[:, 0:3].unsqueeze(2)  # (N, 3, 1)
    t = ((-1)*torch.bmm(R, p_temp)).squeeze(2)
    return R, t


def compute_alpha(p1, p2):
    '''comput rotation along x-axis
    args: p1, p2: (N, [x, y, z, nx, ny, nz])
    '''

    R, t = transformRT(p1)
    p2_temp = p2[:, 0:3].unsqueeze(2)
    mpt = (torch.bmm(R, p2_temp) + t.unsqueeze(2)).squeeze(2)
    alpha = torch.atan2(-mpt[:, 2], mpt[:, 1])
    alpha[torch.sin(alpha) * mpt[:, 2] > 0] *= -1
    return alpha


def point_pair_matching(sr, si, mr, mi):
    '''sr, si, mr, mi: (N, [x, y, z, nx, ny, nz])'''
    N = sr.shape[0]
    R, t = transformRT(mr)
    iTcm = Rt2matrix(R, t)

    R, t = transformRT(sr)
    Tcs = Rt2matrix(R, t, inverse=True)

    alphaModel = compute_alpha(mr, mi)
    alphaScene = compute_alpha(sr, si)
    alphaAngle = alphaModel - alphaScene
    alphaAngle[alphaAngle > np.pi] -= 2 * np.pi
    alphaAngle[alphaAngle < -np.pi] += 2 * np.pi
    Talpha = torch.zeros((N, 4, 4)).cuda()
    Talpha[:, 3, 3] = 1.
    Talpha[:, 0:3, 0:3] = matrix_x(alphaAngle)
    Tpose = torch.bmm(torch.bmm(Tcs, Talpha), iTcm)
    return Tpose


def ppf_filtering(scene_xyz, scene_norm, cano_xyz, cano_norm, num_fps, r, i):
    '''
    r, i: a sampled point set indices from the same point set
    '''
    scene_point = torch.cat([scene_xyz, scene_norm], dim=1)  # (1000, 6)
    model_point_pd = torch.cat([cano_xyz, cano_norm], dim=1)  # (1000, 6)
    r = r[:, None].repeat(1, num_fps).flatten()
    i = i[None, :].repeat(num_fps, 1).flatten()
    
    valid_idx = (r != i)
    sr = scene_point[r]
    si = scene_point[i]
    mr_pd = model_point_pd[r]
    mi_pd = model_point_pd[i]
    Tpose_pd = point_pair_matching(sr, si, mr_pd, mi_pd)
    pd_R = Tpose_pd[:, 0:3, 0:3][valid_idx].contiguous()  # (9900, 3, 3)
    pd_t = Tpose_pd[:, 0:3, 3][valid_idx, None, :].contiguous()  # (9900, 1, 3)

    '''select confident point pairs'''
    N = pd_R.shape[0]
    matching = torch.bmm(scene_xyz.repeat(N, 1, 1) - pd_t, pd_R)  # (9900, 1000, 3)
    dis_mnx = L2_Dis(matching, cano_xyz)  # (9900, )
    idx_mnx = torch.sort(dis_mnx)[1]
    out_Rx = pd_R[idx_mnx][0:int(0.1*N)]
    out_tx = pd_t[idx_mnx][0:int(0.1*N)]
    return out_Rx, out_tx


if __name__ == "__main__":
    from transforms3d.axangles import axangle2mat
    axis = torch.tensor([0.,0.,1.])
    angle = torch.tensor([3.1415926535/2])
    r = axis_angle_to_matrix(axis, angle).squeeze()
    r2 = torch.tensor(axangle2mat(axis, angle))
    normal = torch.tensor([1., 0., 0.]).unsqueeze(1)
    # print(r, r2)
    print(torch.mm(r, normal))
