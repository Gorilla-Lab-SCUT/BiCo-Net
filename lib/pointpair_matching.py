import torch
import numpy as np

def matrix_x(angle: torch.Tensor) -> torch.Tensor:
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack([
        torch.stack([o, z, z], dim=-1),
        torch.stack([z, c, -s], dim=-1),
        torch.stack([z, s, c], dim=-1),
    ], dim=-2)

def matrix_y(angle: torch.Tensor) -> torch.Tensor:
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack([
        torch.stack([c, z, s], dim=-1),
        torch.stack([z, o, z], dim=-1),
        torch.stack([-s, z, c], dim=-1),
    ], dim=-2)

def xyz_to_angles(xyz):
    xyz = torch.nn.functional.normalize(xyz, p=2, dim=-1)
    xyz = xyz.clamp(-1, 1)
    beta = torch.acos(xyz[..., 1])
    alpha = torch.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta

def angles_to_matrix(alpha, beta, gamma):
    # alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)

def axis_angle_to_matrix(axis, angle):
    # axis, angle = torch.broadcast_tensors(axis, angle[..., None])
    alpha, beta = xyz_to_angles(axis)
    R = angles_to_matrix(alpha, beta, torch.zeros_like(beta))
    Ry = matrix_y(angle)
    return R @ Ry @ R.transpose(-2, -1)

'-------------------------------------------------------------'

def transformRT(p):# (N, 6)
    N = p.shape[0]
    Angle = torch.acos(p[:, 3])# (N, )
    Axis = torch.zeros((N, 3)).cuda()
    Axis[:, 1] = p[:, 5]
    Axis[:, 2] = -p[:, 4]
    idx = (Axis == torch.tensor([0., 0., 0.]).cuda()).all(1).nonzero().flatten()
    if len(idx): Axis[idx] = torch.tensor([0., 1., 0.]).cuda()
    Axis = Axis / torch.norm(Axis, dim=1, keepdim=True)
    R = axis_angle_to_matrix(Axis, Angle)
    p_temp = p[:, 0:3].unsqueeze(2)
    t = ((-1)*torch.bmm(R, p_temp)).squeeze(2)
    return R, t# (N, 3, 3) (N, 3)

def compute_alpha(p1, p2):# (N, 6) (N, 6)
    R, t = transformRT(p1)
    p2_temp = p2[:, 0:3].unsqueeze(2)
    mpt = (torch.bmm(R, p2_temp) + t.unsqueeze(2)).squeeze(2)
    alpha = torch.atan2(-mpt[:, 2], mpt[:, 1])
    alpha[torch.sin(alpha) * mpt[:, 2] > 0] *= -1
    return alpha

def compute_pose(sr, si, mr, mi):
    N = sr.shape[0]
    R, t = transformRT(mr)
    T = torch.zeros((N, 4, 4)).cuda()
    T[:, 3, 3] = 1.
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = t

    R, t = transformRT(sr)
    iT = torch.zeros((N, 4, 4)).cuda()
    iT[:, 3, 3] = 1.
    iT[:, 0:3, 0:3] = R.permute(0, 2, 1)
    iT[:, 0:3, 3] = (-1) * torch.bmm(R.permute(0, 2, 1), t.unsqueeze(2)).squeeze(2)

    alphaModel = compute_alpha(mr, mi)
    alphaScene = compute_alpha(sr, si)
    alphaAngle = alphaModel - alphaScene
    alphaAngle[alphaAngle > np.pi] -= 2 * np.pi
    alphaAngle[alphaAngle < -np.pi] += 2 * np.pi
    Talpha = torch.zeros((N, 4, 4)).cuda()
    Talpha[:, 3, 3] = 1.
    Talpha[:, 0:3, 0:3] = matrix_x(alphaAngle)
    Tpose = torch.bmm(torch.bmm(iT, Talpha), T)
    return Tpose 