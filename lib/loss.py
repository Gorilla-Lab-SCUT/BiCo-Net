import torch
from torch.nn.modules.loss import _Loss

from lib.knn.__init__ import KNearestNeighbor


class Loss(_Loss):
    def __init__(self, sym_list):
        super(Loss, self).__init__(True)
        self.sym_list = sym_list
        self.knn = KNearestNeighbor(1)


    def L2_Dis(self, pred, target):
        return torch.norm(pred - target, dim=2).mean(1)


    def ADDS_Dis(self, pred, target):
        num_p, num_point_mesh, _ = pred.size()
        target = target[0].transpose(1, 0).contiguous().view(3, -1)
        pred = pred.permute(2, 0, 1).contiguous().view(3, -1)
        inds = self.knn.forward(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        pred = pred.view(3, num_p, num_point_mesh).permute(1, 2, 0).contiguous()
        target = target.view(3, num_p, num_point_mesh).permute(1, 2, 0).contiguous()
        return torch.norm(pred - target, dim=2).mean(1)


    def forward(self, out_rx, out_tx, out_mx, out_nx, out_ax, out_bx, cad, model_points, points, normal, target_r, target_t, cls, epoch):
        sym_flag = True if cls[0] in self.sym_list else False
        bs, num_p, _ = out_rx.size()

        out_Rx = torch.cat(((1.0 - 2.0 * (out_rx[:, :, 2] ** 2 + out_rx[:, :, 3] ** 2)).view(bs, num_p, 1), \
                (2.0 * out_rx[:, :, 1] * out_rx[:, :, 2] - 2.0 * out_rx[:, :, 0] * out_rx[:, :, 3]).view(bs, num_p, 1), \
                (2.0 * out_rx[:, :, 0] * out_rx[:, :, 2] + 2.0 * out_rx[:, :, 1] * out_rx[:, :, 3]).view(bs, num_p, 1), \
                (2.0 * out_rx[:, :, 1] * out_rx[:, :, 2] + 2.0 * out_rx[:, :, 3] * out_rx[:, :, 0]).view(bs, num_p, 1), \
                (1.0 - 2.0 * (out_rx[:, :, 1] ** 2 + out_rx[:, :, 3] ** 2)).view(bs, num_p, 1), \
                (-2.0 * out_rx[:, :, 0] * out_rx[:, :, 1] + 2.0 * out_rx[:, :, 2] * out_rx[:, :, 3]).view(bs, num_p, 1), \
                (-2.0 * out_rx[:, :, 0] * out_rx[:, :, 2] + 2.0 * out_rx[:, :, 1] * out_rx[:, :, 3]).view(bs, num_p, 1), \
                (2.0 * out_rx[:, :, 0] * out_rx[:, :, 1] + 2.0 * out_rx[:, :, 2] * out_rx[:, :, 3]).view(bs, num_p, 1), \
                (1.0 - 2.0 * (out_rx[:, :, 1] ** 2 + out_rx[:, :, 2] ** 2)).view(bs, num_p, 1)), dim=2).view(bs*num_p, 3, 3).contiguous()
        out_tx = (points + out_tx).view(bs*num_p, 1, 3).contiguous()
        pred_R = torch.mean(out_Rx, dim=0, keepdim=True)
        pred_t = torch.mean(out_tx, dim=0, keepdim=True)

        pred = torch.bmm(model_points.repeat(bs*num_p, 1, 1), out_Rx.transpose(2, 1)) + out_tx
        target = torch.bmm(model_points, target_r.transpose(2, 1)) + target_t

        matching1_gt = torch.bmm(points - target_t, target_r)
        matching2_gt = torch.bmm(normal, target_r)
        matching3_gt = torch.bmm(cad[:, :, 0:3], target_r.transpose(2, 1)) + target_t
        matching4_gt = torch.bmm(cad[:, :, 3:6], target_r.transpose(2, 1))

        matching1_pd = torch.bmm(points - pred_t, pred_R)
        matching2_pd = torch.bmm(normal, pred_R)
        matching3_pd = torch.bmm(cad[:, :, 0:3], pred_R.transpose(2, 1)) + pred_t
        matching4_pd = torch.bmm(cad[:, :, 3:6], pred_R.transpose(2, 1))      

        if epoch != 0 and sym_flag:
            loss_pose = self.ADDS_Dis(pred, target).mean(0)
            loss_aux1 = self.L2_Dis(out_mx, matching1_pd.detach())
            loss_aux2 = self.L2_Dis(out_nx, matching2_pd.detach())
            loss_aux3 = self.L2_Dis(out_ax, matching3_pd.detach())
            loss_aux4 = self.L2_Dis(out_bx, matching4_pd.detach())            
        else:
            loss_pose = self.L2_Dis(pred, target.repeat(bs*num_p, 1, 1)).mean(0)
            loss_aux1 = self.L2_Dis(out_mx, matching1_gt)
            loss_aux2 = self.L2_Dis(out_nx, matching2_gt)
            loss_aux3 = self.L2_Dis(out_ax, matching3_gt)
            loss_aux4 = self.L2_Dis(out_bx, matching4_gt)               

        loss = loss_pose + loss_aux1 + 0.05 * loss_aux2 + loss_aux3 + 0.05 * loss_aux4
        return loss, loss_pose, loss_aux1, loss_aux2, loss_aux3, loss_aux4
