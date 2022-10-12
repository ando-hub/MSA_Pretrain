import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class MultiLabelAngleProtoCELoss(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(MultiLabelAngleProtoCELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = nn.CrossEntropyLoss()

    def forward(self, x, label):
        assert len(x.shape) == 2
        # add neutral label + remove zero-class
        _label = torch.cat((label, (label.sum(dim=1)==0).unsqueeze(0).T), 1)
        _label = _label[:, _label.sum(dim=0)>0].float()
        _n_centroid = _label.shape[-1]
        # expand input vector x
        x_expand = torch.stack([x[i[0]] for i in _label.nonzero()], 0)
        label_expand = _label.nonzero()[:, 1]
        if _n_centroid:
            # get average embeddings
            anchor = torch.stack(
                    [torch.mean(x_expand[label_expand==i], dim=0) for i in range(_n_centroid)]
                    )
            cos_sim_matrix = F.cosine_similarity(
                    x_expand.unsqueeze(-1), anchor.unsqueeze(-1).transpose(0, 2)
                    )
            torch.clamp(self.w, 1e-6)
            cos_sim_matrix = cos_sim_matrix * self.w + self.b
            nloss = self.criterion(cos_sim_matrix, label_expand)
        else:
            nloss = 0
        return nloss 


class ModalMetricCELoss(nn.CrossEntropyLoss):
    def __init__(self, lossfunc, lossweight):
        super(ModalMetricCELoss, self).__init__()
        self.lossfunc = lossfunc
        self.lossweight = lossweight
        self.angleproto = MultiLabelAngleProtoCELoss()
    
    def forward(self, y, t, e):
        """
        Args:
            y (torch.Tensor): model output w/ modal combinations (B, n_comb, out_dim)
            t (torch.Tensor): teacher label of the given lossfunc (B, ) / (B, out_dim)
            e (list of torch.Tensor): modal embeddings [h_v, h_a, h_t]; (B, n_emb) or None
        Returns:
            loss (torch.Tensor): Modal Selective Training Loss := lossfunc + (lossweight * gate_loss)
        """
        metric_loss = [self.angleproto(ei, t) for ei in e if ei is not None]
        metric_loss = sum(metric_loss)/len(metric_loss)
        loss = self.lossfunc(y, t) + self.lossweight * metric_loss
        return loss


if __name__ == '__main__':
    x = torch.rand(3, 4)
    l = torch.tensor([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                ])
    lossfunc = MultiLabelAngleProtoCELoss()
    loss = lossfunc(x, l)
    print(loss)

