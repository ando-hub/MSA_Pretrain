import torch
import torch.nn as nn


class GatePooling(nn.Module):
    def __init__(self, in_dim, n_modal):
        super(GatePooling, self).__init__()
        self.n_modal = n_modal
        self.fcs = nn.ModuleDict({str(i): nn.Linear(in_dim*n_modal, 1) for i in range(n_modal)})

    def forward(self, x, need_weights=False):
        """
        Args:
            x (torch.Tensor): input multimodal embedding (B, n_modal, embed_dim)
            need_weights (torch.Tensor): given gate weights (B, n_comb, n_modal)

        Returns:
            x (torch.Tensor): gated sum of the modal embedding(B, embed_dim)
            alpha (torch.Tensor): predicted gate weights (B, n_modal)
        """
        assert x.shape[1] == self.n_modal, 'input tensor must be [n_bat, n_modal, n_dim]!'

        n_bat, n_modal, n_dim = x.shape
        h = x.view(n_bat, -1)                           # (B, n_modal*n_dim)
        alpha = torch.cat(
                [torch.sigmoid(self.fcs[str(i)](h)) for i in range(n_modal)],
                1)                                      # (B, n_modal)

        w = alpha.unsqueeze(1)                          # (B, 1, n_modal)
        x = torch.bmm(w, x)                             # (B, 1, n_dim)
        x = x.squeeze(1)                                # (B, n_dim)

        if need_weights:
            return x, alpha
        else:
            return x
