import torch
import torch.nn as nn
import torch.nn.functional as F


"""
The 3D permeability field (also state variable) has shape: [8, 40, 40], in a mini-batch: [N, 8, 40, 40].
We would like to map it the latent space with dimension num = 256. (a hyper-parameter needed to be tuned)
"""


class PermAE(nn.Module):
    def __init__(self):
        super(PermAE, self).__init__()
        self.encode_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),  # keep dim
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 16, kernel_size=2, stride=2, padding=0, bias=False),  # reduce 2x [4, 20, 20]
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.Conv3d(16, 64, kernel_size=3, padding=1, bias=False),  # keep dim
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=2, stride=2, padding=0, bias=False),  # reduce 2x [2, 10, 10]
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 256, kernel_size=3, padding=1, bias=False),  # keep dim
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=2, stride=2, padding=0, bias=False),  # reduce 2x [1, 5, 5]
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(256, 128, kernel_size=3, padding=1, bias=False),  # keep dim
        )

        self.encode_fc = nn.Sequential(
            nn.Linear(3200, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 128, bias=False),
        )

        self.decode_fc = nn.Sequential(
            nn.Linear(128, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 3200, bias=False),  # [3200] --> [128, 1, 5, 5]
        )

        self.decode = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),  # [256, 1, 5, 5]

            nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2, bias=False),  # [2, 10, 10]
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2, bias=False),  # [4, 20, 20]
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2, bias=False),  # [8, 40, 40]
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(),

            nn.Conv3d(1, 1, kernel_size=1, padding=0)

        )

    def forward(self, x):
        x = x.unsqueeze(1)  # add a channel dimension
        x = self.encode_conv(x)  # CNN encode

        x = x.view(x.size(0), -1)  # flatten
        x = self.encode_fc(x)  # MLP

        x = self.decode_fc(x)  # MLP
        x = x.view(x.size(0), 128, 1, 5, 5)  # resize to 3d shape
        x = self.decode(x)  # CNN decode
        x = x.squeeze(1)  # squeeze the channel dimension
        return x


if __name__ == '__main__':
    input_perm = torch.randn((16, 8, 40, 40))
    print(f'Input variable shape: {input_perm.shape}')
    ae = PermAE()

    reconstruct = ae(input_perm)
    print(f'Reconstructed variable shape: {reconstruct.shape}')

    print('-'*50)
    print(f'total params: {sum(p.numel() for p in ae.parameters() if p.requires_grad)}')


