from mfai.mfai.pytorch.models import unetrpp
import torch.nn as nn

class UNetRPP(unetrpp.BaseModel):
    def __init__(self, in_channels, out_channels, input_shape, settings=unetrpp.UNetRPPSettings()):
        super().__init__()
        # ... (outras inicializações do modelo original)
        self.out1 = nn.Sequential(
            unetrpp.UnetOutBlock(spatial_dims=settings.spatial_dims, in_channels=settings.hidden_size // 16, out_channels=out_channels),
            nn.Sigmoid()
        )
        if self.do_ds:
            self.out2 = nn.Sequential(
                unetrpp.UnetOutBlock(spatial_dims=settings.spatial_dims, in_channels=settings.hidden_size // 8, out_channels=out_channels),
                nn.Sigmoid()
            )
            self.out3 = nn.Sequential(
                unetrpp.UnetOutBlock(spatial_dims=settings.spatial_dims, in_channels=settings.hidden_size // 4, out_channels=out_channels),
                nn.Sigmoid()
            )

    def forward(self, x_in):
        logits = super().forward(x_in)
        return logits  # A saída já será normalizada por Sigmoid