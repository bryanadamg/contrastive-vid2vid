import logging

import torch
import torch.nn as nn

from .swin_unet import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, input_nc=3):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                patch_size=4,
                                in_chans=input_nc,
                                num_classes=self.num_classes,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x, layers=[], encode_only=False):
        if encode_only:
            logits, feats = self.swin_unet.forward_features(x)
            return feats
        else:
            logits = self.swin_unet(x)
            return logits


if __name__ == "__main__":
    model = SwinUnet(None, num_classes=3)
    input = torch.rand(1, 3, 224, 224)
    output = model(input)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(n_parameters)
    print(output.size())
 