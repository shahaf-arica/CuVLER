import torch
from dino.vision_transformer import vit_base, vit_small
import re

def get_model(arch, device="cpu"):

    if "dinov2" in arch:
        patch_size = 14
        if arch == "dinov2_s14":
            from dinov2.hubconf import dinov2_vits14
            model = dinov2_vits14()
        elif arch == "dinov2_b14":
            from dinov2.hubconf import dinov2_vitb14
            model = dinov2_vitb14()
        else:
            raise NotImplementedError
    elif "dino" in arch:
        patch_size = int(re.search(r"\d+", arch).group(0))
        # if pattern in s<int> then the arch is vit_small if b<int> then vit_base
        if arch == "dino_s16":
            url = "dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            vit = vit_small
        elif arch == "dino_s8" :
            url = "dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            vit = vit_small
        elif arch == "dino_b16":
            url = "dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            vit = vit_base
        elif arch == "dino_b8":
            url = "dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            vit = vit_base
        else:
            raise ValueError("Unknown dino v1 arch {}".format(arch))
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/" + url
        )
        model = vit(patch_size=patch_size, num_classes=0)
        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError("Unknown arch {}".format(arch))
    model.eval()
    model.to(device)
    return model, patch_size

