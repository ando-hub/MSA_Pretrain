#!/usr/bin/env python
import torch
import numpy as np

from extlib.VGGFace2.models import resnet as ResNet
from extlib.VGGFace2.models import senet as SENet
from extlib.VGGFace2.utils import load_state_dict
from torchvision import transforms as transforms

torch.use_deterministic_algorithms(True)
N_IDENTITY = 8631  # the number of identities in VGGFace2 for which ResNet and SENet are trained


class VGGFace2Encoder():
    def __init__(self, model_path, arch_type='senet50_ft', device='cpu'):
        self.device = device
        # load model
        if 'resnet' in arch_type:
            self.model = ResNet.resnet50(num_classes=N_IDENTITY, include_top=False)
        else:
            self.model = SENet.senet50(num_classes=N_IDENTITY, include_top=False)
        self.embed_size = 2048
        load_state_dict(self.model, model_path)
        self.model.eval()
        self.model.to(self.device)

        self.mean_bgr = torch.tensor([91.4953, 103.8827, 131.0912]).unsqueeze(-1).unsqueeze(-1)  # from resnet50_ft.prototxt
        self.resize = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                    ]
                )
        self.resize_torch = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    ]
                )

    def extract(self, images):
        transformed_images = [self.transform_image(im) for im in images if im is not None]
        image_indices = [i for i, im in enumerate(images) if im is not None]

        embeds = np.zeros((len(images), self.embed_size), dtype=np.float32)
        if transformed_images:
            x = torch.stack(transformed_images).to(device=self.device)
            with torch.no_grad():
                y = self.model(x)
                y = y.view(y.size(0), -1).cpu().detach().numpy()
            for i, yi in zip(image_indices, y):
                embeds[i] = yi
        return embeds

    def transform_image(self, image):
        if isinstance(image, torch.Tensor):
            image = self.resize_torch(image)
            image = image[[2, 1, 0]]            # RGB -> BGR
        else:
            image = self.resize(image) * 255    # [0, 1] -> [0, 255]
        return image - self.mean_bgr.expand(-1, image.shape[1], image.shape[2])
