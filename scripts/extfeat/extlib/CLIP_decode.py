import pdb
import torch
import cv2
import numpy as np
from extlib.CLIP import clip
from PIL import Image

torch.use_deterministic_algorithms(True)


def convert_cv2topil(array):
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


class CLIPEncoder():
    def __init__(self, arch_type='ViT-L/14', get_layer_results=False, device='cpu'):
        self.device = device
        self.get_layer_results = get_layer_results
        self.model, self.preprocess = clip.load(arch_type, device=device)
        self.model.eval()
        self.embed_size = 1024 if get_layer_results else 768
        self.nlayers = 24

    def extract_test(self, imgf):
        image = self.preprocess(Image.open(imgf)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features, layer_results = self.model.encode_image_layers(image)
        if self.get_layer_results:
            embed = torch.cat([x[:, 0].unsqueeze(1) for x in layer_results], dim=1)
        else:
            embed = image_features
        return embed.squeeze().cpu().detach().numpy()

    def extract(self, images):
        transformed_images = []
        image_indices = []
        for i, im in enumerate(images):
            if im is None:
                continue
            transformed_images.append(self.preprocess(convert_cv2topil(im)))
            image_indices.append(i)

        if self.get_layer_results:
            embeds = np.zeros((len(images), self.nlayers, self.embed_size), dtype=np.float32)
        else:
            embeds = np.zeros((len(images), self.embed_size), dtype=np.float32)

        if transformed_images:
            x = torch.stack(transformed_images).to(self.device)
            # decode
            with torch.no_grad():
                image_features, layer_results = self.model.encode_image_layers(x)

            # get embeddings
            if self.get_layer_results:
                # get first segment ([CLS] token)
                y = torch.cat([x[:, 0].unsqueeze(1) for x in layer_results], dim=1)
            else:
                y = image_features
            y = y.cpu().detach().numpy()
            for i, yi in zip(image_indices, y):
                embeds[i] = yi
        return embeds


if __name__ == '__main__':
    image = '../../../data/dataset/cmumosei/video/face_dbg_facenet_VGGFace2/RImmdklTOW0_3/00000.jpg'
    images = [
            '../../../data/dataset/cmumosei/video/face_dbg_facenet_VGGFace2/RImmdklTOW0_3/00000.jpg',
            '../../../data/dataset/cmumosei/video/face_dbg_facenet_VGGFace2/RImmdklTOW0_3/00001.jpg',
            '../../../data/dataset/cmumosei/video/face_dbg_facenet_VGGFace2/RImmdklTOW0_3/00002.jpg'
            ]

    data = [cv2.imread(i) for i in images]
    data.insert(2, None)
    enc = CLIPEncoder(get_layer_results=True)
    y = enc.extract_test(image)
    ys = enc.extract(data)
    pdb.set_trace()
