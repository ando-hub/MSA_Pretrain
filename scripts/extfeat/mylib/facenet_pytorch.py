import pdb
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms as transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

from mylib.util import convert_cv2topil, convert_piltocv2


class FacenetFaceDetector():
    def __init__(self, face_size=160, device='cpu', post_process=False):
        self.device = device
        self.post_process = post_process
        self.mtcnn = MTCNN(
                image_size=face_size,
                select_largest=True,
                keep_all=False,
                margin=0,
                device=device,
                post_process=post_process
                )

    def detect(self, images, save_path=None):
        # convert np.array -> PIL format
        images = [convert_cv2topil(im) for im in images]
        faces = self.mtcnn(images, save_path)
        if not self.post_process:
            # convert (3, face_size, face_size)-RGB to (face_size, face_size, 3)-BGR (cv2 format)
            _faces = []
            for f in faces:
                if f is None:
                    _faces.append(f)
                else:
                    # (size, size, 3)-RGB
                    f = f.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
                    _faces.append(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            faces = _faces
        return faces


class FacenetEncoder():
    def __init__(self, device='cpu'):
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.embed_size = 512

    def extract(self, images):
        # get input non-zero (face-detected) data
        transformed_images = []
        image_indices = []
        for i, im in enumerate(images):
            if im is None:
                continue
            transformed_images.append(self.preproc(im))
            image_indices.append(i)

        embeds = np.zeros((len(images), self.embed_size), dtype=np.float32)
        if transformed_images:
            x = torch.stack(transformed_images).to(self.device)
            with torch.no_grad():
                y = self.model(x)
                y = y.view(y.size(0), -1).cpu().detach().numpy()
            for i, yi in zip(image_indices, y):
                embeds[i] = yi
        pdb.set_trace()
        return embeds
    
    def preproc(self, image):
        if isinstance(image, torch.Tensor):
            return image
        else:
            raise ValueError('image must be facenet MTCNN output')
