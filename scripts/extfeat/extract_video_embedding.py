import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import pdb
try:
    import cv2
except ImportError as e:
    print(e)

from mylib.face_detector import FaceDetector
from mylib.facenet_pytorch import FacenetFaceDetector, FacenetEncoder
from extlib.CLIP_decode import CLIPEncoder
try:
    from extlib.VGGFace2_decode import VGGFace2Encoder
except ModuleNotFoundError as e:
    print(e)


def _parse():
    parser = argparse.ArgumentParser(description='Visual feature extraction')
    parser.add_argument('ind', type=str, help='input video dir')
    parser.add_argument('outd', type=str, help='output video feature dir')
    parser.add_argument('--fps', type=int, default=3,
                        help='feature extraction frequency')
    parser.add_argument('--face-detect-method',
                        choices=['dlib', 'dlib_dnn', 'opencv', 'facenet'],
                        default='facenet',
                        help='face detection method [dlib|dlib_dnn|opencv|facenet]')
    parser.add_argument('--face-detect-model', type=str,
                        help='face detection model (opencv/dlib_dnn only)')
    parser.add_argument('--encoder-type', type=str,
                        choices=['VGGFace2', 'facenet', 'CLIP'],
                        default='CLIP',
                        help='feature extraction method [VGGFace2|facenet|CLIP]')
    parser.add_argument('--encoder-model', type=str,
                        help='feature extraction model (VGGFace2/facenet only)')
    parser.add_argument('--face-outd', type=str,
                        help='detected faces output dir')
    parser.add_argument('--image-outd', type=str,
                        help='subsampled images output dir')
    parser.add_argument('--gpuid', type=int, default=-1,
                        help='gpu id (run cpu if gpuid < 0)')
    parser.add_argument('--face-size', type=int, default=256,
                        help='cropped face image size (default: 256)')
    parser.add_argument('--get-layer-results', action='store_true', default=False,
                        help='get results of the all encoder layers')
    return parser.parse_args()


class VideoEmbeddingExtractor():
    def __init__(self,
                 face_detect_method,
                 encoder_type,
                 face_detect_model=None,
                 encoder_model=None,
                 device='cpu',
                 max_proc_frames=100,
                 face_size=160,
                 fps=3,
                 get_layer_results=False
                 ):
        self.max_proc_frames = max_proc_frames
        self.fps = fps

        if face_detect_method in ['opencv', 'dlib', 'dlib_dnn']:
            self.face_detector = FaceDetector(
                    backend=face_detect_method,
                    model=face_detect_model,
                    )
        elif face_detect_method in ['facenet']:
            post_process = True if encoder_type == 'facenet' else False
            self.face_detector = FacenetFaceDetector(
                    face_size=face_size,
                    device=device,
                    post_process=post_process
                    )
        else:
            raise ValueError('invalid face_detect_method: {}'.format(face_detect_method))

        if encoder_type == 'VGGFace2':
            self.feat_extractor = VGGFace2Encoder(
                    model_path=encoder_model,
                    arch_type='senet50_ft',
                    device=device
                    )
        elif encoder_type == 'facenet':
            self.feat_extractor = FacenetEncoder(
                    device=device
                    )
        elif encoder_type == 'CLIP':
            self.feat_extractor = CLIPEncoder(
                    arch_type='ViT-L/14',
                    get_layer_results=get_layer_results,
                    device=device
                    )
        else:
            raise ValueError('invalid encoder_type: {}'.format(encoder_type))

    def split_sublist(self, images, max_proc_frames, outfs=None):
        subs = []
        for i in range(0, len(images), max_proc_frames):
            if outfs:
                u = (images[i:i+max_proc_frames], outfs[i:i+max_proc_frames])
            else:
                u = (images[i:i+max_proc_frames], None)
            subs.append(u)
        return subs

    def detect_face(self, images, face_outd=None):
        if face_outd:
            os.makedirs(face_outd, exist_ok=True)
            outfs = [os.path.join(face_outd, '{:05d}.jpg'.format(i)) for i in range(len(images))]
        else:
            outfs = None

        faces = []
        for _images, _outfs in self.split_sublist(images, int(self.max_proc_frames), outfs):
            faces.extend(self.face_detector.detect(_images, _outfs))
        return faces

    def decode(self, images):
        embeds = []
        for _images, _ in self.split_sublist(images, int(self.max_proc_frames/2)):
            embeds.append(self.feat_extractor.extract(_images))
        return np.concatenate(embeds, axis=0)

    def extract_fromclip(self, clip, embed_outf, frame_outd=None, face_outd=None):
        frames = get_frames(clip, self.fps, dst_height=720, image_outd=frame_outd)
        faces = self.detect_face(frames, face_outd)
        embeds = self.decode(faces)
        if len(embeds.shape) == 3:
            # layer results: [nlen, nlay, ndim] -> [nlay, nlen, ndim]
            embeds = embeds.transpose(1, 0, 2)
        np.save(embed_outf, embeds)


def get_frames(inf, fps, dst_height=-1, image_outd=None):
    def scale_to_height(img, height):
        h, w = img.shape[:2]
        width = round(w * (height / h))
        dst = cv2.resize(img, dsize=(width, height))
        return dst

    cap = cv2.VideoCapture(inf)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    resamp_rate = int(src_fps/float(fps))

    images = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (i % resamp_rate) == 0:
            if dst_height > 0:
                frame = scale_to_height(frame, dst_height)
            images.append(frame)
        i += 1
    cap.release()

    if image_outd:
        os.makedirs(image_outd, exist_ok=True)
        for i, image in enumerate(images):
            cv2.imwrite(os.path.join(image_outd, '{:05d}.jpg'.format(i)), image)
    return images


def load_images(imgd):
    return [cv2.imread(f) for f in sorted(glob.glob(os.path.join(imgd, '*.jpg')))]


def _main():
    args = _parse()
    device = 'cuda:{}'.format(args.gpuid) if args.gpuid >= 0 else 'cpu'
    os.makedirs(args.outd, exist_ok=True)
    if args.image_outd:
        os.makedirs(args.image_outd, exist_ok=True)
    if args.face_outd:
        os.makedirs(args.face_outd, exist_ok=True)

    print('prepare extractor')
    extractor = VideoEmbeddingExtractor(
            args.face_detect_method,
            args.encoder_type,
            face_detect_model=args.face_detect_model,
            encoder_model=args.encoder_model,
            face_size=args.face_size,
            device=device,
            fps=args.fps,
            get_layer_results=args.get_layer_results
            )

    # get unproceeded files for resume
    infs = []
    for f in glob.glob(os.path.join(args.ind, '*.mp4')):
        clip_id = os.path.splitext(os.path.basename(f))[0]
        outf = os.path.join(args.outd, clip_id+'.npy')
        if not os.path.exists(outf):
            infs.append(f)

    # process
    print('start feature extraction')
    for inf in tqdm(infs):
        clip_id = os.path.splitext(os.path.basename(inf))[0]
        outf = os.path.join(args.outd, clip_id+'.npy')
        image_outd = os.path.join(args.image_outd, clip_id) if args.image_outd else None
        face_outd = os.path.join(args.face_outd, clip_id) if args.face_outd else None
        try:
            extractor.extract_fromclip(inf, outf, image_outd, face_outd)
        except RuntimeError as e:
            print(e)


if __name__ == '__main__':
    _main()
