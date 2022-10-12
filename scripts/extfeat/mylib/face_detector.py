import os
import argparse
import numpy as np
import pdb
import json
from tqdm import tqdm
from scipy.signal import medfilt
try:
    import cv2
except ImportError as e:
    print(e)
try:
    import dlib
except ImportError as e:
    print(e)


def _parse():
    parser = argparse.ArgumentParser(description='Detect Speech Interval')
    parser.add_argument('ind', type=str, help='input jpeg dir')
    parser.add_argument('outd', type=str, help='output jpeg dir')
    parser.add_argument('-r', type=float, default=1.,
                        help='resize video')
    parser.add_argument('-b', choices=['opencv', 'dlib', 'dlib_dnn'], default='dlib',
                        help='face detection backend')
    parser.add_argument('-o', type=str,
                        help='output jpeg dir with face rectangle')
    parser.add_argument('--dlib-dnn-model', type=str,
                        default='./dlib-models/mmod_human_face_detector.dat',
                        help='dlib-dnn model path')
    parser.add_argument('--opencv-model', type=str,
                        default='./opencv/data/haarcascades/haarcascade_frontalface_default.xml',
                        help='opencv haar-cascade model path')
    return parser.parse_args()


class FaceDetector():
    def __init__(self, backend='dlib', model=None):
        self.backend = backend
        assert os.path.exists(model), 'No exist: {}'.format(model)
        if backend == 'opencv':
            self.detector = cv2.CascadeClassifier(model)
        elif backend == 'dlib':
            self.detector = dlib.get_frontal_face_detector()
        elif backend == 'dlib_dnn':
            self.detector = dlib.cnn_face_detection_model_v1(model)
        else:
            raise ValueError('backend must be either opencv or dlib!')

    def __rect2coordinates(self, rect, resize, im):
        if self.backend == 'opencv':
            left = int(rect[0] / resize)
            top = int(rect[1] / resize)
            right = int((rect[0] + rect[2]) / resize)
            bottom = int((rect[1] + rect[3]) / resize)
        elif self.backend == 'dlib':
            left = int(rect.left() / resize)
            top = int(rect.top() / resize)
            right = int(rect.right() / resize)
            bottom = int(rect.bottom() / resize)
        elif self.backend == 'dlib_dnn':
            left = int(rect.rect.left() / resize)
            top = int(rect.rect.top() / resize)
            right = int(rect.rect.right() / resize)
            bottom = int(rect.rect.bottom() / resize)
        top = max([0, top])
        bottom = min([bottom, im.shape[0]])
        left = max([0, left])
        right = min([right, im.shape[1]])
        return top, bottom, left, right

    def detect(self, images, save_path=None, rect_save_path=None):
        images_cropped = []
        for i, image in enumerate(images):
            #resize = self.height / image.shape[0]
            #image_reshape = cv2.resize(image, dsize=None, fx=resize, fy=resize)
            image_reshape = image
            
            # detect
            if self.backend == 'opencv':
                frame_gray = cv2.cvtColor(image_reshape, cv2.COLOR_BGR2GRAY)
                facerect = self.detector.detectMultiScale(
                        frame_gray,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(30, 30)
                        )
            elif self.backend == 'dlib' or self.backend == 'dlib_dnn':
                facerect = self.detector(image_reshape)
            
            # return cropped face image
            if len(facerect) < 1:
                im_cropped = None
            else:
                # select the largest face as detection result
                regions = [self.__rect2coordinates(r, resize, image) for r in facerect]
                areas = np.array([(v[1]-v[0])*(v[3]-v[2]) for v in regions])
                (top, bottom, left, right) = regions[areas.argmax()]
                im_cropped = image[top:bottom, left:right, :]
            images_cropped.append(im_cropped)

            if save_path:
                if im_cropped is not None:
                    cv2.imwrite(save_path[i], im_cropped) 

            # save original image with face rectangles
            if rect_save_path:
                for r in facerect:
                    top, bottom, left, right = self.__rect2coordinates(r, resize, image)
                    cv2.rectangle(
                            image,
                            (left, top),
                            (right, bottom),
                            (0, 255, 0),
                            thickness=2
                            )
                cv2.imwrite(rect_save_path[i], image, [cv2.IMWRITE_JPEG_QUALITY, 100])


        return images_cropped


def _main():
    args = _parse()

    face_detector = FaceDetector(args.b, args.dlib_dnn_model, args.opencv_model)
    rectf = None
    for dpath, dname, fnames in os.walk(args.ind):
        for fname in fnames:
            if fname.endswith('.jpg'):
                inf = os.path.join(dpath, fname)
                middir = dpath.replace(args.ind, '').strip(os.sep)
                outf = os.path.join(args.outd, middir, fname)
                nfacef = os.path.join(args.outd, middir, fname.replace('.jpg', '.nfaces'))
                os.makedirs(os.path.dirname(outf), exist_ok=True)
                if args.o:
                    rectf = os.path.join(args.o, middir, fname)
                    os.makedirs(os.path.dirname(rectf), exist_ok=True)
                if not os.path.exists(outf):
                    face_detector.detect(inf, outf, nfacef, args.r, rectf)


if __name__ == '__main__':
    _main()

