from PIL import Image
import cv2
import numpy as np


# PIL: RGB, OpenCV: BGR
def convert_cv2topil(array):
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


def convert_piltocv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    import pdb
    vidf='../../data/dataset/cmumosei/video/original/--qXJuDtHPw_5.mp4'
    imgf='./test_image/test.jpg'

    cap = cv2.VideoCapture(vidf)

    while True:
        ret, frame = cap.read()
        cv2.imwrite(imgf, frame)

        pil_image = Image.open(imgf)
        pil_frame = np.array(Image.open(imgf))
        cv2_frame = cv2.imread(imgf)

        pdb.set_trace()
