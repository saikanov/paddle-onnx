import os
from argparse import ArgumentParser
from warnings import filterwarnings

import cv2
import numpy

from nets import nn
from utils import util
filterwarnings("ignore")

detection = nn.Detection('/home/saika/skkmigas/ocr/paddle-onnx/models/PP-OCRv5_mobile_det_infer.onnx')
recognition = nn.Recognition('/home/saika/skkmigas/ocr/paddle-onnx/models/en_PP-OCRv4_mobile_rec_infer.onnx')


def main():
    parser = ArgumentParser()
    parser.add_argument('filepath', type=str, help='image file path')

    args = parser.parse_args()

    frame = cv2.imread(args.filepath)
    image = frame.copy()

    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)  # inplace

    points = detection(frame)
    points = util.sort_polygon(list(points))

    # draw detected polygon
    for point in points:
        point = numpy.array(point, dtype=numpy.int32)
        cv2.polylines(image,
                      [point], True,
                      (0, 255, 0), 2)

    cropped_images = [util.crop_image(frame, x) for x in points]
    results, confidences = recognition(cropped_images)

    # draw recognized text
    for i, result in enumerate(results):
        point = points[i]
        x, y, w, h = cv2.boundingRect(point)
        image = cv2.putText(image, result, (int(x), int(y - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (200, 200, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.basename(args.filepath), image)

    for result in results:
        print(result + "\n")


if __name__ == '__main__':
    main()
