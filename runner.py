import os
from argparse import ArgumentParser
from warnings import filterwarnings

import cv2
import numpy

from nets import nn
from utils import util
filterwarnings("ignore")

def paddle_onnx_run(file_path, det:nn.Detection, rec:nn.Recognition):

    detection = det
    recognition = rec

    frame = cv2.imread(file_path)
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

    return results, confidences