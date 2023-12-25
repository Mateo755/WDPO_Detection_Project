import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm

from cvzone.ClassificationModule import Classifier


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        A dictionary with the number of each object.
    """

    ## Model import
    classifier = Classifier('Model/leaves_classifier_model.h5', 'Model/labels.txt')

    ## Convert indexes of classes to names

    TrainClasses = {'aspen': 0, 'birch': 1, 'hazel': 2, 'maple': 3, 'oak': 4}
    ResultMap = {}
    for idx, leaf in zip(TrainClasses.values(), TrainClasses.keys()):
        ResultMap[idx] = leaf

    aspen = 0
    birch = 0
    hazel = 0
    maple = 0
    oak = 0

    ## Image preprocessing
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgGray, 250, 255, cv2.THRESH_BINARY_INV)

    imgBlur = cv2.GaussianBlur(thresh, (5, 5), 0)
    imgCanny = cv2.Canny(imgBlur, 120, 120)

    countours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in countours:
        area = cv2.contourArea(cnt)
        if area > 80:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = img[y:y + h, x:x + w]

            prediction, index = classifier.getPrediction(roi, draw=False)
            leaf = ResultMap[index]

            match leaf:
                case "aspen":
                    aspen += 1
                case "birch":
                    birch += 1
                case "hazel":
                    hazel += 1
                case "maple":
                    maple += 1
                case "oak":
                    oak += 1

    return {'aspen': aspen, 'birch': birch, 'hazel': hazel, 'maple': maple, 'oak': oak}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        leaves = detect(str(img_path))
        results[img_path.name] = leaves

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
