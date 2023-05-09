import os
import sys
import random
from quickdraw import QuickDrawData
import re

app_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        "../app"))
sys.path.append(app_dir)

from constants import SHAPE_NAMES, ADDITIONAL_SHAPE_AMOUNT


def main():
    qdraw = QuickDrawData(recognized=True, max_drawings=1, cache_dir="cache")
    all_shape_names = qdraw.drawing_names

    basic_shape_names = SHAPE_NAMES
    for shape in basic_shape_names:
        if shape in all_shape_names:
            all_shape_names.remove(shape)

    sample_size = min(ADDITIONAL_SHAPE_AMOUNT, len(all_shape_names))
    additional_shape_names = random.sample(all_shape_names, sample_size)

    constants_path = os.path.join(app_dir, 'constants.py')

    with open(constants_path, 'r') as f:
        content = f.read()

    content = re.sub(
        r'ADDITIONAL_SHAPE_NAMES\s*=\s*\[.*\]',
        f'ADDITIONAL_SHAPE_NAMES = {additional_shape_names}',
        content)

    with open(constants_path, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    main()
