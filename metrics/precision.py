"""
Calculate the precision metric, which is defined as the overlap of the pixels of the 
explanation contained in the bounding box and all pixels of the explanation.
"""

from PIL import Image
import numpy as np


def calc_precision(exp_image, img_size, fname, bboxes):
    """Calculate the precision for the explanation image."""
    # Resize the attention map to be the same size as the image via bilinear interpolation.
    slot_image = np.array(
        exp_image.resize((img_size, img_size), resample=Image.BILINEAR),
        dtype=np.uint8,
    )
    attention_sum = slot_image.sum()  # sum of complete attention map

    max_box_sum = 0
    # Calculate the sum of attention map pixels that fall inside the bounding box.
    for box in bboxes[fname]:
        # Determine what pixels of the attention map fall into the bounding box.
        box_pixels = slot_image[box[1] : box[3], box[0] : box[2]]
        # Take their sum.
        box_sum = box_pixels.sum()
        max_box_sum = max(box_sum, max_box_sum)  # Take the bounding box with the highest overlap.

    precision = max_box_sum / attention_sum

    return precision