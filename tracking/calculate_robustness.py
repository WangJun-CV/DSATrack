from typing import Union

from typing.io import TextIO
import numpy as np
from numba import jit
from util import  calculate_iou

@jit(nopython=True)
def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
        idx_ += rle[i]
def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)

    # create mask from RLE within target region
    mask = rle_to_mask(rle, region_w, region_h)
    region = [tl_x, tl_y, region_w, region_h]

    return mask, (tl_x, tl_y), region
def parse(string):
    """
    parse string to the appropriate region format and return region object
    """
    from vot.region.shapes import Rectangle, Polygon, Mask


    if string[0] == 'm':
        # input is a mask - decode it
        m_, offset_, region = create_mask_from_string(string[1:].split(','))
        # return Mask(m_, offset=offset_)
        return region
    else:
        # input is not a mask - check if special, rectangle or polygon
        raise NotImplementedError
    print('Unknown region format.')
    return None
def read_file(fp: Union[str, TextIO]):
    if isinstance(fp, str):
        with open(fp) as file:
            lines = file.readlines()
    else:
        lines = fp.readlines()

    regions = []
    # iterate over all lines in the file
    for i, line in enumerate(lines):
        # print(line)
        regions.append(parse(line.strip()))
    return regions

iou_threshold=0.7
vot_path="/media/gg/data/datasets/VOT2020/annotations/"
def calculate_robustness(list1,list2):
    total_boxes=len(list1)
    iou_value=[]
    for i in range(total_boxes):
        iou=calculate_iou(list1[i],list2[i])
        iou_value.append(iou)
    iou_up_threshold=sum(1 for iou in iou_value if iou>iou_threshold)
    robustness=iou_up_threshold/total_boxes
    return robustness
"""
--vot_path
----dir1
------pic1
------pic2
------groundtruth.txt
----dir2
"""
import os

robustness_list=[]
for dir in os.listdir(vot_path):
    if "list" in dir:continue
    anno_path=os.path.join(vot_path,dir,"groundtruth.txt")
    # anno_path="/home/cr7/python/dataset/test/VOT2020/ants1/groundtruth.txt"
    ground_truth_rect = read_file(str(anno_path))  ###vot_dataset.py
# print(ground_truth_rect)



    anno2_path="/media/gg/AC12-0E7B/vot20/{}.txt".format(dir)
    pred_box_list=[]
    with open(anno2_path,"r+") as file:
        lines=file.readlines()

        for line in lines:
            pred_box=list(map(int,line.strip().split("\t")))
            pred_box_list.append(pred_box)
    robustness=calculate_robustness(ground_truth_rect,pred_box_list)
    print(dir,robustness)
    robustness_list.append(robustness)
print(sum(robustness_list)/len(robustness_list))




