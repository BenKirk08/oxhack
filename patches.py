# face regions split the skin mask into 6 regions
# inner = the middle of  face -> the part a face swap replaces (cheeks, nose)
# outer = near the edge of the face -> usually left alone by face swap (forehead, jaw)
# each region = the outline around a set of facemesh points using cut down to skin pixels only

from dataclasses import dataclass

import cv2
import numpy as np

from signals import ColourBuffer, mean_colour  #one buffer per region

#face mesh points of each region
#all are kept away from the eyes, brows and mouth as move the most change in colour eg speaking, blinking
#gotten from claude the numbers
RIGHT_CHEEK_POINTS = [ 36, 47, 48, 49, 50, 64, 100, 101, 102, 117, 118, 119, 126, 129, 142, 165, 187, 203, 205, 206, 209 ]
LEFT_CHEEK_POINTS = [ 266, 277, 279, 280, 329, 330, 331, 346, 347, 348, 358, 371, 391, 411, 423, 425, 426 ]
NOSE_POINTS = [1, 3, 4, 5, 44, 45, 51, 195, 196, 197, 248, 274, 275, 281, 419, 457, 459]
FOREHEAD_POINTS = [10, 67, 69, 103, 104, 108, 109, 151, 297, 299, 332, 333, 337, 338]
RIGHT_JAW_POINTS = [58, 135, 136, 138, 150, 169, 170, 172, 202, 210, 212, 214]
LEFT_JAW_POINTS = [288, 364, 365, 367, 379, 394, 395, 397, 422, 430, 432, 434]
REGIONS = [  #(name, inner?, points) for every region
    ("right cheek", True, RIGHT_CHEEK_POINTS),
    ("left cheek", True, LEFT_CHEEK_POINTS),
    ("nose", True, NOSE_POINTS),
    ("forehead", False, FOREHEAD_POINTS),
    ("right jaw", False, RIGHT_JAW_POINTS),
    ("left jaw", False, LEFT_JAW_POINTS),
]
MIN_REGION_PIXELS = 100 #a region with fewer skin pixels than this in frame is dropped


@dataclass
class Region:
    #one face region in one frame
    # like recor: Region(name, inner, mask)
    name: str # e.g. "left cheek"
    inner: bool# True = inner face (face swap replaces it) False = outer face
    mask: np.ndarray #255 = this region's skin pixels, 0 = not


def region_masks(landmark_points: np.ndarray, skin: np.ndarray) -> list[Region]:
    # return the regions that have at least MIN_REGION_PIXELS skin pixels this frame
    regions = []
    x, y, width, height = cv2.boundingRect(landmark_points) #smallest box around the face
    right, bottom = x + width, y + height  #the boxs far edges
    x, y = max(x, 0), max(y, 0)  #keep the box inside the picture
    box_skin = skin[y:bottom, x:right] #the skin mask cut down to that box
    if box_skin.size == 0: # face completely outside the picture -> no regions
        return regions
    box_points = landmark_points - (x, y) #move the points so (0, 0) is the boxs corner
    taken = np.zeros_like(box_skin)  # pixels already in a region, so none counted twice
    for name, inner, points in REGIONS:
        box_mask = np.zeros_like(box_skin) # start all 0 (same size as the box)
        outline = cv2.convexHull(box_points[points]) #outline around the regions points
        cv2.fillConvexPoly(box_mask, outline, 255)  #fill the region 255
        box_mask[box_skin == 0] = 0  #keep skin pixels only
        box_mask[taken > 0] = 0  # and only ones no earlier region has
        taken[box_mask > 0] = 255  #mark them as taken
        if cv2.countNonZero(box_mask) >= MIN_REGION_PIXELS: #enough skin seen to be useful
            mask = np.zeros_like(skin)  # a whole picture mask all 0
            mask[y:bottom, x:right] = box_mask  # with the box's region pasted in
            regions.append(Region(name, inner, mask))
    return regions


def add_region_colours(buffers: dict[str, ColourBuffer], time_s: float, frame_bgr: np.ndarray, regions: list[Region]) -> None:
    #add each regions mean colour to its own buffer
    #all buffers stay on the same frames so non get lost
    names = [region.name for region in regions] #the regions seen this frame
    for name in list(buffers):  # list() = a copy of the keys, because we delete while looping
        if name not in names:
            del buffers[name]
    for region in regions:
        colour = mean_colour(frame_bgr, region.mask) #average BGR of this regions skin
        if colour is not None:
            if region.name not in buffers: #a new region or one that came back
                buffers[region.name] = ColourBuffer()
            buffers[region.name].add(time_s, colour)
