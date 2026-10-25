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


#background patch (step 12) -> a control that has no pulse
#beating at the faces rate = flicker or camera noise, not a heartbeat
BACKGROUND_SHARE = 0.2  #each box is 20% of the pictures width and height
BACKGROUND_CLEARANCE = 0.5  # a box has to stay this many face widths away from the face box
BACKGROUND_NAMES = ("background left", "background right") #one track per top corner


def background_boxes(  #step 12
    landmark_points: np.ndarray, frame_shape: tuple[int, ...]  # face points + picture size
) -> dict[str, tuple[int, int, int, int]]: #name: (x, y, width, height)
    #the 2 top corners of the picture, if half a face width clear of the face
    # corners are nearly always wall / ceiling -> lit like the face but no pulse
    #and dont move w/ the head (beside the head would catch the ears which pulse)
    height, width = frame_shape[:2]  #picture size
    face_x, face_y, face_width, _ = cv2.boundingRect(landmark_points)  #_ = face height (not needed)
    clearance = BACKGROUND_CLEARANCE * face_width # gap to keep (room for hair)
    box_width = round(BACKGROUND_SHARE * width)  #each boxs size
    box_height = round(BACKGROUND_SHARE * height) #and height
    boxes = {}  # boxes clear of the face, by name
    corners = zip(BACKGROUND_NAMES, (0, width - box_width), strict=True)  #(name, x) pairs
    for name, box_x in corners:  #top left corner then top right
        left_of_face = box_x + box_width < face_x - clearance  # ends b4 face starts
        right_of_face = box_x > face_x + face_width + clearance #starts afer face ends
        above_face = box_height < face_y - clearance  #ends above the face (and hair)
        if left_of_face or right_of_face or above_face:  # clear of face one way or another
            boxes[name] = (box_x, 0, box_width, box_height)  #full size from top edge
    return boxes  #0, 1 or 2  boxes


def background_colours(  #step 12
    frame_bgr: np.ndarray, landmark_points: np.ndarray  #picture + face points
) -> dict[str, np.ndarray]:  # name: mean BGR
    #mean BGR of each clear corner by name (empty if the face leaves no room)
    #each corner its own track -> colours differ so mixng would make the mean jump
    colours = {} #name: mean BGR
    boxes = background_boxes(landmark_points, frame_bgr.shape)  #corners clear of the face
    for name, (x, y, box_width, box_height) in boxes.items():  # .items() -> (name, box) pairs
        box = frame_bgr[y : y + box_height, x : x + box_width]  #a view of the box (not a copy)
        colours[name] = np.array(cv2.mean(box)[:3]) #[:3] = B, G, R only
    return colours  # 0, 1 or 2 corners


#neck patch -> step 13s positive control
#swap tools paste a new face from eyebrows to chin but NEVER the neck
# so the neck keeps the persons real pulse
#clear neck pulse + none in the face = "a swap removed the pulse" not "a poor video"
CHIN = 152  #face mesh point at bottom of the chin
JAW_CORNERS = [172, 397]  # face mesh points at the 2 jaw corners
NECK_TOP = 0.05  #neck box starts this share of the face hieght below chin
NECK_BOTTOM = 0.4 #and ends this far below
NECK_COLOUR_TOLERANCE = 0.06  # how far a pixels red/green share can be from the face skin
NECK_BRIGHTNESS = (0.4, 1.6)  #allowed brightness as a share of the faces (necks are shaded)
MIN_NECK_PIXELS = 1500 #fewer neck pixels than this -> neck not used


def neck_colour(  #step 13
    frame_bgr: np.ndarray,  #the picture
    landmark_points: np.ndarray,  # face mesh points in pixels
    skin_values: np.ndarray | None, #skin filters typical values (None = off)
) -> np.ndarray | None:  #mean BGR (None = no neck reading)
    #mean B, G, R of the neck skin just under the chin (None if not enough)
    #only pixels that look like this faces skin count, not hoodie / scarf / beard
    if skin_values is None:  #no typical skin colour yet (or skin filter off)
        return None
    typical_brightness, typical_red, typical_green = skin_values[:3]  #[:3] -> 1st 3
    height, width = frame_bgr.shape[:2]  # picture size
    face_height = cv2.boundingRect(landmark_points)[3] #[3] = face box height
    chin_y = landmark_points[CHIN][1]  #[1] = y of chin point
    left, right = sorted(landmark_points[JAW_CORNERS][:, 0])  # jaw corners x, left first
    top = max(round(chin_y + NECK_TOP * face_height), 0)  #kept inside the picture
    bottom = min(round(chin_y + NECK_BOTTOM * face_height), height) #and the bottom
    left, right = max(left, 0), min(right, width)  # same sideways
    if bottom - top < 2 or right - left < 2:  #neck out of pictrue -> give up
        return None  #no reading
    box = frame_bgr[top:bottom, left:right]  #view of box under chin (no copy)
    smooth = cv2.GaussianBlur(box, (5, 5), 0).astype(np.float32) # blur like the skin fliter
    total = np.maximum(smooth.sum(axis=2), 1)  #B + G + R per pixel (min 1 -> no dividing by 0)
    red_share = smooth[:, :, 2] / total  #BGR so 2 = red
    green_share = smooth[:, :, 1] / total  #1 = green
    colour_gap = np.hypot(red_share - typical_red, green_share - typical_green) #vs the face
    low, high = NECK_BRIGHTNESS  # unpack both limits
    brightness = total / 3  #avg of B, G and R
    bright_enough = brightness >= low * typical_brightness  #not too dark (shaded is ok)
    not_too_bright = brightness <= high * typical_brightness # and not too bright
    skin = (colour_gap <= NECK_COLOUR_TOLERANCE) & bright_enough & not_too_bright  #all 3 pass
    if np.count_nonzero(skin) < MIN_NECK_PIXELS:  #too little neck skin to turst
        return None
    return box[skin].mean(axis=0)  # skin pixels mean BGR (from unblurred box)
