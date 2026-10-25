# adaptive skin filter keep face pixels that look like the person cheeks
# so things like hair, beard, glasses and shadows are removed -> is only a trim (uses the mask from face.py)
# compares both brightness + colour
#3 places (cheeks and nose) so 1 thing wont throw it off

import cv2
import numpy as np

#landmark points around the middle of each cheek where almost always skin
LEFT_CHEEK = [346, 347, 330, 266, 425, 411, 352]  # retireved these values again from claude
RIGHT_CHEEK = [117, 118, 101, 36, 205, 187, 123]
NOSE_RIDGE = [197, 3, 51, 5, 281, 248]  # brigde of the nose but  below where glasses sit

BRIGHTNESS_TOLERANCE = 0.3 #30% darker / brighter than the cheeks
COLOUR_TOLERANCE = 0.06  # how far a pixels red/green from cheeks
COVERED_CHEEK_LIMIT = 0.5 #  a cheek under half as bright as the other is covered (hair, hand)
MIN_CHEEK_PIXELS = 50  # fewer than this and cheek not trusted
BLUR_SIZE = 5 # blur copy first socamera grain dont make pixels flicker in & out
MIN_COLOURFULNESS = 0.35  # need 35% of the skin's colourfulness (drops grey/white things)
REFERENCE_SECONDS = 1.0  # how long typical until face lost


class SkinReference:
    #remembers skin filters typical values (eg brightness, red amount, green amount)
    # averaged over REFERNCE_SECONDS so no flicker -> changes dont happen duratically epsecially on boundary

    def __init__(self) -> None:
        self.values: np.ndarray | None = None  # the averaged values (None before 1st frame)
        self.time_s = 0.0 #when they were last updated, in seconds

    def smooth(self, time_s: float, values: np.ndarray) -> np.ndarray:
        #move averaged values partly towards this frames values and return them
        #how far depends on real tiem since last frame so any frame rate works
        gap = time_s - self.time_s #secs since last update
        if self.values is None or gap > REFERENCE_SECONDS:  #first frame / face lost
            smoothed = values  #start again from this frames values
        else:
            share = 1 - np.exp(-gap / REFERENCE_SECONDS)  # 1/30 s gap -> 3% of the way there
            smoothed = self.values + share * (values - self.values)
        self.values = smoothed
        self.time_s = time_s
        return smoothed


def cheek_masks(landmark_points: np.ndarray, face_mask: np.ndarray) -> list[np.ndarray]:
    #return one mask per cheek (left, right)
    masks = []
    for cheek in (LEFT_CHEEK, RIGHT_CHEEK):
        mask = np.zeros_like(face_mask)  # start all 0 (same size as the face mask)
        cheek_outline = cv2.convexHull(landmark_points[cheek]) #outline around the cheek points
        cv2.fillConvexPoly(mask, cheek_outline, 255) #fill the cheek 255
        mask[face_mask == 0] = 0# drop any part outside the face mask
        masks.append(mask)
    return masks


def skin_mask(frame_bgr: np.ndarray, face_mask: np.ndarray, landmark_points: np.ndarray, reference: SkinReference | None = None, time_s: float = 0.0) -> np.ndarray:
    # return the face mask eith removed parts that arent like cheek
    x, y, width, height = cv2.boundingRect(face_mask)  # smallest box around the face mask -> keep fast
    skin = np.zeros_like(face_mask) # all 0 outside the box
    if width == 0 or height == 0:  #empty face mask -> nothing to filter
        return skin
    skin[y : y + height, x : x + width] = _filter_face_box(
        frame_bgr[y : y + height, x : x + width],
        face_mask[y : y + height, x : x + width],
        landmark_points - (x, y),  # move the points so (0, 0) is the box's top-left corner
        reference, #the steady typical skin values (None = measure this frame only)
        time_s  # this frames time in seconds
    )
    return skin


def _filter_face_box(frame_bgr: np.ndarray, face_mask: np.ndarray, landmark_points: np.ndarray, reference: SkinReference | None = None, time_s: float = 0.0) -> np.ndarray:
    #the actuall filter run on the box around the face
    smooth = cv2.GaussianBlur(frame_bgr, (BLUR_SIZE, BLUR_SIZE), 0).astype(np.float32) #decimals -> using guassian distribution
    total_rgb = np.maximum(smooth.sum(axis=2), 1)  #B + G + R per pixel (min 1: no dividing by 0)
    brightness = total_rgb / 3  # average of the 3 channels
    red_amount = smooth[:, :, 2] / total_rgb  #BGR so 2 = red (0 index)
    green_amount = smooth[:, :, 1] / total_rgb  #1 = green

    cheeks = [] #true/false masks of cheeks that can be seen
    cheek_brightnesses = [] #median brightness of eahc  cheeks
    cheek_reds = []  # median red share of each of those cheeks
    cheek_greens = []# median green share of each of those cheeks
    for mask in cheek_masks(landmark_points, face_mask):
        inside = mask > 0 #true for every pixel in this cheek
        if np.count_nonzero(inside) >= MIN_CHEEK_PIXELS:  #skip cheek turned out of view
            cheeks.append(inside)
            cheek_brightnesses.append(np.median(brightness[inside]))  # median ignores odd pixels
            cheek_reds.append(np.median(red_amount[inside]))  # cheeks typical red mount
            cheek_greens.append(np.median(green_amount[inside]))  #and green
    if not cheeks:  #no cheeks
        return np.zeros_like(face_mask)

    nose = np.zeros_like(face_mask)  # same steps as cheek_masks for the bridge of the nose
    cv2.fillConvexPoly(nose, cv2.convexHull(landmark_points[NOSE_RIDGE]), 255)  # fill it 255
    nose_inside = (nose > 0) & (face_mask > 0)  #only nose pixels inside the face mask
    patch_brightnesses = list(cheek_brightnesses) #copy of cheeks brightness
    patch_reds = list(cheek_reds)  #copy of cheeks red amount
    patch_greens = list(cheek_greens)  #copy of cheeks green amount
    if np.count_nonzero(nose_inside) >= MIN_CHEEK_PIXELS:  #only if enough nose can be seen
        patch_brightnesses.append(np.median(brightness[nose_inside]))  #the noses brightness
        patch_reds.append(np.median(red_amount[nose_inside]))  # its red amount
        patch_greens.append(np.median(green_amount[nose_inside]))  #green
    #copy for medians of all
    typical_brightness = np.median(patch_brightnesses)  #median value of cheeks + nose
    typical_red = np.median(patch_reds)  #so one odd patch can't change it
    typical_green = np.median(patch_greens)

    darker_cheek = min(cheek_brightnesses) #shaded cheek
    if reference is not None:  # use values averaged over ab 1 s
        frame_values = np.array([typical_brightness, typical_red, typical_green, darker_cheek])
        steady = reference.smooth(time_s, frame_values)  #this frame blended into the average
        typical_brightness, typical_red, typical_green, darker_cheek = steady  #unpack all 4

    darkest_skin = max(darker_cheek, COVERED_CHEEK_LIMIT * typical_brightness) # shade (not cover)
    lowest = (1 - BRIGHTNESS_TOLERANCE) * darkest_skin  #allow shaded cheek
    brightness_close = brightness >= lowest  #in range brightness per pixel
    #numpy actually requires & and ~ not 'and' and 'not'!!!!

    red_difference = red_amount - typical_red  #the truested cheeks red amount
    green_difference = green_amount - typical_green #and for green
    colour_close = np.hypot(red_difference, green_difference) <= COLOUR_TOLERANCE  #use both colours
    colourfulness = np.hypot(red_amount - 1 / 3, green_amount - 1 / 3) #distance from grey
    skin_colourfulness = np.hypot(typical_red - 1 / 3, typical_green - 1 / 3)  # and for skin
    colour_close &= colourfulness >= MIN_COLOURFULNESS * skin_colourfulness  # also drop greys

    filtered = face_mask.copy()  # start from the face mask
    filtered[~(brightness_close & colour_close)] = 0  #remove pixels failing any check
    return filtered
