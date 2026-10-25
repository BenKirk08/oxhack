# Drawing helpers
# remeber to make sure draws ONLY on copy and leaves the input frame untouched
import cv2
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

from patches import Region
from signals import MIN_SNR_DB, RESAMPLE_HZ, RegionReading

MESH_COLOUR = (0, 0, 0)  # BGR
MASK_COLOUR = (0, 0, 255)  # BGR
MASK_OPACITY = 0.4
MESH_CONNECTIONS: list = (FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION) # state list as false error


def draw_face_mesh(frame: np.ndarray, landmark_points: np.ndarray) -> np.ndarray:
    # return a copy of frame with the face mesh drawn on it
    canvas = frame.copy()  # keep original pixels same for colour sampling quality scores
    for connection in MESH_CONNECTIONS:  # each connection is one edge of the mesh
        start = tuple(landmark_points[connection.start])  # cv2 -> points as tuples
        end = tuple(landmark_points[connection.end])
        cv2.line(
            canvas, start, end, MESH_COLOUR, 1, cv2.LINE_AA
        )  # 1 px wide + anti-aliased (makes smooth says claude as wasnt b4)
    return canvas


def draw_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # return a copy of frame with mask tinted on top -> semi-transparent
    canvas = frame.copy()
    x, y, width, height = cv2.boundingRect(mask)  # smallest box around the mask -> save power
    if width == 0 or height == 0:  #empty
        return canvas
    area = canvas[y : y + height, x : x + width]  # a view (not a copy)-> changing it changes canvas
    inside = mask[y : y + height, x : x + width] > 0  #true for masks pixels in  box
    tint = np.full_like(area, MASK_COLOUR)  # a frame sized image of 1 colour
    blended = cv2.addWeighted(area, 1 - MASK_OPACITY, tint, MASK_OPACITY, 0)  #mix the 2 images
    area[inside] = blended[inside]  # only tint pixels inside the mask
    return canvas


PANEL_WIDTH = 900 # width of the BPM panel beside the camera, in pixels
PANEL_COLOUR = (30, 30, 30)  # dark grey (BGR)
TEXT_COLOUR = (255, 255, 255) # white (BGR)
WAVE_COLOUR = (80, 220, 80) # green (BGR)
SCALE_FROM_MEDIAN = 2.0  # y scale = 2x the waves median size (room for bigger beats)
EDGE_SECONDS = 2.0  #filter keeps re-working the newest and oldest 2 s as POS, so they aren't drawn


def y_scale(pulse: np.ndarray) -> float:
    #median ignores a few huge values, so a spike in first 8s can't spoil it
    return SCALE_FROM_MEDIAN * float(np.median(np.abs(pulse))) or 1.0  # or 1.0 -> a flat wave


def draw_bpm_panel(frame: np.ndarray, bpm: float | None, pulse: np.ndarray | None, status: str, scale: float | None = None) -> np.ndarray:
    # shows bpm/calibrating
    height = frame.shape[0]  # the panel is as tall as the camera picture
    panel = np.full((height, PANEL_WIDTH, 3), PANEL_COLOUR, dtype=np.uint8)  # blank panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bpm is not None:  # a reading
        text = f"{bpm:.0f} BPM"  # 0dp
        cv2.putText(panel, text, (20, 80), font, 2, TEXT_COLOUR, 3, cv2.LINE_AA)  # big
    else:  # no reading yet: show the status, smaller
        cv2.putText(panel, status, (20, 70), font, 0.8, TEXT_COLOUR, 2, cv2.LINE_AA)  # smaller
    if pulse is not None and len(pulse) > 1:  # theres wave to draw
        edge = int(EDGE_SECONDS * RESAMPLE_HZ)  #2 s worth of samples (60)
        if len(pulse) > 3 * edge:  # enough wave to trim shld alw be true once enough data
            pulse = pulse[edge:-edge]  # slice off both ends -> only steady middle is drawn
        middle = height / 2  # vertically middel of panel
        half_height = height / 6  # and fills a third of its height **
        biggest = float(np.max(np.abs(pulse))) or 1.0  # Can't divide by 0
        if scale is not None:  # a fixed y axis scale so no re-scaling (did 1st time and was messy)
            biggest = scale
        xs = np.linspace(10, PANEL_WIDTH - 10, len(pulse))  # spread the samples across the panel
        ys = middle - pulse / biggest * half_height  # minus: y counts down from the top
        room = 2 * half_height  # 2/3 of the panel's height so bigger beats fit
        ys = np.clip(ys, middle - room, middle + room)  # a spike stops at the edge of that space
        points = np.column_stack([xs, ys]).astype(np.int32)  # (x, y) pairs for OpenCV
        cv2.polylines(panel, [points], False, WAVE_COLOUR, 2, cv2.LINE_AA)  # false = open line
    return np.hstack([frame, panel])  # the panel to the right of the frame (a new picture)


def draw_method_label(display: np.ndarray, method: str) -> np.ndarray:
    # write which pulse method is on (POS / Green)
    x = display.shape[1] - PANEL_WIDTH + 20  # the panel is the right-most PANEL_WIDTH pixels
    text = f"{method}  (m = switch)" #e.g. "POS  (m = switch)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display, text, (x, 125), font, 0.8, TEXT_COLOUR, 2, cv2.LINE_AA) #under the BPM
    return display


INNER_COLOUR = (0, 165, 255)  # orange (BGR): inner face regions
OUTER_COLOUR = (255, 120, 0) #blue (BGR): outer face regions
REGION_OPACITY = 0.5 #how strongly the regions are tinted


def draw_regions(frame: np.ndarray, regions: list[Region]) -> np.ndarray:
    #tint faec regions
    canvas = frame.copy()  #draw on a copy so camera frame stays untouched
    for region in regions: #for every region
        #this is mostly same as the face mask
        x, y, width, height = cv2.boundingRect(region.mask)#smallest box around the region
        area = canvas[y : y + height, x : x + width]
        inside = region.mask[y : y + height, x : x + width] > 0  #True for the region's pixels
        colour = INNER_COLOUR if region.inner else OUTER_COLOUR
        tint = np.full_like(area, colour) # a box-sized picture of that one colour
        blended = cv2.addWeighted(area, 1 - REGION_OPACITY, tint, REGION_OPACITY, 0) #fast mix
        area[inside] = blended[inside] #only regions own pixels get the mix
    return canvas


GOOD_COLOUR = (80, 220, 80)  #green (BGR) -> a region with a clear pulse
BAD_COLOUR = (80, 80, 230) #red (BGR) -> a region without one

def draw_region_scores(display: np.ndarray, readings: dict[str, RegionReading]) -> np.ndarray:
    #write each regions quality score
    x = display.shape[1] - 330  #near the right edge of the panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row, (name, reading) in enumerate(readings.items()):  #row = 0, 1, 2, ...
        score = round(reading.snr_db)
        colour = GOOD_COLOUR if score >= MIN_SNR_DB else BAD_COLOUR
        text = f"{name}: {score:+.0f} dB" if score != 0 else f"{name}: {score:.0f} dB"
        # e.g. "left cheek: +6 dB"
        cv2.putText(display, text, (x, 40 + 30 * row), font, 0.7, colour, 2, cv2.LINE_AA)
    return display
#NB snr_db compares how storng heartbeat is compared w the noise
