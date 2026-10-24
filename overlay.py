# Drawing helpers
# remeber to make sure draws ONLY on copy and leaves the input frame untouched
import cv2
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

MESH_COLOUR = (0, 0, 0)  # BGR
MASK_COLOUR = (0, 0, 255)  # BGR
MASK_OPACITY = 0.4
MESH_CONNECTIONS: list = (
    FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
)  # state list as false error


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
    tint = np.full_like(frame, MASK_COLOUR)  # a frame sized image of 1 colour
    blended = cv2.addWeighted(frame, 1 - MASK_OPACITY, tint, MASK_OPACITY, 0)  # mix the 2 images
    canvas[mask > 0] = blended[mask > 0]  # only tint pixels inside the mask
    return canvas


PANEL_WIDTH = 900 # width of the BPM panel beside the camera, in pixels
PANEL_COLOUR = (30, 30, 30)  # dark grey (BGR)
TEXT_COLOUR = (255, 255, 255) # white (BGR)
WAVE_COLOUR = (80, 220, 80) # green (BGR)
SCALE_FROM_MEDIAN = 2.0  # y scale = 2x the waves median size (room for bigger beats)


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
