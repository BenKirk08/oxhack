#Drawing helpers
#remeber to make sure draws ONLY on copy and leaves the input frame untouched
import cv2
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

MESH_COLOUR = (0, 0, 0)  #BGR
MASK_COLOUR = (0, 0, 255)  #BGR
MASK_OPACITY = 0.4
MESH_CONNECTIONS: list = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION #state list as false error


def draw_face_mesh(frame: np.ndarray, landmark_points: np.ndarray) -> np.ndarray:
    #return a copy of frame with the face mesh drawn on it
    canvas = frame.copy() #keep original pixels same for colour sampling quality scores
    for connection in MESH_CONNECTIONS: #each connection is one edge of the mesh
        start = tuple(landmark_points[connection.start])  #cv2 -> points as tuples
        end = tuple(landmark_points[connection.end])
        cv2.line(canvas, start, end, MESH_COLOUR, 1, cv2.LINE_AA)  #1 px wide + anti-aliased (makes smooth says claude as wasnt b4)
    return canvas


def draw_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    #return a copy of frame with mask tinted on top -> semi-transparent
    canvas = frame.copy()
    tint = np.full_like(frame, MASK_COLOUR) #a frame sized image of 1 colour
    blended = cv2.addWeighted(frame, 1 - MASK_OPACITY, tint, MASK_OPACITY, 0) #mix the 2 images
    canvas[mask > 0] = blended[mask > 0] #only tint pixels inside the mask
    return canvas
