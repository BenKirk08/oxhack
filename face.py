#MediaPipe face landmarker -> model downlod, setup, detection and  face mask

import urllib.request
from pathlib import Path

import cv2

#easier names
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerResult,
    FaceLandmarksConnections,
)

#addrees and Path for googles AI model for the face tracker
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path(__file__).parent / "data" / "face_landmarker.task"

CUTOUT_MARGIN_FRACTION = 0.03 #fraction of the face width added around each cut-out -> edge of eyes/lips cut too

def _landmark_indices(connections: list) -> list[int]:
    #return sorted +unique landmark indices used by a list of mesh connections
    unique_indices = set() #set -> removes repeats
    for edge in connections:
        for index in (edge.start, edge.end):
            unique_indices.add(index)
    return sorted(unique_indices)


#claude used to figure out how face is mapped with googles tool, find correct landmarks where to cut out -> constant on all, checked with camera mesh in live.py
FACE_OVAL = _landmark_indices(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)
LEFT_NOSTRIL = [309, 392, 305, 290, 250, 458, 459, 461, 462]
RIGHT_NOSTRIL = [79, 166, 75, 60, 20, 238, 239, 241, 242]
CUTOUT_REGIONS = [  #non-skin parts removed from the face mask
    _landmark_indices(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE),
    _landmark_indices(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE),
    _landmark_indices(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW),
    _landmark_indices(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW),
    _landmark_indices(FaceLandmarksConnections.FACE_LANDMARKS_LIPS),
    LEFT_NOSTRIL,
    RIGHT_NOSTRIL,
]

def ensure_model() -> Path:
    #download the face landmarker model on first run and return path
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(exist_ok = True)  #create path in the file if not there
        partial_path = MODEL_PATH.with_name(MODEL_PATH.name + ".part") #.part then renamed as sometime interrupted the actual model
        urllib.request.urlretrieve(MODEL_URL, partial_path)  #fetch model from google
        partial_path.replace(MODEL_PATH) #now rename after the download finished
    return MODEL_PATH


def create_landmarker() -> FaceLandmarker: #A tool that find the landmarkers on face wuickly -> kinda like a reader
    #create face landmarker in video mode
    options = vision.FaceLandmarkerOptions(
        base_options = BaseOptions(model_asset_path = str(ensure_model())), #model using
        running_mode = vision.RunningMode.VIDEO, #says that 1 frame will come after this 1
        num_faces = 1, #oinly for 1 face
        output_face_blendshapes = True, # returns score liek how closed in eye & how open is jaw, needed quality rating
    )
    return FaceLandmarker.create_from_options(options)


#frame_bgr is the frame input -> bgr is order colour in
def detect_face(landmarker: FaceLandmarker, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceLandmarkerResult:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #RGB needed by mediapipe so convert
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data = frame_rgb) #fromat that mediapipe accepts
    return landmarker.detect_for_video(image, timestamp_ms) #returns the landmarker tool


def landmarks_to_pixels(result: FaceLandmarkerResult, frame_shape: tuple[int, ...]) -> np.ndarray | None: #none when no face detetcted
    #return the 1st faces landmarks as (N, 2) int array of pixel ([...,...], ..)
    if not result.face_landmarks: #no face
        return None
    height, width = frame_shape[:2] #gets display size
    face_landmarks = result.face_landmarks[0]  #only 1 face
    rows = []
    for landmark in face_landmarks:
        rows.append((landmark.x, landmark.y)) #row per point this is the (N,2) shaep
    points = np.array(rows)
    return (points * (width, height)).astype(np.int32) #for mediapipes normalised 0-1 pixels


def face_mask(landmark_points: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
    #returns a uint8 mask (255 = face skin, 0 = not) -> does on the whole face
    height, width = frame_shape[:2] #displkay size
    mask = np.zeros((height, width), dtype=np.uint8)  #start all 0
    face_outline = cv2.convexHull(landmark_points[FACE_OVAL])  #outline around the oval points
    cv2.fillConvexPoly(mask, face_outline, 255)  #fill the whole face 255

    cutouts = np.zeros_like(mask)  #same shape and type as mask (SECOND MASK)
    for this_region in CUTOUT_REGIONS:
        region_outline = cv2.convexHull(landmark_points[this_region])  #outline around this region
        cv2.fillConvexPoly(cutouts, region_outline, 255)  #mark the region for removal (255 as will remove 255 places from actual mask)

    face_width = cv2.boundingRect(face_outline)[2]  #boundingRect gives (x, y, width, height) so [2] picks width
    margin = max(1, round(face_width * CUTOUT_MARGIN_FRACTION))  #at least 1 pixel
    brush_size = 2 * margin + 1 #for removing
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (brush_size, brush_size))  #round brush
    cutouts = cv2.dilate(cutouts, brush)  # grow each cut-out by the margin
    mask[cutouts > 0] = 0 #change 255 to 0
    return mask #final combined mask
