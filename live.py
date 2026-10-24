#OpenCV development window -> webcam feed w/ the face tracking
#Keys: l = show/hide the mesh lines
#      k = show/hide the mask
#      q = quit
import time #theme of this hackathon:)
import cv2

#from other files
from face import create_landmarker, detect_face, face_mask, landmarks_to_pixels
from overlay import draw_face_mesh, draw_mask

WINDOW_NAME = "PulseCheck"
CAMERA_INDEX = 0

#show camera with mesh on face until q -> quit
def main() -> None:
    capture = cv2.VideoCapture(CAMERA_INDEX)  #0 means the default camera i.e. my mac camera
    if not capture.isOpened():  #no camera / camera permission denied
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    show_mesh = True
    show_mask = True
    previous_timestamp_ms = -1
    try:
        with create_landmarker() as landmarker: #with -> javas try (...reader) then closes on exit
            while True:
                frame_found, frame = capture.read()  #read() returns (success, frame) tuple
                if not frame_found: #camera disconnected / stream ended
                    break
                now_ms = int(time.monotonic() * 1000)  #real time (not using fps as links with theme)
                timestamp_ms = max(now_ms, previous_timestamp_ms + 1) #check must increase
                previous_timestamp_ms = timestamp_ms

                result = detect_face(landmarker, frame, timestamp_ms)
                landmark_points = landmarks_to_pixels(result, frame.shape)  #None if no face
                display = frame  #overlay return copy so frame remains untouched
                if landmark_points is not None: #face found can dp
                    mask = face_mask(landmark_points, frame.shape)
                    if show_mask:
                        display = draw_mask(display, mask) #tint the face skin
                    if show_mesh:
                        display = draw_face_mesh(display, landmark_points)  #mesh over the tint if enabled
                cv2.imshow(WINDOW_NAME, display) #show the overlaid copy on the OpenCV screen

                key = cv2.waitKey(1) & 0xFF # gets key and keeps just the key (Google says to)
                if key == ord("q"): #ord for charcter code
                    break
                if key == ord("l"):
                    show_mesh = not show_mesh  #toggle the mesh lines
                if key == ord("k"):
                    show_mask = not show_mask  #toggle the mask lines
    finally:
        capture.release() #turns the camera off
        cv2.destroyAllWindows()  #close OpenCV window


if __name__ == "__main__":
    main()
