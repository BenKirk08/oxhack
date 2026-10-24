# OpenCV development window -> webcam feed w/ the face tracking
# Keys: l = show/hide the mesh lines
#      k = show/hide the mask
#      s = skin filter on/off (compare the mask with and without it)
#      q = quit
import time  # theme of this hackathon:)

import cv2

# from other files
from face import create_landmarker, detect_face, face_mask, landmarks_to_pixels
from overlay import (
    draw_bpm_panel,
    draw_face_mesh,
    draw_mask,
    y_scale,  # £ sets the wave's y axis scale once
)
from signals import MIN_SECONDS, ColourBuffer, estimate_bpm, green_pulse, mean_colour
from skin import skin_mask

WINDOW_NAME = "PulseCheck"
CAMERA_INDEX = 0


# show camera with mesh on face until q -> quit
def main() -> None:
    capture = cv2.VideoCapture(CAMERA_INDEX)  # 0 means the default camera i.e. my mac camera
    if not capture.isOpened():  # no camera / camera permission denied
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    show_mesh = True
    show_mask = True
    use_skin_filter = True
    previous_timestamp_ms = -1
    colour_buffer = ColourBuffer()  # the last 10 s of mean skin colour
    wave_scale = None  # £ the wave's y axis scale: set once from the first 8 s, then never changed
    try:
        with create_landmarker() as landmarker:  # with -> javas try (...reader) then closes on exit
            while True:
                frame_found, frame = capture.read()  # read() returns (success, frame) tuple
                if not frame_found:  # camera disconnected / stream ended
                    break
                now_ms = int(
                    time.monotonic() * 1000
                )  # real time (not using fps as links with theme)
                timestamp_ms = max(now_ms, previous_timestamp_ms + 1)  # check must increase
                previous_timestamp_ms = timestamp_ms

                result = detect_face(landmarker, frame, timestamp_ms)
                landmark_points = landmarks_to_pixels(result, frame.shape)  # None if no face
                display = frame  # overlay return copy so frame remains untouched
                if landmark_points is not None:  # face found can dp
                    mask = face_mask(landmark_points, frame.shape)
                    if use_skin_filter:
                        mask = skin_mask(
                            frame, mask, landmark_points
                        )  # remove hair, glasses, shadows, really bright ligth
                    colour = mean_colour(frame, mask)  # average skin colour (None if no skin)
                    if colour is not None:  # only measure when theres skin to average
                        colour_buffer.add(timestamp_ms / 1000, colour)  # ms -> seconds
                    if show_mask:
                        display = draw_mask(display, mask)  # tint the face skin
                    if show_mesh:
                        display = draw_face_mesh(
                            display, landmark_points
                        )  # mesh over the tint if enabled
                bpm = None  # no reading unless there's enough data
                pulse = None
                collected = colour_buffer.seconds()  # seconds of samples so far
                status = (
                    f"Calibrating {collected:.0f}/{MIN_SECONDS:.0f} s"  # e.g. 3/8 s -> ints / ints
                )
                if landmark_points is None:  # nothing to measure without a face
                    status = "No face"  # shown instead of a BPM
                elif collected >= MIN_SECONDS:  # enough samples for a reading
                    pulse = green_pulse(colour_buffer)  # the filtered pulse wave
                    bpm = estimate_bpm(pulse)  # strongest pulse frequency x 60
                    if wave_scale is None:  # £ the first wave (made from the first 8 s)
                        wave_scale = y_scale(pulse)  # £ set the y axis scale from it, just once
                display = draw_bpm_panel(
                    display, bpm, pulse, status, wave_scale
                )  # panel beside the camera
                cv2.imshow(WINDOW_NAME, display)  # show the overlaid copy on the OpenCV screen

                key = cv2.waitKey(1) & 0xFF  # gets key and keeps just the key (Google says to)
                if key == ord("q"):  # ord for charcter code
                    break
                if key == ord("l"):
                    show_mesh = not show_mesh  # toggle the mesh lines
                if key == ord("k"):
                    show_mask = not show_mask  # toggle the mask lines
                if key == ord("s"):
                    use_skin_filter = not use_skin_filter  # toggle filtered skin
                    colour_buffer.clear()  # the mask changed, so start measuring again
    finally:
        capture.release()  # turns the camera off
        cv2.destroyAllWindows()  # close OpenCV window


if __name__ == "__main__":
    main()
