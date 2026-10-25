#same steps as live.py's loop (face -> mask -> skin filter -> regions -> pulse -> BPM), but draws nothing,
#so the Gradio app (and the video checker in step 10) can use it and all show the same numbers

from dataclasses import dataclass

import numpy as np

import patches  #for background_colours() + neck_colour() (steps 12-13)
from face import blinking_or_talking, create_landmarker, detect_face, face_mask, landmarks_to_pixels
from patches import Region, add_region_colours, region_masks
from signals import (
    MIN_SECONDS,
    ColourBuffer,
    RegionReader,
    RegionReading,
    combine_regions,
    green_pulse,
    mean_colour,
    pos_pulse
)
from skin import SkinReference, skin_mask


@dataclass
class FrameResult:
    #everything worked out for one frame ready to put on a webpage
    bpm: float | None # the BPM to show (None = show the stauts instead
    status: str  #e.g. "Calibrating 3/8 s", "No face" or "Hold still"
    collected: float  #seconds of skin colour collected so far
    pulse: np.ndarray | None  # the pulse wave to draw (None before there's a reading)
    readings: dict[str, RegionReading] #each regions own BPM + quality score
    landmark_points: np.ndarray | None  # face mesh points in pixels (None = no face)
    mask: np.ndarray | None #skin mask (None = no face)
    regions: list[Region]  #face regions found this frame
    colour: np.ndarray | None  # skins mean B, G, R this frame (None = left out)
    region_colours: dict[str, np.ndarray] #each regions mean BGR ({} = frame left out)
    backgrounds: dict[str, np.ndarray] #each background corners mean BGR ({} = none)
    neck: np.ndarray | None  # the necks mean BGR (None = not seen / frame left out)


class Pipeline:
    # one camera's worth of everything that has to be remembered between frames: its own face
    #tracker, colour buffers and settings. make one per camera, so two feeds shown side by
    # side never mix their samples

    def __init__(self) -> None:
        self.landmarker = create_landmarker()  #this feeds own tracker -> follows 1 face
        self.colour_buffer = ColourBuffer() #last 10s of the whole skins mean colour
        self.skin_reference = SkinReference()  # skin filters steady typical colour
        self.region_buffers: dict[str, ColourBuffer] = {}  #one colour buffer per region
        self.region_reader = RegionReader() #each regions pulse, BPM and quality
        self.use_pos = True  # True = POS (all 3 colours) False = green only
        self.use_skin_filter = True  #False -> whole face mask
        self.previous_timestamp_ms = -1 #mediapipe needs every frames time later then the last

    def process(self, frame_bgr: np.ndarray, time_s: float) -> FrameResult:
        #run one frame through every step (time_s = real clock time it was taken)
        timestamp_ms = max(int(time_s * 1000), self.previous_timestamp_ms + 1) #must increase
        self.previous_timestamp_ms = timestamp_ms
        seconds = timestamp_ms / 1000  # time used for every buffer
        result = detect_face(self.landmarker, frame_bgr, timestamp_ms)
        landmark_points = landmarks_to_pixels(result, frame_bgr.shape)  #None if no face
        mask = None
        regions: list[Region] = []
        colour = None #stays None if theres no face
        region_colours: dict[str, np.ndarray] = {}  # stays empty unless frame gets measured
        backgrounds: dict[str, np.ndarray] = {} #empty if no face
        neck = None  #None unless frame measured + neck can be seen
        if landmark_points is not None:  # face found
            mask = face_mask(landmark_points, frame_bgr.shape)
            if self.use_skin_filter:  #remove hair, glasses, shadows & glare
                mask = skin_mask(frame_bgr, mask, landmark_points, self.skin_reference, seconds)
            regions = region_masks(landmark_points, mask) #the 6 face regions
            backgrounds = patches.background_colours(frame_bgr, landmark_points) #no pulse check
            colour = mean_colour(frame_bgr, mask)  # average skin colour (None if no skin)
            if blinking_or_talking(result): #type: ignore #blinks & talking chnage colour a lot
                colour = None  #so frame left out like one with no skin
            if colour is not None: # only measure when theres skin to average
                self.colour_buffer.add(seconds, colour)
                add_region_colours(self.region_buffers, seconds, frame_bgr, regions)
                for name, buffer in self.region_buffers.items():  #read each regions colour back
                    if buffer.times and buffer.times[-1] == seconds:  #if it got one just now
                        region_colours[name] = buffer.colours[-1]  # (no extra work -> faster)
                skin_values = self.skin_reference.values if self.use_skin_filter else None #ref
                neck = patches.neck_colour(frame_bgr, landmark_points, skin_values) #the neck
        bpm = None  #no reading unless enough data
        pulse = None
        collected = self.colour_buffer.seconds() # secs of samples so far
        status = f"Calibrating {collected:.0f}/{MIN_SECONDS:.0f} s"  #eg "Calibrating 3/8 s"
        if landmark_points is None:  #nothing to measure w/o a face
            status = "No face"
        elif collected >= MIN_SECONDS: # enough samples for a reading
            if self.use_pos:  #POS -> all 3 colours (main method)
                pulse = pos_pulse(self.colour_buffer)
            else:  #green only (step 4 baseline)
                pulse = green_pulse(self.colour_buffer)
            self.region_reader.step(self.region_buffers, self.use_pos) #one region per frame
            bpm = combine_regions(self.region_reader.readings)  # good regions combinded
            status = "Hold still" #shown instead of number if no region good enough
        return FrameResult(  #name=value -> args by name, clearer than 12 in a row
            bpm=bpm,
            status=status,
            collected=collected,
            pulse=pulse,
            readings=self.region_reader.readings,
            landmark_points=landmark_points,
            mask=mask,
            regions=regions,
            colour=colour,
            region_colours=region_colours,  #step 12 -> real vs fake features
            backgrounds=backgrounds,  # for the background check
            neck=neck,  #step 13: neck (a face swap never replaces it)
        )

    def set_skin_filter(self, on: bool) -> None:
        # switch skin filter on/off -> mask changes so start measuring again
        if on != self.use_skin_filter:  #only if it actaully changed
            self.use_skin_filter = on
            self.colour_buffer.clear()
            self.region_buffers.clear()

    def close(self) -> None:
        #free the face tracker when camera is finished with
        self.landmarker.close()
