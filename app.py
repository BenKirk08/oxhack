#run: .venv/bin/python app.py  then open http://127.0.0.1:7860  (Ctrl+C stops it)

import html
import logging
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI, Response
from gradio import themes  #gradio looks (gr.themes upsets vscodes type checker)
from matplotlib.figure import Figure  #graphs w/o opening a window (safe in a server)

from detector import Features, Verdict, face_is_clear
from overlay import (
    EDGE_SECONDS,
    INNER_COLOUR,
    OUTER_COLOUR,
    draw_face_mesh,
    draw_mask,
    draw_regions,
    y_scale
)
from patches import REGIONS
from pipeline import FrameResult, Pipeline
from presage_check import PresageReading, presage_bpm, run_presage
from signals import MIN_SECONDS, MIN_SNR_DB, RESAMPLE_HZ, RegionReading
from video import VideoResult, analyse_video, pulse_times

CAMERA_INDEX = 0  # 0 = default camera (same as live.py)
HOST = "127.0.0.1" #only this laptop can open the page
PORT = 7860 #page is at http://127.0.0.1:7860
CAMERA_ROUTE = "/camera.jpg"  #web address the page gets each camera pic from
CAMERA_JS = (Path(__file__).parent / "assets" / "camera.js").read_text()  # browser code
CSS_PATH = Path(__file__).parent / "assets" / "app.css" #page colours, sizes + layout
JPEG_QUALITY = 80  #0-100 -> higher = sharper but more data every frame
MIRROR = True #flip left-right so it moves like a mirror
PAGE_EVERY_SECONDS = 0.1  # how often BPM, wave + region scores upadte on the page
FPS_WINDOW_SECONDS = 2.0  #cameras fps counted over the last 2s
OVERLAYS = ["Mask", "Regions", "Mesh"] #what can be drawn on the camera
WAVE_WIDTH = 600  #waves drawing size (browser streches it to fit the card)
WAVE_HEIGHT = 150
SNR_RANGE_DB = 10.0  # quality bars go -10 dB (empty) to +10 dB (full)
MIN_ZOOM = 0.25 #wave zoom slider goes from 0.25x = a quarter as tall
MAX_ZOOM = 4.0 #up to 4x as tall (1x = auto scale)
LOGGER = logging.getLogger(__name__)  #prints error reports in the termianl
PLOT_GREEN = "#16a34a"  # pulse plot wave colour
PLOT_RED = "#e11d48" #pulse plot BPM line colour (rose red)
PLOT_ORANGE = "#f59e0b" #marks the edge where the wave runs off the graph
FONT = "Geist"  # page font from google fonts (others to try: "Inter", "Space Grotesk")
MONO_FONT = "Geist Mono"  #font for code-like text (needs internet, else system fonts)
WAVE_PERCENTILE = 85  #85% of the waves samples are closer to the middle than this
WAVE_HEADROOM = 1.3 #graph goes 1.3x past that so only big swings run off
MIN_PLOT_SECONDS = 5.0  #if cutting calibrating leaves less wave than this keep it all
VERDICT_ICONS = {"real": "✓", "fake": "✕", "unsure": "?"}  # the round badge by the verdict
FEATURE_NAMES = { #each features row in "How it decided" -> (what it is, format)
    "windows": ("10 s windows measured", "{:.0f}"),
    "face_seconds": ("Face measured for", "{:.0f} s"),
    "usable_share": ("Frames measured", "{:.0%}"),
    "face_bpm": ("Face's strongest beat rate", "{:.0f} BPM"),
    "face_snr_db": ("Face's pulse strength", "{:+.1f} dB"),
    "bpm_stability": ("Face's BPM change between windows", "±{:.1f} BPM"),
    "regions_agree": ("Face patches on the face's rate", "{:.0%}"),
    "region_corr": ("Face patches' waves alike", "{:.2f}"),
    "region_bpm_spread": ("Face patches' BPM spread", "±{:.1f} BPM"),
    "inner_bpm": ("Cheeks and nose's heart rate", "{:.0f} BPM"),
    "outer_bpm": ("Forehead and jaw's heart rate", "{:.0f} BPM"),
    "inner_snr_db": ("Cheeks and nose's pulse strength", "{:+.1f} dB"),
    "outer_snr_db": ("Forehead and jaw's pulse strength", "{:+.1f} dB"),
    "inner_outer_corr": ("Inner vs outer face's waves alike", "{:.2f}"),
    "neck_bpm": ("Neck's heart rate", "{:.0f} BPM"),
    "neck_snr_db": ("Neck's pulse strength", "{:+.1f} dB"),
    "neck_stability": ("Neck's BPM change between windows", "±{:.1f} BPM"),
    "neck_face_snr_gap": ("Neck's pulse strength minus the face's", "{:+.1f} dB"),
    "face_neck_bpm_diff": ("Face vs neck BPM gap", "{:.0f} BPM"),
    "background_snr_db": ("Background's 'pulse' strength", "{:+.1f} dB"),
    "background_share": ("Background beating with the face", "{:.0%}"),
    "background_bpm": ("Background's beat rate", "{:.0f} BPM"),
}
PROGRESS = gr.Progress()  #gradio swaps in a progres bar here each Analyse click
PRESAGE_EVERY_SECONDS = 1.0  #how often the presage card checks for presages answer
PRESAGE_JOBS: dict[str, Future[PresageReading | str]] = {}  # video path -> its presage run
PRESAGE_LOCK = threading.Lock()  #1 presage at a time (each 1 is a whole node program)


@dataclass
class Snapshot:
    #newest camera frame + everything worked out from it
    #a new one every frame + never changed after so any thread can read it safely
    number: int  #counts up by 1 each frame so a browser knows theres a new one
    frame: np.ndarray #untouched camera picture (BGR)
    result: FrameResult # pipelines BPM, wave, regions etc
    use_pos: bool  #which method made it (True = POS, False = green)
    scale: float | None  #that methods fixed y scale (None b4 its 1st wave)
    frames_per_second: float # how fast the camera is actually running


class LiveCamera:

    def __init__(self, capture: cv2.VideoCapture) -> None:
        self.capture = capture
        self.pipeline = Pipeline()  #same steps as live.py
        self.use_pos = True  #page settings -> camera thread applies them each frame
        self.use_skin_filter = True
        self.overlays = list(OVERLAYS)  # what to draw (Mask / Regions / Mesh)
        self.wave_zoom = 1.0 #"Wave zoom" slider (1 = auto y scale)
        self.scales: dict[bool, float | None] = {True: None, False: None}  #POS?: its y scale
        self.frame_times: deque[float] = deque()  #when recent frames arived (deque = fast queue)
        self.snapshot: Snapshot | None = None  # newest frame (None till the 1st)
        self.error = "" #last error msg shown on the page ("" = all fine)
        self.new_snapshot = threading.Condition() #lets browsers sleep till the next frame
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)  #daemon -> cant block exit

    def run(self) -> None:
        #the camera thread -> read a frame, measure it, share it, again till stop()
        number = 0
        while self.running:
            frame_found, frame = self.capture.read()  #waits for the cameras next pciture
            if not frame_found:  # camera disconnected
                self.error = "The camera stopped sending pictures"
                break
            now_s = time.monotonic() #real time it arrived (never assume a fixed fps)
            try:  #one bad frame musnt freeze the page so errors get caught + shown
                snapshot = self.measure(number, frame, now_s)
            except Exception as error: #any kind of error
                message = f"{type(error).__name__}: {error}"  #eg "IndexError: index 720 ..."
                if message != self.error:  #report each new error once not 30x a second
                    LOGGER.exception("frame failed") # full error report in the terminal
                self.error = message
                continue
            self.error = "" #this frame worked
            with self.new_snapshot:  #"with" locks it while the new snapshot gets swapped in
                self.snapshot = snapshot
                self.new_snapshot.notify_all()  #wake every browser waiting for a frame
            number += 1

    def measure(self, number: int, frame: np.ndarray, now_s: float) -> Snapshot:
        # put 1 frame through the pipeline w/ the pages current settings
        self.pipeline.use_pos = self.use_pos
        self.pipeline.set_skin_filter(self.use_skin_filter)  #starts again if it got swtiched
        result = self.pipeline.process(frame, now_s)
        use_pos = self.pipeline.use_pos
        if result.pulse is not None and self.scales[use_pos] is None:  #this methods 1st wave
            self.scales[use_pos] = y_scale(result.pulse) #its y scale, set once (like live.py)
        fps = self.frames_per_second(now_s)
        return Snapshot(number, frame, result, use_pos, self.scales[use_pos], fps)

    def frames_per_second(self, now_s: float) -> float:
        #how many frames a sec the camera is actually giving (over FPS_WINDOW_SECONDS)
        self.frame_times.append(now_s)
        while self.frame_times[0] < now_s - FPS_WINDOW_SECONDS:  #forget older frames
            self.frame_times.popleft() #popleft() removes the oldest (left) item
        if len(self.frame_times) < 2:  # need 2 frames to measure a gap
            return 0.0
        return (len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])

    def next_picture(self, after: int = -1) -> Response:
        #what the server sends for CAMERA_ROUTE -> 1st frame newer than "after"
        #w/ overlays, as one jpeg. camera.js asks again as soon as each one arrives
        # so it plays like a video in every browser (safari too)
        with self.new_snapshot:  #"with" locks it like javas synchronized
            if not self.has_new(after):  #sleep till camera thread has a newer frame
                self.new_snapshot.wait(timeout=1) # (or 1 s goes by)
        snapshot = self.snapshot
        if snapshot is None: #no frame yet (camera still starting)
            return Response(status_code=204)  #204 = "nothing yet" -> page asks again
        picture = draw_camera(snapshot, self.overlays)
        _, jpeg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        headers = {"X-Frame-Number": str(snapshot.number), "Cache-Control": "no-store"}
        return Response(jpeg.tobytes(), media_type="image/jpeg", headers=headers)

    def has_new(self, shown: int) -> bool:
        #True when theres a frame newer than shown or its time to stop
        return not self.running or (self.snapshot is not None and self.snapshot.number != shown)

    def set_method(self, method: str) -> None:
        #pages "Pulse method" choice (used from next frame)
        self.use_pos = method == "POS"

    def set_skin_filter(self, choice: str) -> None:
        # "Skin filter" choice (On / Off)
        self.use_skin_filter = choice == "On"

    def set_overlays(self, overlays: list[str]) -> None:
        #"Show on the camera" ticks
        self.overlays = list(overlays)  #list() = its own copy

    def set_wave_zoom(self, zoom: float) -> None:
        #"Wave zoom" slider
        self.wave_zoom = zoom

    def settings(self) -> tuple[str, str, list[str], float]:
        # current settings so a page opened later shows the right choises
        method = "POS" if self.use_pos else "Green"
        skin_filter = "On" if self.use_skin_filter else "Off"
        return method, skin_filter, self.overlays, self.wave_zoom

    def page_update(self) -> tuple[str, str]:
        #right side of the page from the newest snapshot (10x a second)
        snapshot = self.snapshot
        readings = {}  #no region scores till theres a reading (same as live.py)
        if snapshot is not None and snapshot.result.pulse is not None:
            readings = snapshot.result.readings
        return monitor_html(snapshot, self.error, self.wave_zoom), regions_html(readings)

    def stop(self) -> None:
        #stop the camera thread then turn camera off + free the face tracker
        self.running = False
        with self.new_snapshot:
            self.new_snapshot.notify_all() #wake waiting browsers so they can finish
        self.thread.join(timeout=5)  # wait (up to 5s) for the thread to finish its frame
        self.capture.release() #camera light off
        self.pipeline.close()


def draw_camera(snapshot: Snapshot, overlays: list[str]) -> np.ndarray:
    #camera picture w/ the chosen overlays (overlay.py draws on copies so frame stays clean)
    picture = snapshot.frame
    result = snapshot.result
    if result.landmark_points is not None and result.mask is not None:  #a face found
        if "Mask" in overlays:
            picture = draw_mask(picture, result.mask)  #tint the skin
        if "Regions" in overlays:
            picture = draw_regions(picture, result.regions) # orange = inner, blue = outer
        if "Mesh" in overlays:
            picture = draw_face_mesh(picture, result.landmark_points)
    if MIRROR:
        picture = cv2.flip(picture, 1) #1 = flip left-right (new picture)
    return picture


def css_colour(bgr: tuple[int, int, int]) -> str:
    #an opencv colour (blue, green, red) as a web colour eg "rgb(255, 165, 0)"
    blue, green, red = bgr
    return f"rgb({red}, {green}, {blue})"


def wave_svg(pulse: np.ndarray | None, scale: float | None) -> str:
    #pulse wave as an SVG line (picture made of shapes so its sharp at any size)
    #trimmed + scaled the same as live.pys panel
    heights = np.zeros(2)  #no wave yet -> flat line
    if pulse is not None and len(pulse) > 1:
        edge = int(EDGE_SECONDS * RESAMPLE_HZ) #2s of samples (ends still changing)
        if len(pulse) > 3 * edge:  #enough wave to trim
            pulse = pulse[edge:-edge] # only the steady middle gets drawn
        heights = np.clip(pulse / (scale or y_scale(pulse)), -2, 2)  #±2 = top / bottom edge
    xs = np.linspace(0, WAVE_WIDTH, len(heights))  #spread samples across the width
    ys = WAVE_HEIGHT / 2 - heights * WAVE_HEIGHT / 4 #minus -> SVG y counts down from top
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))  #"x,y x,y"
    middle = WAVE_HEIGHT / 2
    return (
        f"<svg viewBox='0 0 {WAVE_WIDTH} {WAVE_HEIGHT}' preserveAspectRatio='none'>"
        f"<line x1='0' y1='{middle}' x2='{WAVE_WIDTH}' y2='{middle}'/>"  #zero line
        f"<polyline points='{points}'/></svg>"  # the wave itself
    )


def monitor_html(snapshot: Snapshot | None, error: str, zoom: float = 1.0) -> str:
    #the dark "heart monitor" card -> big BPM (or whats happening instead), the method,
    # camera fps, a calibrating bar + the pulse wave
    headline = "<div class='status'>Starting camera…</div>"  #b4 the 1st frame
    details = ""
    progress = ""
    pulse, scale = None, None
    if snapshot is not None:
        result = snapshot.result
        pulse, scale = result.pulse, snapshot.scale
        if scale is not None: #zoom: 2x = half the scale = wave twice as tall
            scale = scale / zoom
        method = "POS" if snapshot.use_pos else "Green"
        details = f"{method} · {snapshot.frames_per_second:.0f} frames/s"
        face_found = result.landmark_points is not None
        if result.bpm is not None:  # a reading -> the big number
            headline = f"<div class='bpm'>{result.bpm:.0f}<small>BPM</small></div>"
        elif face_found and result.collected >= MIN_SECONDS:  #"Hold still" in amber
            headline = f"<div class='status warn'>{result.status}</div>"
        else: #"No face" or "Calibrating 3/8 s"
            headline = f"<div class='status'>{result.status}</div>"
        if face_found and result.collected < MIN_SECONDS:  #calibrating -> show how far along
            share = 100 * result.collected / MIN_SECONDS  #as a %
            progress = f"<div class='progress'><div style='width:{share:.0f}%'></div></div>"
    if error:  #somethin went wrong -> show it (terminal has the full report)
        details = f"<span class='error'>{html.escape(error)}</span>" # escape -> < > shown as text
    return (
        f"<div class='label'><span>Heart rate</span><span>{details}</span></div>"
        f"{headline}{progress}{wave_svg(pulse, scale)}"
    )


def regions_html(readings: dict[str, RegionReading]) -> str:
    #one row per face region -> colour on the camera, quality bar, SNR + own BPM
    # green bar = good enough to count towards the BPM, amber = left out
    rows = []
    for name, inner, _ in REGIONS:  #_ = not needed here (the regions points)
        colour = css_colour(INNER_COLOUR if inner else OUTER_COLOUR)
        reading = readings.get(name) #.get() gives None if no reading for it
        quality, fill, score, bpm = "missing", 0.0, "–", ""  # not seen / not measured yet
        if reading is not None:
            quality = "good" if reading.snr_db >= MIN_SNR_DB else "bad"  #as in combine_regions
            share = (reading.snr_db + SNR_RANGE_DB) / (2 * SNR_RANGE_DB) #-10 dB = 0, +10 = 1
            fill = 100 * min(max(share, 0), 1)  #as a %, kept 0-100
            whole = round(reading.snr_db)  # whole dB
            score = f"{whole:+d} dB" if whole != 0 else "0 dB" #+d -> always shows the sign
            bpm = f"{reading.bpm:.0f} BPM"
        rows.append(
            f"<div class='region {quality}'><i style='background:{colour}'></i>"
            f"<span>{name}</span><div class='bar'><div style='width:{fill:.0f}%'></div></div>"
            f"<b>{score}</b><small>{bpm}</small></div>"
        )
    inner_colour, outer_colour = css_colour(INNER_COLOUR), css_colour(OUTER_COLOUR)
    legend = (
        f"<div class='legend'><i style='background:{inner_colour}'></i>inner face (cheeks, nose) "
        f"<i style='background:{outer_colour}'></i>outer face (forehead, jaw): a face swap "
        "usually replaces both</div>"
    )
    return f"<div class='label'>Face regions</div>{''.join(rows)}{legend}"


def video_summary_html(result: VideoResult | None) -> str:
    #dark card on the video check tab -> the videos BPM (or why there isnt one) + facts
    label = "<div class='label'><span>Video heart rate</span><span>{}</span></div>"
    if result is None:  #nothing analysed yet
        return label.format("") + "<div class='status'>Upload a video, then Analyse</div>"
    method = "POS" if result.use_pos else "Green"
    details = f"{method} · {result.seconds:.0f} s · {result.frames} frames"
    clear = face_is_clear(result.features)  #verdicts own test for a pulse we can trust
    if result.bpm is not None and clear: # median of every BPM shown while it played
        headline = f"<div class='bpm'>{result.bpm:.0f}<small>BPM</small></div>"
    elif result.bpm is not None:  #some readings but to weak to trust -> no big number
        headline = "<div class='status warn'>No clear pulse</div>"
        details += f" · readings wandered around {result.bpm:.0f} BPM"  #+= adds to the end
    elif result.face_share == 0:
        headline = "<div class='status warn'>No face found</div>"
    elif result.seconds < MIN_SECONDS: #never got past calibrating
        headline = f"<div class='status warn'>Too short (needs {MIN_SECONDS:.0f} s)</div>"
    else:  #calibrated but no region was ever good enough
        headline = "<div class='status warn'>No clear pulse</div>"
    facts = (
        f"<div class='facts'>Face found {result.face_share:.0%} of the time · BPM shown "
        f"{result.reading_share:.0%} of the time after calibrating · moved ±"
        f"{result.bpm_spread:.1f} BPM</div>"
    )
    return label.format(details) + headline + facts


def feature_rows(features: Features) -> str:
    # the "How it decided" table -> every features name + value
    rows = []
    for name, value in asdict(features).items():  #asdict -> dataclass as {name: value}
        text, number_format = FEATURE_NAMES[name]
        shown = number_format.format(value) if np.isfinite(value) else "–"  #– = not measured
        rows.append(f"<tr><td>{text}</td><td>{shown}</td></tr>")
    return f"<table>{''.join(rows)}</table>"


def check_list(checks: list[tuple[str, bool | None]]) -> str:
    #checks behind the verdict: ✓ passed, ✗ failed, – couldnt be checked (eg no neck)
    items = []
    for text, passed in checks:
        mark, css_class = "–", "skip"
        if passed is True:
            mark, css_class = "✓", "pass"
        elif passed is False:
            mark, css_class = "✗", "fail"
        items.append(f"<li class='{css_class}'><b>{mark}</b>{html.escape(text)}</li>")
    return f"<ul class='checks'>{''.join(items)}</ul>"


def verdict_html(verdict: Verdict | None, features: Features | None = None) -> str:
    if verdict is None:  # nothing analysed yet
        return (
            "<div class='verdict waiting'><div class='label'><span>Verdict</span></div>"
            "<div class='headline'>Upload a video, then Analyse</div></div>"
        )
    score = "" #no score if the quality check stopped it
    if verdict.score is not None:
        score = (
            f"<div class='score'><span>Authenticity score {verdict.score:.0f}/100 (from the "
            "pulse evidence only)</span>"
            f"<div class='bar'><div style='width:{verdict.score:.0f}%'></div></div></div>"
        )
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in verdict.reasons)
    table = feature_rows(features) if features is not None else ""
    pos_note = (
        "<p>The verdict always uses POS: the Pulse method only changes the heart-rate card, the "
        "plot and the region scores.</p>"
    )
    details = (
        f"<details><summary>How it decided</summary>{check_list(verdict.checks)}{table}"
        f"{pos_note}</details>"
    )
    return (
        f"<div class='verdict {verdict.label}'>"
        f"<div class='label'><span>Verdict</span><span>{html.escape(verdict.method)}</span></div>"
        f"<div class='headline'><i>{VERDICT_ICONS[verdict.label]}</i>{verdict.headline}</div>"
        f"{score}<ul class='reasons'>{reasons}</ul>{details}"
        f"<p class='note'>{html.escape(verdict.caveat)}</p></div>"
    )


def wave_limit(wave: np.ndarray) -> float:
    #how far up / down the pulse graph goes -> 1.3x the waves 85th percentile height
    #not its biggest spike so normal beats fill the graph + the few big swings run off
    # not the median either, stretches w/ no face would drag it right down
    heights = np.abs(wave[np.isfinite(wave)])  #each samples distnace from the middle
    if len(heights) == 0: #nothing to measure
        return 1.0
    return WAVE_HEADROOM * float(np.percentile(heights, WAVE_PERCENTILE)) or 1.0  #flat -> 1.0


def pulse_figure(result: VideoResult) -> Figure | None:
    #the pulse plot -> whole videos pulse wave (top) + BPM shown over time (bottom)
    if result.pulse is None:  # too short / no face -> nothing to plot
        return None
    wave_times = pulse_times(result) #time of every sample in the wave
    start = result.calibrated_time  #cut off calibrating secs (no BPM can show there)
    if start is None or wave_times[-1] - start < MIN_PLOT_SECONDS:  #never / too little left
        start = wave_times[0] # so show the whole wave
    figure = Figure(figsize=(9, 5.5), layout="constrained")  #size in inches
    wave_axes, bpm_axes = figure.subplots(2, 1, sharex=True)  #2 graphs sharing the time axis
    wave_axes.plot(pulse_times(result), result.pulse, color=PLOT_GREEN, linewidth=1.2)
    wave_axes.set_xlim(start, wave_times[-1])  #sharex -> BPM graph gets the same time range
    visible = wave_times >= start #numpy -> one True/False per sample (True = shown)
    shown_times, shown_wave = wave_times[visible], result.pulse[visible]  # the shown part
    limit = wave_limit(shown_wave)  #from the shown part only (cut off start cant count)
    wave_axes.set_ylim(-limit, limit) #bigger swings run off the top + bottom
    run_off = np.abs(shown_wave) > limit  # True where the wave goes past the edge
    marks = np.where(run_off, np.sign(shown_wave) * limit, np.nan)  #where(test, yes, no)
    wave_axes.plot(  #nan = no mark, square markers show a 1 sample run off too
        shown_times, marks, color=PLOT_ORANGE, linewidth=4, marker="s", markersize=3, clip_on=False
    )
    wave_axes.set_ylabel("Pulse wave")
    wave_axes.set_yticks([]) #waves height has no useful units
    bpm_axes.plot(result.times, result.bpms, color=PLOT_RED, linewidth=2)  # nan = a gap
    if result.bpm is not None:  #dahsed line at the median w/ its value in a key
        median = f"median {result.bpm:.0f} BPM"
        bpm_axes.axhline(result.bpm, color="grey", linestyle="--", linewidth=1, label=median)
        bpm_axes.legend(loc="upper right")
        shown = [bpm for bpm in result.bpms if not np.isnan(bpm)]
        bpm_axes.set_ylim(min(shown) - 10, max(shown) + 10) #10 BPM room above + below
    bpm_axes.set_ylabel("BPM shown")
    bpm_axes.set_xlabel("Time in the video (s)")
    for axes in (wave_axes, bpm_axes):
        axes.grid(alpha=0.3) #faint grid lines
        axes.spines[["top", "right"]].set_visible(False)  # no box lines top + right
    return figure


def analyse_upload(
    path: str | None, method: str, progress: gr.Progress = PROGRESS
) -> tuple[str, str, Figure | None, str]:
    #the Analyse button -> check the uploaded video + fill in the video check tab
    if not path:  #nothing uploaded yet
        raise gr.Error("Upload a video first.") # shows a red msg on the page
    start_presage(path)  #presage starts now on its own thread, alongside PulseCheck below
    try:
        result = analyse_video(
            Path(path), method == "POS", lambda share: progress(share, desc="Analysing")
        )
    except ValueError as error:  #a file opencv cant read
        raise gr.Error(str(error)) from error
    verdict = verdict_html(result.verdict, result.features) #verdict box
    return verdict, video_summary_html(result), pulse_figure(result), regions_html(result.regions)


def clear_results() -> tuple[str, str, None, str]:
    # video check tabs right side like it is b4 anything gets analysed
    return verdict_html(None), video_summary_html(None), None, regions_html({})


def presage_html(
    reading: PresageReading | None = None, problem: str = "", status: str = ""
) -> str:
    #card at the bottom of the video check tab -> Presages own BPM for the same video
    #presage = a company whose API measures pulse from video, a 2nd opinion to compare w/
    label = "<div class='label'><span>Presage API heart rate</span><span>{}</span></div>"
    if problem:  #presage couldnt run -> say why (amber)
        return label.format("") + f"<div class='text'>{html.escape(problem)}</div>"
    if reading is None:  # not analysed yet or still measuring (grey)
        text = html.escape(status or "Analyse a video to compare")
        return label.format("") + f"<div class='text waiting'>{text}</div>"
    bpm, sure = presage_bpm(reading)
    if bpm is None:  # ran fine but never got a reading -> show its most common warning
        hint = max(reading.hints, key=reading.hints.get, default="")  #type: ignore -> most frames
        text = "No reading. " + (f"Presage says: {hint}" if hint else "It needs about 12 s of a "
                                 "still, well-lit face with the upper chest in view.")
        return label.format("") + f"<div class='text'>{html.escape(text)}</div>"
    note = "" if sure else "low confidence"  #none of its readings were stable
    return label.format(note) + f"<div class='number'>{bpm:.0f}<small>BPM</small></div>"


def presage_job(job: Future[PresageReading | str], path: Path) -> None:
    #runs on its own thread -> presages reading (or what went wrong) goes into job
    with PRESAGE_LOCK:  #with -> lock always let go after, like javas synchronized
        try:
            job.set_result(run_presage(path))
        except RuntimeError as error: # no node / no key / slow internet -> said on the card
            job.set_result(str(error))
        except Exception:  #anything else -> full error in the terminal + a short note
            LOGGER.exception("presage failed")
            job.set_result("Presage failed (the details are in the terminal).")


def start_presage(path: str) -> None:
    #analyse_upload calls this 1st -> presage starts on the video in the background
    #a video already measured (or still measuring) isnt done again but a failed 1 is retried
    job = PRESAGE_JOBS.get(path)
    if job is None or (job.done() and isinstance(job.result(), str)):
        job = Future()
        PRESAGE_JOBS[path] = job
        thread = threading.Thread(target=presage_job, args=(job, Path(path)), daemon=True)
        thread.start()  #daemon -> Ctrl+C on the app never waits for presage to finish


def presage_status(path: str | None) -> str:
    #timer tick -> the card for the video thats loaded right now (another videos never shows)
    job = PRESAGE_JOBS.get(path) if path else None
    if job is None:  #not analysed yet
        return presage_html()
    if not job.done():
        return presage_html(status="Measuring with Presage...")
    found = job.result()
    return presage_html(problem=found) if isinstance(found, str) else presage_html(found)


def build_page(camera: LiveCamera) -> gr.Blocks:
    #the webpage built from python calls (gradio turns them into html)
    with gr.Blocks(title="PulseCheck", analytics_enabled=False) as page:  #"with" -> put inside
        gr.HTML(
            "<h1>PulseCheck</h1><p>Live heart rate from tiny colour changes in your skin</p>",
            elem_id="header",
        )
        with gr.Tab("Live BPM"), gr.Row(equal_height=False):  #side by side in the tab
            with gr.Column(scale=3): #left -> scale = how wide compared to the other column
                gr.HTML("<canvas></canvas>", elem_id="camera", js_on_load=CAMERA_JS)  #JS draws
                with gr.Row():  # settings under the camera: 2 side by side
                    method = gr.Radio(["POS", "Green"], value="POS", label="Pulse method")
                    skin_filter = gr.Radio(["On", "Off"], value="On", label="Skin filter")
                overlays = gr.CheckboxGroup( #then this one under them, full width
                    OVERLAYS, value=OVERLAYS, label="Show on the camera"
                )
            with gr.Column(scale=2):  #right -> the readings
                monitor = gr.HTML(monitor_html(None, ""), elem_id="monitor")
                zoom = gr.Slider(  # small sldier under the wave (taller/shorter)
                    MIN_ZOOM, MAX_ZOOM, value=1.0, step=0.25, label="Wave zoom"
                )
                regions = gr.HTML(regions_html({}), elem_id="regions")
        with gr.Tab("Video Check"), gr.Row(equal_height=False): #check a video file
            with gr.Column(scale=2): #left: the video + Analyse button
                upload = gr.Video(label="Video to check", sources=["upload"])
                video_method = gr.Radio(["POS", "Green"], value="POS", label="Pulse method")
                analyse = gr.Button("Analyse", variant="primary") #primary = coloured button
            with gr.Column(scale=3): # right: what it found
                video_verdict = gr.HTML(verdict_html(None), elem_id="verdict") 
                video_summary = gr.HTML(video_summary_html(None), elem_id="video-summary")
                pulse_plot = gr.Plot(label="Pulse and BPM over the video")
                video_regions = gr.HTML(regions_html({}), elem_id="video-regions")
                presage_card = gr.HTML(presage_html(), elem_id="presage")  #presages BPM at bottom
        timer = gr.Timer(PAGE_EVERY_SECONDS)  #ticks 10x a second while page is open
        timer.tick(camera.page_update, outputs=[monitor, regions]) # refresh the right side
        method.change(camera.set_method, inputs=method)  #tell the camera ab each change
        skin_filter.change(camera.set_skin_filter, inputs=skin_filter)
        overlays.change(camera.set_overlays, inputs=overlays)
        zoom.change(camera.set_wave_zoom, inputs=zoom)
        page.load(camera.settings, outputs=[method, skin_filter, overlays, zoom])  #settings
        upload.change(  #new video -> clear the old answer so it cant sit next to the new one
            clear_results,
            outputs=[video_verdict, video_summary, pulse_plot, video_regions],
            concurrency_id="video_check", #waits for a running Analyse then clears after
        )
        analyse.click(  # check the video then fill in the 4 parts on the right
            analyse_upload,
            inputs=[upload, video_method],
            outputs=[video_verdict, video_summary, pulse_plot, video_regions],
            concurrency_id="video_check",  #shares a queue w/ the clear above (1 at a time)
        )
        presage_timer = gr.Timer(PRESAGE_EVERY_SECONDS)  #checks presage once a second
        presage_timer.tick(  #card follows whichever video is loaded, hidden -> no loading flash
            presage_status, inputs=upload, outputs=presage_card, show_progress="hidden"
        )
    return page


def page_theme() -> themes.Base:
    #gradios built in "Soft" look in teal w/ the FONT + MONO_FONT
    #no internet -> fonts cant download so the browser uses the system ones
    font = themes.GoogleFont(FONT, weights=(400, 600, 700))  #normal, semi-bold + bold
    mono = themes.GoogleFont(MONO_FONT)
    return themes.Soft(
        primary_hue="teal",
        font=[font, "ui-sans-serif", "system-ui", "sans-serif"], #then fallbacks in order
        font_mono=[mono, "ui-monospace", "monospace"],
    )


def main() -> None:
    capture = cv2.VideoCapture(CAMERA_INDEX)  #open the camera here like live.py
    if not capture.isOpened():  # no camera / permission denied
        print(f"Could not open camera {CAMERA_INDEX}.")
        return
    camera = LiveCamera(capture)
    camera.thread.start()  #start measuring now so its calibarted by the time the page opens
    try: #try/finally -> camera always turned off, even after an error / Ctrl+C
        server = FastAPI()  #the web server (gradio is built on fastapi)
        server.add_api_route(CAMERA_ROUTE, camera.next_picture)  #camera pictures address
        server = gr.mount_gradio_app(  # gradio page at "/" on the same server
            server,
            build_page(camera),
            path="/",
            server_name=HOST,
            server_port=PORT,
            footer_links=[], #no "Built with Gradio" links at bottom
            theme=page_theme(),  #built in look (teal) + the FONT
            css_paths=[CSS_PATH],
        )
        print(f"PulseCheck is running: open http://{HOST}:{PORT} in your browser (Ctrl+C stops)")
        uvicorn.run(server, host=HOST, port=PORT, log_level="warning", timeout_graceful_shutdown=1)
    except KeyboardInterrupt:  #Ctrl+C = normal way to stop so no error msg
        pass
    finally:
        camera.stop()

if __name__ == "__main__":  # only when run directly (python app.py) not imported
    main()
