import argparse
import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from detector import (
    FACE,
    FEATURES_VERSION,
    MIN_CLIPS_PER_LABEL,
    MODEL_FEATURES,
    MODEL_PATH,
    NECK,
    Features,
    Verdict,
    add_sample,
    fit_model,
    judge,
    load_model,
    quality_problem,
    save_model,
    video_features,
)
from patches import REGIONS
from pipeline import Pipeline
from signals import MIN_SECONDS, RESAMPLE_HZ, ColourBuffer, RegionReading, green_pulse, pos_pulse

DATA_PATH = Path(__file__).parent / "data"
LABELS = ["real", "fake"] #batch reads data/real/ + data/fake/ (folder = label)
RESULTS_PATH = DATA_PATH / "results.csv"  # where batch mode writes its table
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"} #file endings that are videos
FALLBACK_FPS = 30.0  #only if a file has no frame times or fps


@dataclass
class VideoResult:
    #everything worked out for 1 whole video
    path: Path
    use_pos: bool  # True = POS, False = green only
    seconds: float  # how long the video is
    frames: int #frames read
    face_share: float #share of frames with a face (0-1)
    reading_share: float  # share past calibrating that showed a BPM (not "Hold still")
    bpm: float | None  #median of every BPM shown (None = never showed one)
    bpm_spread: float #how much shown BPM moved (std dev)
    times: list[float]  # every frames time in the file (s)
    bpms: list[float]  #BPM shown at each time (nan = none shown)
    pulse: np.ndarray | None #whole videos pulse wave (None if too short / no face)
    pulse_start: float # time of the waves 1st sample (then 1/RESAMPLE_HZ s each)
    calibrated_time: float | None  #when calibrating finished -> 1st time a BPM could show
    regions: dict[str, RegionReading]  #each regions median BPM + median score (SNR dB)
    features: Features #real vs fake features
    verdict: Verdict #real / fake / unsure w/ score + reasons


def video_files(folder: Path) -> list[Path]:
    #videos in a folder in name order (empty list if no folder)
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in VIDEO_TYPES)


def analyse_video(
    path: Path, use_pos: bool = True, on_progress: Callable[[float], object] | None = None
) -> VideoResult:
    # put every frame through the pipeline then sum up the whole clip
    #on_progress (if given) gets called with how far along it is, 0 to 1
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():  #missing file or a format opencv cant read
        raise ValueError(f"Could not open the video {path.name}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # only for progress (can be 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS #only used if a frame has no time
    pipeline = Pipeline() #fresh one per video (own tracker + bufers)
    pipeline.use_pos = use_pos
    times: list[float] = []
    bpms: list[float] = []
    faces = 0  #frames w/ a face
    checked = 0  # frames past calibrating (number or "Hold still")
    colour_times: list[float] = [] #every sample measrued, for the whole wave
    colours: list[np.ndarray] = []
    region_snrs: dict[str, list[float]] = {name: [] for name, _, _ in REGIONS} #name: scores
    region_bpms: dict[str, list[float]] = {name: [] for name, _, _ in REGIONS}
    last_readings = None  # readings already counted (change once a round)
    calibrated_time: float | None = None  #when calibrating finished (None = not yet)
    tracks: dict[str, ColourBuffer] = {} #every sample of each patch, whole video
    try:  #try/finally -> file + tracker always closed even after an error (like java)
        while True:
            frame_found, frame = capture.read()
            if not frame_found:  #end of vidoe
                break
            time_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 # this frames own time
            if times and time_s <= times[-1]:  #file has no proper times -> use its fps
                time_s = times[-1] + 1 / fps
            result = pipeline.process(frame, time_s)
            times.append(time_s)
            bpms.append(result.bpm if result.bpm is not None else np.nan)  #nan = no number
            faces += result.landmark_points is not None
            if result.colour is not None: #a measured sample (not gated / missing)
                colour_times.append(time_s)
                colours.append(result.colour)
                add_sample(tracks, FACE, time_s, result.colour)  # whole skins track
            for name, region_colour in result.region_colours.items():  #each regions track
                add_sample(tracks, name, time_s, region_colour)
            for name, background in result.backgrounds.items(): #each corner = own track
                add_sample(tracks, name, time_s, background)  # (for the flicker check)
            if result.neck is not None: #neck track
                add_sample(tracks, NECK, time_s, result.neck)
            if result.pulse is not None:  #past calibrating, like live.pys scores
                checked += 1
                if calibrated_time is None:  # 1st frame past calibrating
                    calibrated_time = time_s #video check plot starts here
                if result.readings is not last_readings:  #"is" = same object (like == in java)
                    last_readings = result.readings
                    for name, reading in result.readings.items():
                        region_snrs[name].append(reading.snr_db)
                        region_bpms[name].append(reading.bpm)
            if on_progress is not None and total_frames > 0:
                on_progress(min(len(times) / total_frames, 1.0))
    finally:
        capture.release()
        pipeline.close()
    shown = [bpm for bpm in bpms if not np.isnan(bpm)]  # only frames that showed a number
    pulse = None
    if len(colour_times) > 1 and colour_times[-1] - colour_times[0] >= MIN_SECONDS:
        everything = ColourBuffer()  #all samples (add() would only keep the last 10s)
        everything.times, everything.colours = colour_times, colours
        pulse = pos_pulse(everything) if use_pos else green_pulse(everything)
    regions = {} #each regions middle vaules as a RegionReading so app.py can show it
    for name, scores in region_snrs.items():
        if scores:  # only regions that got scored
            middle_bpm = float(np.median(region_bpms[name]))
            regions[name] = RegionReading(np.zeros(0), middle_bpm, float(np.median(scores)))
    usable_share = len(colour_times) / len(times) if times else 0.0  #share of frames measured
    features = video_features(tracks, usable_share) # (always POS, see detector.py)
    verdict = judge(features, load_model()) #(+ learned model if theres one saved)
    return VideoResult(
        path=path,
        use_pos=use_pos,
        seconds=times[-1] - times[0] if len(times) > 1 else 0.0,
        frames=len(times),
        face_share=faces / len(times) if times else 0.0,
        reading_share=len(shown) / checked if checked else 0.0,
        bpm=float(np.median(shown)) if shown else None,
        bpm_spread=float(np.std(shown)) if shown else 0.0,
        times=times,
        bpms=bpms,
        pulse=pulse,
        pulse_start=colour_times[0] if colour_times else 0.0,
        calibrated_time=calibrated_time,  #None if never got past calibrating
        regions=regions,
        features=features,
        verdict=verdict,
    )


def pulse_times(result: VideoResult) -> np.ndarray:
    #time of every sample in result.pulse (evenly spaced at RESAMPLE_HZ)
    length = len(result.pulse) if result.pulse is not None else 0
    return result.pulse_start + np.arange(length) / RESAMPLE_HZ


def result_row(label: str, result: VideoResult) -> dict[str, object]:
    #one row of the batch table -> label, summary, then each regions median score
    row: dict[str, object] = {
        "label": label,
        "file": result.path.name,
        "method": "POS" if result.use_pos else "Green",
        "seconds": round(result.seconds, 2),
        "frames": result.frames,
        "face_share": round(result.face_share, 3),
        "reading_share": round(result.reading_share, 3),
        "bpm": round(result.bpm, 1) if result.bpm is not None else "",  #"" = empty cell
        "bpm_spread": round(result.bpm_spread, 2),
    }
    for name, _, _ in REGIONS:
        reading = result.regions.get(name)
        column = "snr_" + name.replace(" ", "_") # e.g. "snr_left_cheek"
        row[column] = round(reading.snr_db, 2) if reading is not None else ""
    for name, value in asdict(result.features).items(): #every feature by name
        row[name] = round(value, 4) if np.isfinite(value) else "" #"" = couldnt measure it
    passed = quality_problem(result.features) is None # got past the quality check?
    row["passed_check"] = "yes" if passed else "no" #--fit only learns from these
    row["features_version"] = FEATURES_VERSION  #--fit refuses rows made w/ older maths
    row["verdict"] = result.verdict.label  # real / fake / unsrue
    score = result.verdict.score
    row["score"] = round(score, 1) if score is not None else ""
    return row


def print_progress(name: str) -> Callable[[float], None]:
    #makes an on_progress function that rewrites 1 terminal line eg "me.mp4: 42%"
    def show(share: float) -> None:
        print(f"\r{name}: {share:.0%}", end="", flush=True)  #\r = back to start of line

    return show


def print_summary(result: VideoResult) -> None:
    # whole video summary in the terminal
    bpm = f"{result.bpm:.0f} BPM" if result.bpm is not None else "no BPM (never a clear pulse)"
    print(f"\r{result.path.name}: {bpm}")
    print(f"  {result.seconds:.1f} s, {result.frames} frames, face found {result.face_share:.0%}")
    print(f"  BPM shown {result.reading_share:.0%} of the time after calibrating, "
          f"spread ±{result.bpm_spread:.1f}")
    for name, reading in result.regions.items():
        print(f"  {name}: {reading.snr_db:+.1f} dB, {reading.bpm:.0f} BPM")
    print("  features:")
    for name, value in asdict(result.features).items():
        print(f"    {name}: {value:.3f}" if isinstance(value, float) else f"    {name}: {value}")
    verdict = result.verdict
    score = f" (score {verdict.score:.0f}/100)" if verdict.score is not None else ""
    print(f"  verdict: {verdict.headline}{score}, from the {verdict.method}")
    for reason in verdict.reasons:
        print(f"    - {reason}")


def print_comparison(checked: list[tuple[str, VideoResult]]) -> None:
    #ach features median for the real clips vs the fake clips
    #side by side so the differences are easy to see
    print(f"\n{'feature':24}{'real':>10}{'fake':>10}") #:24 / :>10 = pad to fixed widths
    for field in fields(Features):  # fields() -> every field of the dataclass in order
        middles = []  #real clips middle value then fake clips
        for label in LABELS:
            results = [result for clip, result in checked if clip == label]  #this labels
            values = [getattr(result.features, field.name) for result in results]
            finite = [value for value in values if np.isfinite(value)] #skip unmeasured
            middles.append(f"{np.median(finite):10.3f}" if finite else f"{'-':>10}")
        print(f"{field.name:24}{middles[0]}{middles[1]}")


def features_from_row(row: dict[str, str]) -> Features:
    #a results.csv row back into Features (empty cell = couldnt measure = nan)
    values = {}
    for field in fields(Features):
        text = row[field.name]
        values[field.name] = int(text) if field.type is int else float(text or "nan")  #"" -> nan
    return Features(**values) # ** = dict keys used as the argument names


def run_fit() -> None:
    #learn the logistic regression from data/results.csv (made by --batch)
    # only saved if it beats the hand-set checks on the same clips (tested farily)
    if not RESULTS_PATH.exists():
        print(f"No {RESULTS_PATH} yet: run  .venv/bin/python video.py --batch  first.")
        return
    with open(RESULTS_PATH, newline="") as file:
        rows = list(csv.DictReader(file))  #one dict per row (column name -> text)
    needed = {"label", "file", "passed_check", "features_version"}  # columns --fit needs
    needed |= {field.name for field in fields(Features)} #|= adds these to the set
    current = str(FEATURES_VERSION) #csv cells are text
    has_columns = bool(rows) and needed <= set(rows[0])  # <= on sets -> "all of these are in"
    if not has_columns or any(row["features_version"] != current for row in rows):
        print(f"{RESULTS_PATH} is empty or from older feature maths: run --batch again first.")
        return
    usable = [  #clips that passed the check + have the models inputs
        row
        for row in rows
        if row["passed_check"] == "yes" and all(row[name] for name in MODEL_FEATURES)
    ]
    clips = [features_from_row(row) for row in usable]
    is_real = np.array([row["label"] == "real" for row in usable], dtype=bool)
    pairs = [Path(row["file"]).stem for row in usable] #real/me1.mov + fake/me1.mp4 = "me1"
    paired = [name for name in set(pairs) if pairs.count(name) == 2]  # 1 real + 1 fake each
    alone = sorted(name for name in set(pairs) if pairs.count(name) == 1)  #no partner
    print(f"{len(paired)} real/fake pairs matched by file name")
    if alone:  #still used, js tested 1 by 1 not as a pair
        print(f"No partner (same name in the other folder) for: {', '.join(alone)}")
    real_clips, fake_clips = int(is_real.sum()), int((~is_real).sum()) # ~ = not
    if min(real_clips, fake_clips) < MIN_CLIPS_PER_LABEL:
        print(
            f"Not enough clips: {real_clips} real and {fake_clips} fake passed the quality check "
            f"with the neck in view (needs {MIN_CLIPS_PER_LABEL} of each). Keeping the checks."
        )
        return
    model = fit_model(clips, is_real, pairs)
    print(
        f"Learned model: {model.model_accuracy:.0%} right (each pair left out in turn), "
        f"hand-set checks: {model.rules_accuracy:.0%} right, on the same {len(usable)} clips"
    )
    if model.model_accuracy <= model.rules_accuracy:  #a tie keeps the checks (easier to explain)
        print("Not saved: the hand-set checks did at least as well.")
        return
    save_model(model)
    print(f"Saved {MODEL_PATH}: Video Check uses it from now on (delete the file to undo).")


def run_batch(use_pos: bool) -> None:
    #check every clip in data/real/ + data/fake/ and write a row for each
    rows = []
    checked: list[tuple[str, VideoResult]] = [] #each clips label + result
    for label in LABELS:
        folder = DATA_PATH / label
        videos = video_files(folder)
        if not videos:
            print(f"No videos in {folder} (skipped)")
        for path in videos:
            try:
                result = analyse_video(path, use_pos, print_progress(f"{label}/{path.name}"))
            except ValueError as error:  # one bad file shouldnt stop the rest
                print(f"\r{label}/{path.name}: {error}")
                continue
            print_summary(result)
            rows.append(result_row(label, result))
            checked.append((label, result))  #(label, result) pairs for print_comparison
    if not rows:
        print("Nothing to write: put clips in data/real/ and data/fake/ first.")
        return
    with open(RESULTS_PATH, "w", newline="") as file: #newline="" -> csv adds its own
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))  # column names = the keys
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {RESULTS_PATH}")
    print_comparison(checked)#real vs fake side by side


def main() -> None:
    parser = argparse.ArgumentParser(description="PulseCheck on video files")  #reads the opitons
    parser.add_argument("video", nargs="?", help="a video file to check") #? = optional
    parser.add_argument("--batch", action="store_true", help="check data/real/ and data/fake/")
    parser.add_argument("--green", action="store_true", help="use green only instead of POS")
    parser.add_argument("--fit", action="store_true", help="learn the model from results.csv")
    options = parser.parse_args()
    if options.fit:  #logistic regression
        run_fit()
    elif options.batch:
        run_batch(not options.green)
    elif options.video:
        path = Path(options.video)
        print_summary(analyse_video(path, not options.green, print_progress(path.name)))
    else:  #nothing asked -> show how to use it
        parser.print_help()


if __name__ == "__main__":  #only when run directly (python video.py) not when improted
    main()
