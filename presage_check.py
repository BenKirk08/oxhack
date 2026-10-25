import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from video import DATA_PATH, LABELS, VideoResult, analyse_video, print_progress, video_files

ROOT = Path(__file__).parent
MEASURE_SCRIPT = ROOT / "presage" / "measure.mjs"
ENV_PATH = ROOT / ".env"  #holds SMARTSPECTRA_API_KEY=...
CSV_PATH = DATA_PATH / "presage.csv"  # where batch mode writes its table
CLOSE_BPM = 5.0  #within 5 BPM = a match (same as the plans smartwatch check)
TIMEOUT_SECONDS = 900  # a long 4K clip can take presage a while

@dataclass
class PresageReading:
    #what presage measured in 1 video
    times: np.ndarray  #when each pulse reading is from (s into the video)
    bpms: np.ndarray # each readings BPM (presage averages the last 12 s for each)
    confidences: np.ndarray  #0-100
    stable: np.ndarray  #True = confidence 40+ -> presage says its right to ±3 BPM
    hints: dict[str, int]  # presages warnings eg face too far -> how many frames


@dataclass
class Comparison:
    #PulseCheck vs presage on 1 video
    presage_bpm: float  #median of presages stable readings (nan = none were stable)
    pulsecheck_bpm: float # PulseCheck's median BPM (nan = never showed one)
    stable_readings: int  #how many of presages readings were stable
    readings: int
    match_share: float  # share of stable readings where PulseCheck was within 5 BPM at that moment
    average_gap: float  #avg BPM gap at those moments


def run_presage(path: Path) -> PresageReading:
    #run presage/measure.mjs on the video w/ node + read the json back
    node = shutil.which("node")  #full path to node (None if not installed)
    if node is None:
        raise RuntimeError("Node.js isn't installed. Run: brew install node")
    if not (MEASURE_SCRIPT.parent / "node_modules").exists():
        raise RuntimeError("Presage's SDK isn't installed. Run: cd presage && npm install")
    if not ENV_PATH.exists():
        raise RuntimeError("No .env file. Make one with the line SMARTSPECTRA_API_KEY=your key")
    command = [node, f"--env-file={ENV_PATH}", str(MEASURE_SCRIPT), str(path.resolve())]
    try:
        finished = subprocess.run(  #runs it + waits, like javas ProcessBuilder
            command, check=False, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
        )  #check=False -> no crash on exit code 1, its json still says what went wrong
        output = json.loads(finished.stdout) #the 1 line of json measure.mjs prints
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Presage took over {TIMEOUT_SECONDS} s, gave up") from None
    except json.JSONDecodeError:  #crashed b4 printing -> show the end of its error
        raise RuntimeError(f"Presage failed: {finished.stderr.strip()[-500:]}") from None # type: ignore
    if output["errors"]:  #eg wrong key or out of credits
        raise RuntimeError("Presage: " + "; ".join(error["message"] for error in output["errors"]))
    pulse = np.array(output["pulse"], dtype=float).reshape(-1, 4)  #-1 = however many rows
    return PresageReading(pulse[:, 0], pulse[:, 1], pulse[:, 2], pulse[:, 3] > 0, output["hints"])


def compare(presage: PresageReading, result: VideoResult) -> Comparison:
    # lines up presages stable readings w/ what PulseCheck showed at the same moments
    times, bpms = presage.times[presage.stable], presage.bpms[presage.stable] #stable ones only
    frame_times, frame_bpms = np.array(result.times), np.array(result.bpms)
    shown = np.full(len(times), np.nan)  #PulseCheck's BPM at each reading (nan = none shown)
    if len(frame_times):
        index = np.searchsorted(frame_times, times, side="right") - 1  # last frame at or b4 it
        shown = np.where(index >= 0, frame_bpms[np.clip(index, 0, None)], np.nan)
    gaps = np.abs(shown - bpms)
    gaps = gaps[np.isfinite(gaps)]  #only moments where both had a number
    return Comparison(
        presage_bpm=float(np.median(bpms)) if len(bpms) else np.nan,
        pulsecheck_bpm=result.bpm if result.bpm is not None else np.nan,
        stable_readings=len(bpms),
        readings=len(presage.bpms),
        match_share=float(np.mean(gaps <= CLOSE_BPM)) if len(gaps) else np.nan,
        average_gap=float(np.mean(gaps)) if len(gaps) else np.nan,
    )


def presage_bpm(presage: PresageReading) -> tuple[float | None, bool]:
    #presages own BPM for the whole video (for the web app) + if its sure
    #median of its stable readings, or of all of them if none were stable (not sure)
    if presage.stable.any():  #any() -> True if at least 1 is True
        return float(np.median(presage.bpms[presage.stable])), True
    if len(presage.bpms):  #readings but none stable
        return float(np.median(presage.bpms)), False
    return None, False  # no readings at all


def print_report(name: str, presage: PresageReading, comparison: Comparison) -> None:
    #1 videos result in the terminal
    print(f"\n{name}")  #\n ends the progress line 1st
    if comparison.readings == 0:
        print("  Presage: no reading (it needs ab 12 s+ of a still, well lit face)")
    elif comparison.stable_readings == 0:
        print(f"  Presage: {np.median(presage.bpms):.0f} BPM but none of its "
              f"{comparison.readings} readings were stable (too unsure to compare)")
    else:
        print(f"  Presage:    {comparison.presage_bpm:.0f} BPM "
              f"({comparison.stable_readings} of {comparison.readings} readings stable)")
    pulsecheck = comparison.pulsecheck_bpm
    shown = f"{pulsecheck:.0f} BPM" if np.isfinite(pulsecheck) else "no BPM"
    print(f"  PulseCheck: {shown}")
    gap = abs(pulsecheck - comparison.presage_bpm)
    if np.isfinite(gap):  #nan if either one had no BPM
        print(f"  -> {gap:.0f} BPM apart: {'match' if gap <= CLOSE_BPM else 'NO match'} "
              f"(within {CLOSE_BPM:.0f} counts)")
    if np.isfinite(comparison.match_share):
        print(f"  same moments: within {CLOSE_BPM:.0f} BPM {comparison.match_share:.0%} of the "
              f"time, {comparison.average_gap:.1f} BPM apart on avg")
    for hint, frames in presage.hints.items():  #.items() -> (key, value) pairs
        print(f"  Presage warning: {hint} ({frames} frames)")


def check(path: Path, name: str) -> Comparison:
    #presage 1st (fails fast if the key is wrong) then PulseCheck on the same video
    print(f"{name}: Presage...", end="", flush=True)
    presage = run_presage(path)
    result = analyse_video(path, on_progress=print_progress(f"{name}: PulseCheck"))
    comparison = compare(presage, result)
    print_report(name, presage, comparison)
    return comparison


def run_batch() -> None:
    #every clip in data/real/ + data/fake/ -> a row each in data/presage.csv
    rows = []
    for label in LABELS:
        for path in video_files(DATA_PATH / label):
            try:
                comparison = check(path, f"{label}/{path.name}")
            except (RuntimeError, ValueError) as error:  # one bad clip shouldnt stop the rest
                print(f"\r{label}/{path.name}: {error}")
                continue
            row: dict[str, object] = {"label": label, "file": path.name}
            for name, value in asdict(comparison).items():  #asdict -> the fields as a dict
                missing = isinstance(value, float) and not np.isfinite(value) #nan -> empty cell
                row[name] = "" if missing else round(value, 3)  # round leaves whole numbers be
            rows.append(row)
    if not rows:
        print("Nothing to check: put clips in data/real/ and data/fake/ first.")
        return
    with open(CSV_PATH, "w", newline="") as file:  #newline="" -> csv adds its own
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {CSV_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PulseCheck vs Presage on video files")
    parser.add_argument("video", nargs="?", help="a video file to check")  #? = optional
    parser.add_argument("--batch", action="store_true", help="check data/real/ and data/fake/")
    options = parser.parse_args()
    try:
        if options.batch:
            run_batch()
        elif options.video:
            check(Path(options.video), Path(options.video).name)
        else:
            parser.print_help()  #nothing asked -> show how to use it
    except RuntimeError as error:  #setup problems eg no node / no key
        print(f"\n{error}")
        raise SystemExit(1) from None  #exit code 1 = failed

if __name__ == "__main__":
    main()
