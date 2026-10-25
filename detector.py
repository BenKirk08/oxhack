import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import detrend, periodogram
from scipy.special import expit

from patches import BACKGROUND_NAMES, REGIONS
from signals import (
    AGREE_BPM,
    BUFFER_SECONDS,
    FFT_LENGTH,
    HIGH_HZ,
    LOW_HZ,
    MAX_GAP_SECONDS,
    MIN_SECONDS,
    RESAMPLE_HZ,
    ColourBuffer,
    bandpass,
    estimate_bpm,
    pos_pulse,
    pulse_snr,
)

FACE = "face"  # track name for the whole skins colour
NECK = "neck"  # necks track, the skin no face swap replaecs
INNER = [name for name, inner, _ in REGIONS if inner] #cheeks + nose
OUTER = [name for name, inner, _ in REGIONS if not inner] #forehead + jaws
WINDOW_SECONDS = BUFFER_SECONDS  #each window 10s long like the live tabs readings
STEP_SECONDS = 1.0  # new window every 1s (2s made results depend on where they fell)
BACKGROUND_SNR_DB = 0.0  #wall "beats w/ the face" from 0 dB at 1 of the faces rates
PEAK_HZ = 0.1  #"at a rate" = within 0.1 Hz = 6 BPM, same as signals.pulse_snr
MIN_USABLE_SHARE = 0.5  #at least half the frames measured, not blinks / no face
MAX_BACKGROUND_SHARE = 0.5 # bg beating w/ the face in half the windows = flikcer
CLEAR_DB = 1.0  # pulse is clear from +1 dB if the clip has 6+ windows = 15s+
CLEAR_DB_SHORT = 2.0 #and from +2 dB in shorter ones as noise gets higher there
LONG_WINDOWS = 6 # how many windows make a clip "long". real camera noise is red + shared
#in the sims noise passed +1 dB in ab 2% of 20 s clips but ab 6% of 10 s ones
ABSENT_DB = -4.0  # face at -4 dB or less = no pulse at all, pure noise sits there
STRONG_DB = 7.0  # score maxes out at this pulse strength
MOST_SURE_REAL = 90.0  #highest score a pulse can get -> its evidance not proof
LEAST_SURE_FAKE = 10.0  #and the lowest
STEADY_BPM = 5.0 # a clear pulses BPM moves less than this bewteen windows
MIN_AGREEING = 2 / 3  # 4+ of the 6 patches must beat at the faces rate
RATE_GAP_BPM = 10.0 #2 heart rates further apart than this cant be 1 persons
NECK_HINT_DB = -2.0  # neck this strong is faint but enough to doubt a face that disagrees,
NECK_HINT_STEADY_BPM = 2.0 #so is a weaker one whose BPM barely moves (noise wanders)
REAL_SCORE = 60.0 #scores from here up = "Likely real"
FAKE_SCORE = 40.0  # from here down = "Likely AI-generated"
MODEL_PATH = Path(__file__).parent / "data" / "detector_model.json" #the learned model
MODEL_FEATURES = ["face_snr_db", "neck_face_snr_gap", "face_neck_bpm_diff"]  # its 3 inputs,
# picked in advance -> few dozen clips so more inputs would js learn them by heart
MIN_CLIPS_PER_LABEL = 8 # learned model needs 8+ real and 8+ fake clips
FEATURES_VERSION = 3  # add 1 when a features maths changes so old saved models get ignored
REGULARISATION = 1.0  #keeps learned weights small so a few clips cant overfit
HEADLINES = {  # the verdict boxs big text for each label
    "real": "Likely real",
    "fake": "Likely AI-generated",
    "unsure": "Not enough signal to judge",
}
CAVEATS = {  # small print under each kind of verdict
    "real": (
        "A real pulse supports a real face, but some high-quality deepfakes copy the original "
        "person's pulse, so this is evidence, not proof."
    ),
    "fake": "Face swaps cause this. Heavy face-only beauty filters can too.",
    "unsure": "This isn't a verdict either way: the video doesn't give clear enough evidence.",
}
TIPS = "Try steady, bright light, a still face with the neck in view, and a less compressed file."


@dataclass
class Window:
    #what one 10 s window shows, nan = couldnt be measured in it
    face_bpm: float # whole skins BPM
    face_snr: float # and its pulse strength in dB
    regions_agree: float  # share of patches whose own BPM is close to the faces
    region_corr: float # how alike the patches waves are, avg over every pair
    region_bpm_spread: float # how far a typical patchs BPM is from the middle one
    inner_bpm: float  # inner face BPM, cheeks + nose together
    outer_bpm: float # outer face BPM (forehead + jaws)
    inner_snr: float # inner faces pulse strength in dB
    outer_snr: float # outer faces too
    inner_outer_corr: float  # how alike inner + outter waves are, -1 to 1
    neck_bpm: float #necks BPM (nan = no neck / set aside)
    neck_snr: float  # and how strong it is
    background_bpm: float  #face rate the background beats at most, nan = no bg
    background_snr: float  # how strong that is in dB


@dataclass
class Features:
    windows: int #10 s windows that could be measured, 0 = too short / face lost
    face_seconds: float #secs of face meausred, 1st to last sample
    usable_share: float  # share of frames measured (not blinks / talking / no face) 0-1
    face_bpm: float #faces heart rate in BPM
    face_snr_db: float  #pulse strength of the whole face (dB)
    bpm_stability: float # BPM consistency: how much it moves between windows
    regions_agree: float  #patch agreement: share of patches beating at the faces rate
    region_corr: float #patch agreement: how alike their waves are, -1 to 1
    region_bpm_spread: float  # patch agreement: how far apart their BPMs are
    inner_bpm: float  #inner vs outer: cheeks + nose BPM
    outer_bpm: float #and forehead + jaws BPM
    inner_snr_db: float #their pulse strengths in dB
    outer_snr_db: float
    inner_outer_corr: float  # how alike their waves are, -1 to 1
    neck_bpm: float # necks heart rate, nan = no usable neck
    neck_snr_db: float  #necks pulse strength (dB)
    neck_stability: float  #how much the necks BPM moves between windows
    neck_face_snr_gap: float  # neck - face pulse strength in dB -> big = face lost it
    face_neck_bpm_diff: float #BPM difference face vs neck
    background_snr_db: float #backgrounds "pulse" strength (dB, nan = no bg)
    background_share: float #share of windows where bg beats w/ the face
    background_bpm: float # bgs beat rate in those windows, nan = none


@dataclass
class Verdict:
    label: str #"real", "fake" or "unsure"
    headline: str  # the big text
    score: float | None #0-100 how real it looks, None = not scored
    reasons: list[str] # up to 3 short reasons in plain english
    method: str #what decided it (quality check / hand-set checks / learned model)
    caveat: str #small print on what this verdict can + cant prove
    checks: list[tuple[str, bool | None]] # each check + if it passed, None = n/a


@dataclass
class Model:
    # a learned logistic regression -> saved as json by save_model, read back by load_model
    version: int  # FEATURES_VERSION it was made w/
    features: list[str] # feature names, same order as the wieghts
    means: list[float] #each features average over the training clips
    stds: list[float]  #and spread, puts every feature on the same scale 1st
    weights: list[float] # how hard each feature pushes to real (+) or fake (-)
    bias: float  # push b4 any feature counts
    real_clips: int  # clips of each label it learned from
    fake_clips: int
    model_accuracy: float  #accuracy when tested fairly by leaving 1 pair out, 0-1
    rules_accuracy: float  # hand-set checks accuracy on the same clips


def add_sample(
    tracks: dict[str, ColourBuffer], name: str, time_s: float, colour: np.ndarray
) -> None:
    #add 1 colour sample to a whole video track, made the 1st time its needed
    # unlike ColourBuffer.add nothing gets dropped -> keeps the whole video
    track = tracks.setdefault(name, ColourBuffer()) #setdefault -> get it or add a new one
    track.times.append(time_s)
    track.colours.append(colour)


def on_grid(times: np.ndarray, colours: np.ndarray, grid: np.ndarray) -> ColourBuffer | None:
    #moves one tracks samples onto the windows even 30 Hz times (the grid)
    # so every wave lines up. None if the track doesnt cover the window ->
    # starts / stops over 1 s inside it or has a gap over 1 s
    near = (times >= grid[0] - MAX_GAP_SECONDS) & (times <= grid[-1] + MAX_GAP_SECONDS)
    near_times = times[near] #samples in the window + up to 1 s either side
    if len(near_times) < 2:  # under 2 samples there
        return None
    starts_late = near_times[0] > grid[0] + MAX_GAP_SECONDS  # over 1 s after the strat
    stops_early = near_times[-1] < grid[-1] - MAX_GAP_SECONDS # over 1 s b4 the end
    if starts_late or stops_early:
        return None
    if np.max(np.diff(near_times)) > MAX_GAP_SECONDS:  # diff = gaps between samples
        return None
    channels = [np.interp(grid, near_times, colours[near, channel]) for channel in range(3)]
    buffer = ColourBuffer() # interp above -> stright lines between the samples
    buffer.times = list(grid)
    buffer.colours = list(np.column_stack(channels))# one (B, G, R) row per grid time
    return buffer


def average_wave(waves: list[np.ndarray]) -> np.ndarray | None:
    #patches waves as one, each scaled to the same size 1st
    # so one noisy patch cant drown out the others. None if no usable wave
    scaled = [wave / wave.std() for wave in waves if wave.std() > 0]  # std 0 = flat  wave
    if not scaled:
        return None
    return np.mean(scaled, axis=0)  # axis=0 -> avergae at each moment


def bpm_of(wave: np.ndarray | None) -> float:
    # a waves BPM, nan if no wave
    return estimate_bpm(wave) if wave is not None else np.nan


def snr_of(wave: np.ndarray | None) -> float:
    #waves pulse strength in dB (nan if no wave)
    return pulse_snr(wave) if wave is not None else np.nan


def correlation(first: np.ndarray | None, second: np.ndarray | None) -> float:
    # how alike 2 waves are: 1 = move together, 0 = unrelated, -1 = opposite
    if first is None or second is None or first.std() == 0 or second.std() == 0:
        return np.nan  #cant compare a missing / flat wave
    return float(np.corrcoef(first, second)[0, 1]) # 2x2 table -> [0, 1] = first vs second


def mean_correlation(waves: list[np.ndarray]) -> float:
    #average correlation over every pair of waves, nan if under 2
    pairs = [correlation(first, second) for first, second in combinations(waves, 2)] # each pair
    finite = [pair for pair in pairs if np.isfinite(pair)] #drop pairs that couldnt compare
    return float(np.mean(finite)) if finite else np.nan


def spread(bpms: list[float]) -> float:
    #how far a typical patchs BPM is from the middle one
    # median distance so one odd patch cant blow it up, nan if under 2
    if len(bpms) < 2:
        return np.nan
    return float(np.median(np.abs(np.array(bpms) - np.median(bpms))))


def middle(values: list[float]) -> float:
    # median of the values that could be measured, nan if none
    finite = [value for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else np.nan


def stability(values: list[float]) -> float:
    # how much a BPM moves window to window -> std dev, nan if under 2
    finite = [value for value in values if np.isfinite(value)]
    return float(np.std(finite)) if len(finite) >= 2 else np.nan


def snrs_at(wave: np.ndarray, bpms: list[float]) -> list[float]:
    #like signals.pulse_snr but at BPMs we pick -> power within 0.1 Hz of each
    # + twice it vs all other power in the pulse band 0.7-4 Hz, in dB
    frequencies, power = periodogram(wave, RESAMPLE_HZ, "hann", FFT_LENGTH)  #same as pulse_snr
    in_band = (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ) #True for 0.7-4 Hz only
    frequencies, power = frequencies[in_band], power[in_band]
    if power.sum() <= 1e-30: #flat wave -> nothing beats at any rate
        return [-100.0] * len(bpms)  #what pulse_snr gives a flat wave too
    strengths = []
    for bpm in bpms:
        rate = bpm / 60  #beats per sec = Hz
        at_rate = np.abs(frequencies - rate) <= PEAK_HZ  # True near the rate
        near = at_rate | (np.abs(frequencies - 2 * rate) <= PEAK_HZ)# or near twice it
        ratio = max(power[near].sum(), 1e-30) / max(power[~near].sum(), 1e-30)  # never / 0
        strengths.append(float(10 * np.log10(ratio)))
    return strengths


def wall_waves(buffer: ColourBuffer) -> list[np.ndarray]:
    #a bg corners 3 kinds of flicker, band-passed like a pulse ->
    # green for brightness eg a lamp, POS's 2 colour mixes S1 & S2 for colour
    # eg white balance hunting. not POS itself as its made to cancel these
    # and would hide the flicker the faces POS can mistake for a pulse
    colours = np.array(buffer.colours)  # one row per sample (B, G, R) already on the grid
    means = colours.mean(axis=0) # each colours average over the window
    if np.any(means < 5) or np.any(means > 250): # black / blown out wall cant shwo it
        return []
    relative = colours / means  #each colour around 1
    if np.ptp(relative, axis=0).max() < 1e-6: #ptp = max - min -> a wall that never changes
        return []  # cant flicker, eg still virtual bg or a screen recroding
    blue, green, red = relative.T #.T = one row per colour
    mixes = [green, green - blue, green + blue - 2 * red]
    return [bandpass(detrend(mix)) for mix in mixes] #detrend -> remove a slow straight drift


def read_window(waves: dict[str, np.ndarray], walls: list[ColourBuffer]) -> Window:
    #the pulse nubmers for one window from each tracks wave + the bg corners (walls)
    # bg checked at the faces own rates. if it beats at the necks rate too
    # the neck gets set aside so a lamp can never pose as the neck pulse
    regions = {name: waves[name] for name, _, _ in REGIONS if name in waves} # patches seen
    inner = average_wave([regions[name] for name in INNER if name in regions])
    outer = average_wave([regions[name] for name in OUTER if name in regions])
    face_bpm = estimate_bpm(waves[FACE])
    region_bpms = [estimate_bpm(wave) for wave in regions.values()]
    agreeing = [bpm for bpm in region_bpms if abs(bpm - face_bpm) <= AGREE_BPM]  # at the faces rate
    enough = len(region_bpms) >= MIN_AGREEING * len(REGIONS)  # 4+ patches seen so can compare
    inner_bpm, outer_bpm = bpm_of(inner), bpm_of(outer)
    neck_bpm, neck_snr = bpm_of(waves.get(NECK)), snr_of(waves.get(NECK)) # .get -> None if no neck
    rates = [bpm for bpm in (face_bpm, inner_bpm, outer_bpm) if np.isfinite(bpm)]  # face rates
    wall_mixes = [wave for wall in walls for wave in wall_waves(wall)] # every corners 3 mixes
    background_bpm, background_snr = np.nan, np.nan  # stay nan if no corner useable
    for wave in wall_mixes: #does a corner flicker at a rate the face beats at?
        for bpm, strength in zip(rates, snrs_at(wave, rates), strict=True):
            if np.isnan(background_snr) or strength > background_snr:  #keep the  strongest
                background_bpm, background_snr = bpm, strength
    if np.isfinite(neck_bpm):  #corner flickering at the necks rate too?
        at_neck = [snrs_at(wave, [neck_bpm])[0] for wave in wall_mixes]
        if at_neck and max(at_neck) >= BACKGROUND_SNR_DB: # yes -> the necks "pulse" might be
            neck_bpm, neck_snr = np.nan, np.nan  #the lamp so it cant be the contorl here
    return Window(  #name=value -> args by name, clearer than 14 in a row
        face_bpm=face_bpm,
        face_snr=pulse_snr(waves[FACE]),
        regions_agree=len(agreeing) / len(region_bpms) if enough else np.nan,  # nan = too few
        region_corr=mean_correlation(list(regions.values())),
        region_bpm_spread=spread(region_bpms),
        inner_bpm=inner_bpm,
        outer_bpm=outer_bpm,
        inner_snr=snr_of(inner),
        outer_snr=snr_of(outer),
        inner_outer_corr=correlation(inner, outer),
        neck_bpm=neck_bpm,
        neck_snr=neck_snr,
        background_bpm=background_bpm,
        background_snr=background_snr,
    )


def window_readings(tracks: dict[str, ColourBuffer]) -> list[Window]:
    #cut the video into 10 s windows, a new one every 1 s, and measure each
    # 8-10 s of face gets 1 shorter window, under 8 s gets none
    face = tracks.get(FACE)
    if face is None or len(face.times) < 2:  # face never measurd
        return []
    span = face.times[-1] - face.times[0]  #how long the face got measured
    if span < MIN_SECONDS:
        return []
    length = min(WINDOW_SECONDS, span)  # each windows length
    arrays = { # every track as numpy arrays, made once so each window slices them fast
        name: (np.array(track.times), np.array(track.colours))
        for name, track in tracks.items()
        if len(track.times) >= 2
    }
    windows = []
    last_start = face.times[-1] - length + 1e-6  #tiny bit more so the last start counts
    for start in np.arange(face.times[0], last_start, STEP_SECONDS): #every windows start
        grid = np.arange(start, start + length, 1 / RESAMPLE_HZ) # windows 30 Hz times
        waves = {} #track name: its POS wave in this window
        walls = [] # bg corners that cover this window
        for name, (times, colours) in arrays.items():
            buffer = on_grid(times, colours, grid)
            if buffer is None: # this track doesnt cover the window
                continue
            if name in BACKGROUND_NAMES:
                walls.append(buffer)
            else:
                waves[name] = pos_pulse(buffer)
        if FACE in waves:  # the face itself has to cover it
            windows.append(read_window(waves, walls))
    return windows


def background_matches(window: Window) -> bool:
    #True if a bg corner clearly flickers at a rate the face beats at
    # whole, inner or outer face all checked as a flickering lamp can leak into
    # the seperate patches even when the whole faces average cancels it
    return window.background_snr >= BACKGROUND_SNR_DB # nan >= 0 is False -> no corner


def video_features(tracks: dict[str, ColourBuffer], usable_share: float) -> Features:
    # face, regions, neck + bg. usable_share = share of frames measured
    windows = window_readings(tracks)
    face = tracks.get(FACE)
    face_seconds = face.times[-1] - face.times[0] if face and len(face.times) >= 2 else 0.0
    checked = [window for window in windows if np.isfinite(window.background_snr)]# had bg
    same_peak = [window for window in checked if background_matches(window)]
    face_bpm = middle([window.face_bpm for window in windows])
    face_snr = middle([window.face_snr for window in windows])
    neck_bpm = middle([window.neck_bpm for window in windows])
    neck_snr = middle([window.neck_snr for window in windows])
    return Features(
        windows=len(windows),
        face_seconds=face_seconds,
        usable_share=usable_share,
        face_bpm=face_bpm,
        face_snr_db=face_snr,
        bpm_stability=stability([window.face_bpm for window in windows]),
        regions_agree=middle([window.regions_agree for window in windows]),
        region_corr=middle([window.region_corr for window in windows]),
        region_bpm_spread=middle([window.region_bpm_spread for window in windows]),
        inner_bpm=middle([window.inner_bpm for window in windows]),
        outer_bpm=middle([window.outer_bpm for window in windows]),
        inner_snr_db=middle([window.inner_snr for window in windows]),
        outer_snr_db=middle([window.outer_snr for window in windows]),
        inner_outer_corr=middle([window.inner_outer_corr for window in windows]),
        neck_bpm=neck_bpm,
        neck_snr_db=neck_snr,
        neck_stability=stability([window.neck_bpm for window in windows]),
        neck_face_snr_gap=neck_snr - face_snr, #nan if no neck
        face_neck_bpm_diff=abs(face_bpm - neck_bpm),
        background_snr_db=middle([window.background_snr for window in windows]),
        background_share=len(same_peak) / len(checked) if checked else np.nan,
        background_bpm=middle([window.background_bpm for window in same_peak]),
    )


def quality_problem(features: Features) -> str | None:
    #why the video cant be judged, None = it can
    if features.face_seconds == 0:
        return "No face could be measured: none was found, or every frame was a blink or talking."
    if features.face_seconds < MIN_SECONDS:
        return (
            f"Only {features.face_seconds:.1f} s of face could be measured: it needs at least "
            f"{MIN_SECONDS:.0f} s."
        )
    if features.windows == 0: #long enough but never 8 s w/o losing the face
        return (
            "The face was lost or left out (talking, blinking, turning away) too often: it "
            f"needs {MIN_SECONDS:.0f} s in a row."
        )
    if features.usable_share < MIN_USABLE_SHARE:
        return (
            f"Only {features.usable_share:.0%} of the frames could be measured (the rest were "
            "blinks, talking or no face)."
        )
    if features.background_share >= MAX_BACKGROUND_SHARE:  #nan = no bg so skipped
        return (
            f"The background flickers {features.background_bpm:.0f} times a minute, like the "
            "face, so the 'pulse' is probably the lighting, not a heartbeat."
        )
    return None


def ramp(value: float, low: float, high: float) -> float:
    # 0 at low, 1 at high, straight line in between, kept 0-1
    return float(np.clip((value - low) / (high - low), 0, 1))


def clear_db(features: Features) -> float:
    #how strong a clear pulse has to be, shorter clips need more
    return CLEAR_DB if features.windows >= LONG_WINDOWS else CLEAR_DB_SHORT


def is_clear(snr_db: float, bpm_stability: float, bar_db: float) -> bool:
    #a clear pulse -> at least bar_db strong + its BPM steady between windows
    # nan stability = only 1 window so cant be checked, counts as steady
    return snr_db >= bar_db and not bpm_stability > STEADY_BPM #nan > 5 is False


def neck_is_clear(features: Features) -> bool:
    # neck has a clear steady pulse
    return is_clear(features.neck_snr_db, features.neck_stability, clear_db(features))


def face_is_clear(features: Features) -> bool:
    # face has a clear steady pulse that most of its patches agree on
    clear = is_clear(features.face_snr_db, features.bpm_stability, clear_db(features))
    return clear and features.regions_agree >= MIN_AGREEING


def neck_shows_a_pulse(features: Features) -> bool:
    #True if the neck shows at least a faint pulse -> fairly strong, or weaker
    # but v steady -> noise wanders aroud, a pulse stays put. nan = False
    if features.neck_snr_db >= NECK_HINT_DB:
        return True
    return features.neck_snr_db > ABSENT_DB and features.neck_stability <= NECK_HINT_STEADY_BPM


def different_rates(first_bpm: float, second_bpm: float) -> bool:
    # True if 2 heart rates cant both be 1 persons -> too far apart
    # and not 1 double the other, as a patch can lock onto its pulses double
    if not (np.isfinite(first_bpm) and np.isfinite(second_bpm)):
        return False
    gaps = [first_bpm - second_bpm, first_bpm - 2 * second_bpm, 2 * first_bpm - second_bpm]
    return all(abs(gap) > RATE_GAP_BPM for gap in gaps)


def neck_disagrees(features: Features) -> bool:
    #True if even a faint neck pulse is at a rate the face cant share
    # so the faces own pulse cant be trusted as real
    return neck_shows_a_pulse(features) and different_rates(features.face_bpm, features.neck_bpm)


def positive_control(features: Features, weak_face_counts: bool) -> tuple[str, float] | None:
    #the evidence "Likely AI-generated" needs as (reason, strength in dB) or None
    # neck beating clearly while the face doesnt / beats at another rate,
    # or 2 clear rates in 1 face. hand-set checks need NO face pulse at all,
    # a learned model can use a face thats js weak
    bar = clear_db(features)  # how strong "clear" has to be here
    neck = features.neck_bpm  # shorter name so mesages fit on a line
    if weak_face_counts:
        face_missing = features.face_snr_db < bar  # not clear
    else:
        face_missing = features.face_snr_db <= ABSENT_DB #nothing at all
    if neck_is_clear(features) and face_missing:
        reason = f"The neck has a clear pulse ({neck:.0f} BPM), but the face doesn't."
        return reason, features.neck_snr_db
    two_clear = neck_is_clear(features) and face_is_clear(features) #both have to be clear
    if two_clear and different_rates(features.face_bpm, neck):
        reason = (
            f"The face beats at {features.face_bpm:.0f} BPM but the neck at {neck:.0f} BPM: "
            "one person can't have two heart rates."
        )
        return reason, features.neck_snr_db
    inner_clear = features.inner_snr_db >= bar
    outer_clear = features.outer_snr_db >= bar
    if inner_clear and outer_clear and different_rates(features.inner_bpm, features.outer_bpm):
        reason = (
            f"The cheeks and nose beat at {features.inner_bpm:.0f} BPM but the forehead and jaw "
            f"at {features.outer_bpm:.0f} BPM: one person can't have two heart rates."
        )
        return reason, min(features.inner_snr_db, features.outer_snr_db) #the weaker half
    return None


def rule_checks(features: Features) -> list[tuple[str, bool | None]]:
    #every check behind the verdict + if it passed, for the "How it decided" list
    # None = couldnt be checked or wasnt used, eg comparing rates needs clear pulses
    bar = clear_db(features)  # the bar for this clip
    measured = np.isfinite(features.face_snr_db) #False if no window got measured
    clear = features.face_snr_db >= bar
    steady = None if np.isnan(features.bpm_stability) else features.bpm_stability <= STEADY_BPM
    agree = None if np.isnan(features.regions_agree) else features.regions_agree >= MIN_AGREEING
    both_halves = features.inner_snr_db >= bar and features.outer_snr_db >= bar # both clear
    neck_seen = bool(np.isfinite(features.neck_snr_db))
    neck_usable = face_is_clear(features) and neck_shows_a_pulse(features)
    background_seen = bool(np.isfinite(features.background_share))
    same_inner_outer = not different_rates(features.inner_bpm, features.outer_bpm)
    same_face_neck = not different_rates(features.face_bpm, features.neck_bpm)
    return [
        (f"Face pulse clear (at least {bar:+.0f} dB)", clear if measured else None),
        (f"Face's heart rate steady (moves under {STEADY_BPM:.0f} BPM)", steady),
        ("At least 4 of the 6 face patches beat at the same rate", agree),
        ("Inner and outer face at the same rate", same_inner_outer if both_halves else None),
        ("Neck pulse clear", neck_is_clear(features) if neck_seen else None),
        ("Face and neck at the same rate", same_face_neck if neck_usable else None),
        ("Background not flickering with the face",
         not features.background_share >= MAX_BACKGROUND_SHARE if background_seen else None),
    ]


def fake_score(evidence_db: float, bar_db: float) -> float:
    # score for a fake -> 40 if evidence is js clear, down to 10 if strong
    strength = ramp(evidence_db, bar_db, STRONG_DB)  # 0 = js clear, 1 = strong
    return FAKE_SCORE - (FAKE_SCORE - LEAST_SURE_FAKE) * strength


def rule_verdict(features: Features) -> tuple[str, float | None, list[str]]:
    #the hand-set checks answer -> (label, score, reasons). score = how strong
    # the deciding evidence was: 60-90 real, 10-40 fake, None if it cant judge
    bar = clear_db(features)# clear bar for this clips length
    face_clear = face_is_clear(features)
    control = positive_control(features, weak_face_counts=False) # evidence of a swap?
    if control is not None:
        reason, evidence_db = control #unpack the pair
        return "fake", fake_score(evidence_db, bar), [reason]
    if face_clear and neck_disagrees(features): #even a faint neck at another rate
        doubt = (
            f"The neck seems to beat at {features.neck_bpm:.0f} BPM, not the face's "
            f"{features.face_bpm:.0f} BPM, but too faintly to be sure."
        )
        return "unsure", None, [doubt, TIPS]
    if face_clear:  # clear steady pulse across the face
        together = (
            f"The face beats together at {features.face_bpm:.0f} BPM "
            f"({features.regions_agree:.0%} of its patches agree)."
        )
        if np.isnan(features.bpm_stability): #only 1 window -> cant check steadyness
            steady = (
                f"A clear pulse ({features.face_snr_db:+.1f} dB; the clip is too short to check "
                "it stays steady)."
            )
        else:
            steady = (
                f"A clear, steady pulse ({features.face_snr_db:+.1f} dB, moving "
                f"±{features.bpm_stability:.1f} BPM)."
            )
        reasons = [together, steady]
        if neck_is_clear(features):  #cant disagree here, checked above
            reasons.append("The neck's pulse matches the face's.")
        strength = ramp(features.face_snr_db, bar, STRONG_DB) # 0 at the bar, 1 at STRONG_DB
        return "real", REAL_SCORE + (MOST_SURE_REAL - REAL_SCORE) * strength, reasons
    return "unsure", None, unsure_reasons(features)


def unsure_reasons(features: Features) -> list[str]:
    #why the checks cant decide -> 1st failed check + how to get a better video
    if not features.face_snr_db >= clear_db(features): #not >= catches nan too
        reasons = ["The face's pulse is too weak to tell real from fake."]
    elif features.bpm_stability > STEADY_BPM:
        reasons = [f"The face's heart rate keeps jumping (±{features.bpm_stability:.0f} BPM)."]
    elif np.isnan(features.regions_agree):
        reasons = ["Too few of the face's patches could be measured to compare them."]
    else:
        reasons = [f"Only {features.regions_agree:.0%} of the face's patches agree on its rate."]
    if np.isnan(features.neck_snr_db):
        reasons.append(
            "The neck couldn't be measured (out of view, covered or in shadow), so there's no "
            "untouched skin to compare with."
        )
    elif not neck_is_clear(features):
        reasons.append("The neck's pulse is too weak to compare the face with.")
    return [*reasons, TIPS]  # * = unpack the list into the new one


def model_verdict(model: Model, features: Features, checks_answer: tuple[str, float | None, list[str]]) -> tuple[str, float | None, list[str]] | None:
    #the learned models answer kept insdie what the checks allow
    # None = keep the checks answer, also if one of its inputs is missing
    # checks "fake" stands, "real" needs the model to agree (else unsure)
    # otherwise the model can only say "fake" w/ a positive control
    numbers = np.array([getattr(features, name) for name in model.features]) #by name
    if not np.all(np.isfinite(numbers)):
        return None
    scaled = (numbers - np.array(model.means)) / np.array(model.stds)  # same as in training
    score = 100 * float(expit(scaled @ np.array(model.weights) + model.bias)) #chance x 100
    said = f"The learned model gives it {score:.0f}/100 for real."
    label, _, reasons = checks_answer
    if label == "fake":
        return None  # the checks "fake" stands
    if label == "real":
        if score >= REAL_SCORE: # model agrees
            return "real", min(score, MOST_SURE_REAL), [said, *reasons[:2]]
        doubt = "The checks found a clear pulse, but the learned model doesn't think it's real."
        return "unsure", None, [said, doubt, TIPS]
    control = positive_control(features, weak_face_counts=True) # checks said "not enough"
    if score <= FAKE_SCORE and control is not None:
        return "fake", max(score, LEAST_SURE_FAKE), [said, control[0]] #[0] = the reason
    return None  # checks "not enough signal" stands


def judge(features: Features, model: Model | None = None) -> Verdict:
    #the verdict for 1 video. quality check 1st, then the hand-set
    # checks decide + a saved learned model can refine it (see model_verdict)
    problem = quality_problem(features)
    method = "hand-set checks"
    checks = rule_checks(features)
    if problem is not None: # cant be judgde at all, none of the checks used
        label, score, reasons = "unsure", None, [problem, TIPS]
        method = "quality check"
        checks = [(text, None) for text, _ in checks]  #_ = the result, not needed
    else:
        label, score, reasons = rule_verdict(features)
        learned = model_verdict(model, features, (label, score, reasons)) if model else None
        if model is not None and learned is not None: # learned model refined it
            label, score, reasons = learned
            method = f"learned model ({model.real_clips} real + {model.fake_clips} fake clips)"
    return Verdict(  #by name so no 2 fields swap places by mistake (happened b4!)
        label=label,
        headline=HEADLINES[label],
        score=score,
        reasons=reasons,
        method=method,
        caveat=CAVEATS[label],
        checks=checks, #type: ignore
    )


def train(
    values: np.ndarray, is_real: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    #logistic regression -> find weights so expit(weights . scaled + bias) is near 1
    # for real clips + near 0 for fake ones (values = the clips feature table)
    #returns means, stds, weights, bias
    means = values.mean(axis=0)  # axis=0 -> down each column = 1 feature
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0  #feature that never changes -> dont divid by 0
    scaled = (values - means) / stds
    signs = np.where(is_real, 1.0, -1.0) # +1 = real, -1 = fake

    def loss(params: np.ndarray) -> tuple[float, np.ndarray]:
        #how wrong these weights are -> log-loss + small penalaty for big weights
        # + its slope, which tells minimize which way to change each weight
        weights, bias = params[:-1], params[-1]  # last number = the bias
        margins = signs * (scaled @ weights + bias)  # @ = matrix x vector, > 0 = right side
        log_loss = np.logaddexp(0, -margins).sum()  # log(1 + e^-m) per clip, no overflow
        total = log_loss + REGULARISATION / 2 * weights @ weights  #plus the penalty
        slopes = -signs * expit(-margins) # how each clips loss changes w/ its score
        gradient = np.append(scaled.T @ slopes + REGULARISATION * weights, slopes.sum())
        return float(total), gradient

    start = np.zeros(values.shape[1] + 1)# every weight + bias  start at 0
    found = minimize(loss, start, jac=True, method="L-BFGS-B")# jac=True -> loss gives slopes
    return means, stds, found.x[:-1], float(found.x[-1])


def make_model(
    values: np.ndarray, is_real: np.ndarray, model_accuracy: float, rules_accuracy: float
) -> Model:
    #train on these clips + package it as a Model
    means, stds, weights, bias = train(values, is_real)
    return Model(
        version=FEATURES_VERSION,
        features=list(MODEL_FEATURES),
        means=means.tolist(), # tolist -> plain python numbers for json
        stds=stds.tolist(),
        weights=weights.tolist(),
        bias=bias,
        real_clips=int(is_real.sum()),
        fake_clips=int((~is_real).sum()),  # ~ = not, on a numpy True/False array
        model_accuracy=model_accuracy,
        rules_accuracy=rules_accuracy,
    )


def fit_model(clips: list[Features], is_real: np.ndarray, groups: list[str]) -> Model:
    #learn the logistic regression from labelled clips + test it fairly
    # groups pairs each real clip w/ its faec swapped copy, left out together
    # so it cant "recognise" a fake from its real twin (same person, light, room)
    #hand-set checks get tested on the same clips, both count "not enough signal" as wrong
    rows = [[getattr(clip, name) for name in MODEL_FEATURES] for clip in clips] #1 per clip
    values = np.array(rows)  #table -> 1 row per clip, 1 column per feature
    labels = np.where(is_real, "real", "fake")  # right answer for each clip
    model_right = 0
    for group in sorted(set(groups)):  #set -> each pairs name once
        left_out = np.array([name == group for name in groups]) # True for this pairs clips
        if left_out.all() or len(set(is_real[~left_out])) < 2: #needs both lables to learn
            continue
        fold = make_model(values[~left_out], is_real[~left_out], 0.0, 0.0)
        for index in np.flatnonzero(left_out): # positions of the left out clips
            model_right += int(judge(clips[index], fold).label == labels[index])
    pairs = zip(clips, labels, strict=True) # pairs of clip + right answer
    rules_right = sum(int(judge(clip).label == label) for clip, label in pairs)
    return make_model(values, is_real, model_right / len(clips), rules_right / len(clips))


def save_model(model: Model, path: Path | None = None) -> None:
    # save the learned model as json so the app uses it (MODEL_PATH by default)
    path = path or MODEL_PATH  # looked up now so a test can point it somewhere else
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(asdict(model), indent=2)) #asdict -> dataclass to dict


def load_model(path: Path | None = None) -> Model | None:
    #the saved learned model or None if theres no usable one ->
    # missing, broken, a dif version / feature list or numbers that dont fit
    path = path or MODEL_PATH # or -> use MODEL_PATH if no path given
    if not path.exists():
        return None
    try:
        model = Model(**json.loads(path.read_text())) #** = the dicts keys as names
    except (ValueError, TypeError, KeyError): #not json or not a models fields
        return None
    if model.version != FEATURES_VERSION or model.features != MODEL_FEATURES:  #out of date
        return None
    sizes = {len(model.means), len(model.stds), len(model.weights), len(MODEL_FEATURES)}
    numbers = [*model.means, *model.stds, *model.weights, model.bias]
    if len(sizes) != 1 or not np.all(np.isfinite(numbers)) or min(model.stds) <= 0:
        return None  # sizes only has 1 entry if all 4 lengths match
    return model
