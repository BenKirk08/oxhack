#tests for detector.py (steps 12-13) -> made up colour tracks instead of videos
#so they run in secs + each one knows the right answer (real, swap, noise, lamp etc)
#MOST important -> a real face (even weak) must NEVER be "Likely AI-generated"
# last test runs the whole pipeline on data/1.mov (skipped if its not there)
#run them w/ .venv/bin/pytest

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

import detector
from detector import (
    FACE,
    FEATURES_VERSION,
    MODEL_FEATURES,
    NECK,
    Features,
    Model,
    Verdict,
    add_sample,
    fit_model,
    judge,
    load_model,
    on_grid,
    ramp,
    save_model,
    video_features,
)
from patches import BACKGROUND_NAMES, REGIONS
from signals import ColourBuffer

FPS = 30.0 #made up cameras frame rate
SKIN = np.array([66.0, 102.0, 155.0])  # B, G, R of skin (like my clip)
NECK_SKIN = np.array([60.0, 94.0, 146.0])  #neck -> bit darker (chin shades it)
WALLS = [np.array([91.0, 170.0, 198.0]), np.array([110.0, 160.0, 180.0])]  #the 2 corners
SIGNATURE = np.array([0.53, 0.77, 0.33])  #how much a pulse changes B, G, R (green most)
TINT = np.array([0.9, 1.0, 1.15])  # warm light -> brighter also means redder
LAMP_TINT = np.array([0.6, 1.0, 1.3]) #warm lamp flicker (POS cancels it on the face)
GREEN_LAMP_TINT = np.array([0.7, 1.0, 0.6])  #greenish lamp flicker -> POS lets this 1 through
BALANCE_TINT = np.array([1.0, 0.0, -1.0])  # white balance hunting: blue up red down, no green
CLIP = Path(__file__).parent.parent / "data" / "1.mov"  #my real clip (weak pulse tho)


def wander(rng: np.random.Generator, length: int, red: bool = False) -> np.ndarray:
    #slow random wobble (0.1-3 Hz, size 1) -> head movement + lighting changes
    #red = most of it at the slow end like a real cameras drift
    # cumsum = running total
    band = butter(2, [0.1, 3.0], "bandpass", fs=FPS, output="sos")
    steps = rng.normal(size=length)
    wobble = sosfiltfilt(band, np.cumsum(steps) if red else steps)
    return wobble / wobble.std()


def make_tracks(
    pulse: float = 0.004,
    bpm: float = 72.0,
    inner_pulse: float | None = None,
    inner_bpm: float | None = None,
    outer_pulse: float | None = None,
    outer_bpm: float | None = None,
    neck_pulse: float | None = None,
    double_region: str | None = None,
    flicker: float = 0.0,
    flicker_bpm: float = 72.0,
    white_balance: float = 0.0,
    red_motion: float = 0.0,
    lamp_tint: np.ndarray = LAMP_TINT,
    hrv: float = 0.0,
    still_walls: bool = False,
    noise: float = 0.15,
    seconds: float = 20.0,
    seed: int = 0,
) -> dict[str, ColourBuffer]:
    #a made up videos colour tracks like video.py collects (patches, face, neck, bg)
    #pulse / bpm = the persons real heartbeat (0.004 = 0.4% colour change = clear)
    #inner_* (cheeks, nose) + outer_* (forehead, jaws) = a swap on part of the face
    #neck_pulse = necks pulse, always at the real bpm (None = no neck)
    #double_region = the patch that beats at 2x
    #flicker = lamp flickering flicker_bpm times a min (lamp_tint = its B, G, R mix)
    #white_balance = camera wb hunting at 72 a min
    #red_motion = slow drifting head movement (moves face + neck, not the wall)
    #hrv = how many BPM the heart rate drifts up and down
    #still_walls = background never changes, noise = camera grain in 0-255 levels
    rng = np.random.default_rng(seed) #seeded -> same "random" numbers every run
    times = np.arange(0, seconds, 1 / FPS)
    length = len(times)
    shared = 1 + 0.003 * wander(rng, length)[:, None] * TINT  # changes every patch at once
    flicker_wave = np.sin(2 * np.pi * flicker_bpm / 60 * times + 0.7)[:, None] #a column
    light = 1 + flicker * flicker_wave * lamp_tint #flickering lamp (1 = no flicker)
    balance_wave = np.sin(2 * np.pi * 72 / 60 * times + 1.9)[:, None]
    light = light * (1 + white_balance * balance_wave * BALANCE_TINT)  # + white balance
    motion = 1 + red_motion * wander(rng, length, red=True)[:, None] * TINT  #face + neck only
    drift = np.cumsum(rng.normal(0, 0.02, length)) / FPS  #heart rate wobbles a bit
    wobble = hrv * np.sin(2 * np.pi * 0.1 * times + rng.uniform(0, 6.3))  # BPM up + down

    def patch(base: np.ndarray, size: float, rate: float) -> np.ndarray:
        #one patchs colour over time -> its pulse, the lighting + camra grain
        phase = 2 * np.pi * np.cumsum((rate + wobble) / 60) / FPS  #beats so far as an angle
        beat = np.sin(phase + drift + rng.uniform(0, 0.2))[:, None]
        colour = base * (1 + size * SIGNATURE * beat) * shared * light * motion
        return colour + noise * rng.normal(size=(length, 3))

    tracks: dict[str, ColourBuffer] = {}
    patches = []
    for name, inner, _ in REGIONS:
        size_override = inner_pulse if inner else outer_pulse  #a swaps pulse here (or None)
        size = pulse if size_override is None else size_override # None -> the real pulse
        rate_override = inner_bpm if inner else outer_bpm
        rate = bpm if rate_override is None else rate_override
        if name == double_region:
            rate = 2 * bpm
        colour = patch(SKIN, size, rate)
        patches.append(colour)
        for time_s, sample in zip(times, colour, strict=True):
            add_sample(tracks, name, float(time_s), sample)
    face = np.mean(patches, axis=0)  #whole skin = patches togther
    for time_s, skin in zip(times, face, strict=True):
        add_sample(tracks, FACE, float(time_s), skin)
    for name, wall_colour in zip(BACKGROUND_NAMES, WALLS, strict=True): #no pulse here
        wall = wall_colour * shared * light + 0.001 * wall_colour * rng.normal(size=(length, 3))
        if still_walls:
            wall = np.tile(wall_colour, (length, 1))  #exact same colour every frame
        for time_s, sample in zip(times, wall, strict=True):
            add_sample(tracks, name, float(time_s), sample)
    if neck_pulse is not None: # neck in view
        neck = patch(NECK_SKIN, neck_pulse, bpm)
        for time_s, sample in zip(times, neck, strict=True):
            add_sample(tracks, NECK, float(time_s), sample)
    return tracks


def verdict_for(usable_share: float = 0.9, **options: Any) -> tuple[Features, Verdict]:
    #features + verdict for 1 made up video
    # **options go to make_tracks (Any = any type, they mix numbers + names)
    features = video_features(make_tracks(**options), usable_share)
    return features, judge(features)


@pytest.mark.parametrize("seed", [0, 1, 2])  #test runs once per seed
def test_real_face_with_its_neck_is_likely_real(seed: int) -> None:
    features, verdict = verdict_for(neck_pulse=0.004, seed=seed)  #clear neck pulse
    assert verdict.label == "real"
    assert verdict.score is not None and verdict.score >= 60
    assert "The neck's pulse matches the face's." in verdict.reasons
    assert abs(features.face_bpm - 72) < 2 # the made up heart rate


def test_real_face_without_a_neck_is_still_likely_real() -> None:
    _, verdict = verdict_for()  #clear steady pulse the whole face agrees on
    assert verdict.label == "real"


def test_swap_that_removes_the_pulse_is_caught_by_the_neck() -> None:
    #demo swap tools replace the WHOLE face (forehead + jaws too) but never the neck
    #"fake" only when the face has no pulse at all so a few borderlnie ones
    #come out "not enough signal" instead -> never "real"
    options = {"inner_pulse": 0.0, "outer_pulse": 0.0, "neck_pulse": 0.004}
    verdicts = [verdict_for(seed=seed, **options)[1] for seed in range(30)]
    labels = [verdict.label for verdict in verdicts]
    assert labels.count("real") == 0
    assert labels.count("fake") >= 20 #(24 of 30 rn)
    fakes = [verdict for verdict in verdicts if verdict.label == "fake"]
    assert all("neck" in verdict.reasons[0] for verdict in fakes)
    assert all(verdict.score is not None and verdict.score <= 40 for verdict in fakes)


def test_swap_that_removes_the_pulse_without_a_neck_is_not_enough_signal() -> None:
    #no untouched skin to compare with -> cant tell a swap from a bad video so dont guess
    _, verdict = verdict_for(inner_pulse=0.0, outer_pulse=0.0)
    assert verdict.label == "unsure"
    assert any("neck couldn't be measured" in reason for reason in verdict.reasons)


def test_swap_with_its_own_heart_rate_is_caught_by_the_neck() -> None:
    _, verdict = verdict_for(inner_bpm=96.0, outer_bpm=96.0, neck_pulse=0.004) #face 96
    assert verdict.label == "fake"
    assert "two heart rates" in verdict.reasons[0]


def test_faint_neck_mostly_stops_a_swap_being_called_real() -> None:
    #grainy video -> swapped face beats clearly at 96, neck only faintly at the real 72
    # the neck at -2 to -5 dB isnt enough for "fake" but usually enough to doubt "real"
    #KNOWN LIMIT: neck down in the noise -> a swap w/ its own clear pulse can pass
    options = {"inner_bpm": 96.0, "outer_bpm": 96.0, "neck_pulse": 0.004, "noise": 0.3}
    labels = [verdict_for(seed=seed, **options)[1].label for seed in range(30)]  #[1] = verdict
    assert labels.count("real") <= 2 # (1 of 30 rn)


def test_swap_with_a_different_inner_rate_is_caught_by_the_forehead() -> None:
    #a swap that keeps forehead + jaw -> cheeks and nose beat at another rate
    _, verdict = verdict_for(inner_bpm=96.0)
    assert verdict.label == "fake"
    assert "cheeks and nose" in verdict.reasons[0]


def test_swap_that_copies_the_pulse_fools_it() -> None:
    #KNOWN LIMIT (Seibold et al. 2025) -> some deepfakes copy the real persons pulse
    #this test documents it so a change in behaviour gets noticed
    _, verdict = verdict_for(inner_pulse=0.0024, outer_pulse=0.0024, neck_pulse=0.004)
    assert verdict.label == "real"


@pytest.mark.parametrize("noise", [0.15, 0.3])
def test_real_faces_with_uneven_or_weak_pulses_are_never_called_fake(noise: float) -> None:
    #the worst mistake -> cheeks pulsing under half the foreheds strength
    # eg make up, beard, side light. all near the "clear" line
    #25 videos each + not 1 may come out "fake"
    for pulse in (0.002, 0.003, 0.004):
        for seed in range(25):
            options = {"pulse": pulse, "inner_pulse": 0.45 * pulse, "seed": seed, "noise": noise}
            options["hrv"] = 3.0  #heart rate drifting ±3 BPM like a real one
            _, verdict = verdict_for(**options)
            assert verdict.label != "fake", options


def test_one_patch_on_double_the_rate_is_not_called_fake() -> None:
    _, verdict = verdict_for(double_region="nose", neck_pulse=0.004)
    assert verdict.label != "fake"


def test_noise_only_is_not_enough_signal() -> None:
    _, verdict = verdict_for(pulse=0.0, neck_pulse=0.0)  #like a dark / blurry video
    assert verdict.label == "unsure"
    assert verdict.score is None #not scored
    assert "too weak" in verdict.reasons[0]


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_lamp_flicker_without_a_pulse_is_not_enough_signal(seed: int) -> None:
    #no heartbeat at all but a warm lamp flickers at pulse speed
    #POS cancels it on the whole face but the walls green + colour mixes still see it
    # these seeds all fooled the 1st version (measured the wall w/ POS)
    features, verdict = verdict_for(pulse=0.0, flicker=0.004, noise=0.1, seed=seed)
    assert verdict.label == "unsure"
    assert "background" in verdict.reasons[0]
    assert abs(features.background_bpm - 72) < 3 #names the lamps rate


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_white_balance_hunting_without_a_pulse_is_not_enough_signal(seed: int) -> None:
    #blue + red swap back and forth 72 times a min, green doesnt change
    #green alone cant see it on the wall but the S1, S2 colour mixes can -> check fires
    _, verdict = verdict_for(pulse=0.0, neck_pulse=0.0, white_balance=0.003, seed=seed)
    assert verdict.label == "unsure"
    assert "background" in verdict.reasons[0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_swap_under_white_balance_hunting_is_never_called_real(seed: int) -> None:
    options = {"inner_pulse": 0.0, "outer_pulse": 0.0, "neck_pulse": 0.004, "seed": seed}
    _, verdict = verdict_for(white_balance=0.003, **options)
    assert verdict.label != "real"


@pytest.mark.parametrize("seed", range(6))
def test_slow_drifting_movement_without_a_pulse_is_never_judged(seed: int) -> None:
    #"red" noise -> a real cameras drift piles up at the slow end of the pulse band
    # there it can look like a steady 40-50 BPM beat every patch shares
    _, verdict = verdict_for(pulse=0.0, neck_pulse=0.0, red_motion=0.006, seed=seed)
    assert verdict.label == "unsure"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_only_the_background_check_stops_a_lamp_that_pos_lets_through(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    #greenish lamp at pulse speed changes skin like blood so POS keeps it as a pulse
    #w/ the check = not enough signal, w/o it = "likely real" -> the check stops it
    # (monkeypatch swaps a setting js for this test)
    options = {"pulse": 0.0, "flicker": 0.004, "lamp_tint": GREEN_LAMP_TINT, "seed": seed}
    _, verdict = verdict_for(**options)
    assert verdict.label == "unsure"
    assert "background" in verdict.reasons[0]
    monkeypatch.setattr(detector, "MAX_BACKGROUND_SHARE", 2.0)  #a share cant reach 2 -> off
    assert verdict_for(**options)[1].label == "real"


def test_still_background_is_not_called_flicker() -> None:
    # a background that never changes (still virtual background) cant flicker
    _, verdict = verdict_for(still_walls=True, neck_pulse=0.004)
    assert verdict.label == "real"


def test_lamp_at_the_necks_noise_rate_does_not_stop_a_real_face() -> None:
    #covered neck (no pulse) whose noise peaks at the lamps 100 BPM
    #only the faces rates count for the bg check, the neck gets set aside instead
    labels = [verdict_for(flicker=0.004, flicker_bpm=100.0, neck_pulse=0.0, seed=seed)[1].label
              for seed in range(10)]
    assert labels.count("real") >= 9


def test_too_few_patches_cannot_show_agreement() -> None:
    tracks = make_tracks()
    for name in ("right cheek", "nose", "right jaw"): #only 3 of 6 patchs seen
        del tracks[name]
    verdict = judge(video_features(tracks, 0.9))
    assert verdict.label == "unsure"
    assert ("At least 4 of the 6 face patches beat at the same rate", None) in verdict.checks


def test_lamp_flicker_at_another_rate_still_lets_a_real_face_through() -> None:
    _, verdict = verdict_for(flicker=0.004, flicker_bpm=100.0)  #lamp 100, heart 72
    assert verdict.label == "real"


def test_mostly_left_out_is_not_enough_signal() -> None:
    _, verdict = verdict_for(usable_share=0.3)  # 70% of frames were blinks / no face
    assert verdict.label == "unsure"
    assert "30%" in verdict.reasons[0]


def test_too_short_is_not_enough_signal() -> None:
    features, verdict = verdict_for(seconds=5.0) #under MIN_SECONDS (8s)
    assert features.windows == 0
    assert verdict.label == "unsure"
    assert "it needs at least 8 s" in verdict.reasons[0]


def test_no_face_at_all_is_not_enough_signal() -> None:
    verdict = judge(video_features({}, 0.0))  #no tracks -> face never found
    assert verdict.label == "unsure"
    assert verdict.reasons[0].startswith("No face could be measured")


@pytest.mark.parametrize(
    "options",
    [{"neck_pulse": 0.004}, {"inner_pulse": 0.0, "outer_pulse": 0.0, "neck_pulse": 0.004}, {}],
)
def test_every_verdict_is_filled_in_properly(options: dict[str, float]) -> None:
    for usable_share in (0.9, 0.3): #0.3 stops it at the quality check
        _, verdict = verdict_for(usable_share=usable_share, **options)
        assert verdict.caveat and isinstance(verdict.caveat, str)
        assert all(isinstance(text, str) for text, _ in verdict.checks)  # (text, passed)
        assert all(passed in (True, False, None) for _, passed in verdict.checks)
        assert 1 <= len(verdict.reasons) <= 3


def test_on_grid_needs_the_whole_window() -> None:
    times = np.arange(0, 12, 1 / FPS)
    colours = np.tile(SKIN, (len(times), 1))  #np.tile -> same colour on evry row
    grid = np.arange(1, 11, 1 / FPS)
    buffer = on_grid(times, colours, grid)
    assert buffer is not None and len(buffer.times) == len(grid) #moved onto the grid
    gap = (times < 4) | (times > 6)  #face lost for 2 s
    assert on_grid(times[gap], colours[gap], grid) is None
    late = times > 3 # starts 2s into the window
    assert on_grid(times[late], colours[late], grid) is None


def test_ramp() -> None:
    assert ramp(4.0, 1.0, 7.0) == pytest.approx(0.5)  #approx -> allows roudning errors
    assert ramp(-5.0, 1.0, 7.0) == 0.0 and ramp(9.0, 1.0, 7.0) == 1.0 #kept to 0-1


def made_up_clips(rng: np.random.Generator, count: int) -> tuple[list[Features], list[str]]:
    #features for count real clips + their count face swapped twins (neck in view)
    # real = face + neck beat alike, fake = face lost its pulse, neck didnt
    real = video_features(make_tracks(neck_pulse=0.004), 0.9)  #starting point to edit
    clips, groups = [], []
    for number in range(count):
        jitter = rng.normal(size=3)
        clips.append(replace(  #replace() -> copy w/ these fields changed
            real,
            face_snr_db=4 + jitter[0],
            neck_face_snr_gap=jitter[1],
            face_neck_bpm_diff=abs(jitter[2]),
        ))
        clips.append(replace(
            real,
            face_snr_db=-6 + jitter[0],
            neck_face_snr_gap=10 + jitter[1],
            face_neck_bpm_diff=30 + 5 * jitter[2],
        ))
        groups += [f"pair{number}", f"pair{number}"] #a real clip + its fake twin
    return clips, groups


def test_fit_model_learns_and_is_saved_and_used(tmp_path: Path) -> None:  #tmp_path = test folder
    clips, groups = made_up_clips(np.random.default_rng(0), 10)
    is_real = np.array([True, False] * 10)  #real, fake, real, fake...
    model = fit_model(clips, is_real, groups)
    assert model.model_accuracy >= 0.9
    assert model.real_clips == 10 and model.fake_clips == 10
    path = tmp_path / "model.json"
    save_model(model, path)
    loaded = load_model(path)
    assert loaded == model  #dataclasses compare field by field (like a java equals())
    verdict = judge(clips[0], loaded) # a real clip judged by the learned model now
    assert verdict.method.startswith("learned model")
    assert verdict.label == "real"


def test_learned_model_cannot_say_fake_without_the_neck() -> None:
    clips, groups = made_up_clips(np.random.default_rng(2), 8)
    model = fit_model(clips, np.array([True, False] * 8), groups)
    no_neck = replace(clips[1], neck_snr_db=np.nan, neck_face_snr_gap=np.nan)  #a fake twin
    verdict = judge(no_neck, model)
    assert verdict.method == "hand-set checks" #model needs the necks numbers
    assert verdict.label != "fake"


def fixed_model(bias: float) -> Model:
    #a "model" that ingores every feature + always gives the same answer
    #bias +10 = always ab 100/100 (real), -10 = always ab 0/100 (fake)
    return Model(
        version=FEATURES_VERSION,
        features=list(MODEL_FEATURES),
        means=[0.0, 0.0, 0.0],
        stds=[1.0, 1.0, 1.0],
        weights=[0.0, 0.0, 0.0],
        bias=bias,
        real_clips=8,
        fake_clips=8,
        model_accuracy=1.0,
        rules_accuracy=0.5,
    )


@pytest.mark.parametrize(
    "options",  #swaps the checks call "fake" -> pulse removed + face at its own rate
    [
        {"inner_pulse": 0.0, "outer_pulse": 0.0, "neck_pulse": 0.004},
        {"inner_bpm": 96.0, "outer_bpm": 96.0, "neck_pulse": 0.004},
    ],
)
def test_learned_model_cannot_overrule_a_fake(options: dict[str, Any]) -> None:
    features, verdict = verdict_for(**options)
    assert verdict.label == "fake"
    assert judge(features, fixed_model(10.0)).label == "fake"  #even a model saying "real"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_learned_model_never_calls_a_clear_real_face_fake(seed: int) -> None:
    features, verdict = verdict_for(neck_pulse=0.004, seed=seed)
    assert verdict.label == "real"
    learned = judge(features, fixed_model(-10.0)) #model that says "fake" to everything
    assert learned.label == "unsure"  # can doubt "real" but never turn it into "fake"
    assert learned.score is None


def test_learned_model_may_call_fake_with_a_clear_neck_and_a_weak_face() -> None:
    options = {"inner_pulse": 0.0012, "outer_pulse": 0.0012, "neck_pulse": 0.004} #weak
    features, verdict = verdict_for(**options)
    assert verdict.label == "unsure"  #checks want NO face pulse at all for "fake"
    learned = judge(features, fixed_model(-10.0))
    assert learned.label == "fake"
    assert "neck" in learned.reasons[1]


def test_load_model_ignores_missing_broken_and_old_files(tmp_path: Path) -> None:
    assert load_model(tmp_path / "missing.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert load_model(broken) is None
    clips, groups = made_up_clips(np.random.default_rng(1), 8)
    model = fit_model(clips, np.array([True, False] * 8), groups)
    old = tmp_path / "old.json"
    save_model(replace(model, features=MODEL_FEATURES[:-1]), old)  #different feature list
    assert load_model(old) is None
    assert json.loads(old.read_text())["features"] == MODEL_FEATURES[:-1]
    save_model(replace(model, version=FEATURES_VERSION - 1), old)  #from b4 a maths change
    assert load_model(old) is None
    save_model(replace(model, weights=model.weights[:-1]), old) #numbers that dont fit
    assert load_model(old) is None


@pytest.mark.skipif(not CLIP.exists(), reason="data/1.mov isn't here")
def test_bens_real_clip_is_never_called_fake() -> None:
    #the whole pipeline on a real video, its pulse is v weak so
    #"not enough signal" or "likely real" are both honest answers, "fake" would be wrong
    from video import analyse_video  #imported here as it loads mediapipe (slow)

    result = analyse_video(CLIP)
    assert result.verdict.label != "fake"
    assert result.features.windows > 0
