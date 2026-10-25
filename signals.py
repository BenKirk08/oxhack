# pulse signal maths -> the last few seconds of  skins mean colour. turn it into an
#                   evenly timed, filtered pulse wave and find the heart rate in BPM

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, detrend, periodogram, sosfiltfilt #called this file signals so dont get confused

BUFFER_SECONDS = 10.0 #how many seconds of colour samples are kept
MIN_SECONDS = 8.0 #show "Calibrating" until this many seconds  been collected
MAX_GAP_SECONDS = 1.0 # gap until face presumed lost and buffer starts again
RESAMPLE_HZ = 30.0 #evenly spaced samples -> cameras fps
LOW_HZ = 0.7 #slowest pulse kept 0.7 beats per second (hz) = 42 BPM
HIGH_HZ = 4.0 #fastest pulse kept 4 beats per second (hz)= 40 BPM
FILTER_ORDER = 3  # called Butterworth steepness: higher = sharper cut-off at the band edges (3 pretty medium )
FFT_LENGTH = 16384 # zero-padding so peak is found in steps of ab 0.1 BPM in FFT
#FFT -> Fast Fourier Transform -> finds storngtest wave with multiple waves (in wobbly waves)

POS_WINDOW_SECONDS = 1.6  #POS uses short overlapping windows -> 1.6s
MIN_SNR_DB = 0.0 #region is good if its pulse beats the noise (0DB)
AGREE_BPM = 6.0#a good region agrees if its own BPM is this close to the combined one

class ColourBuffer:
    #remembers the last BUFFER_SECONDS (10s rn) of (time, mean skin colour) samples.

    def __init__(self) -> None:
        self.times: list[float] = [] #when each sample was taken (in secs
        self.colours: list[np.ndarray] = [] #the mean B, G, R of the skin at that time
        #pre define what datatypes they are

    def add(self, time_s: float, colour_bgr: np.ndarray) -> None:
        #add a sample and forget the samples older than BUFFER_SECONDS
        if self.times and time_s - self.times[-1] > MAX_GAP_SECONDS:  #face was lost a while
            self.clear() #old might join bad onto new-> so start again (easier)
        self.times.append(time_s)
        self.colours.append(colour_bgr)
        while self.times[-1] - self.times[0] > BUFFER_SECONDS: #oldest sample too old -> loop
            self.times.pop(0) #pop(0) removes the first (oldest) item
            self.colours.pop(0)

    def clear(self) -> None:
        #forget every sample
        self.times.clear()
        self.colours.clear()

    def seconds(self) -> float:
        #how many secs samples cover from oldest to  newest
        if len(self.times) < 2: #0 or 1 samples cover no time yet
            return 0.0
        return self.times[-1] - self.times[0] #[-1] = last item, [0] = first item


def mean_colour(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    #return average B, G, R of the frames pixels inside the mask (None if it's empty)
    if cv2.countNonZero(mask) == 0: #no skin pixels to average
        return None
    blue, green, red, _ = cv2.mean(frame_bgr, mask=mask)  #4th value is unused
    return np.array([blue, green, red])


def resample(times: list[float], values: np.ndarray) -> np.ndarray:
    #turn unevenly timed samples into evenly timed ones -> RESAMPLE_HZ per second
    #use real timestamps -> camera that slows down or speeds up still works
    even_times = np.arange(times[0], times[-1], 1 / RESAMPLE_HZ) #start, stop, step i.e. [0.000, 0.033, 0.067,..]
    return np.interp(even_times, times, values)  #interpolate -> staright lines from samples at the times


def bandpass(values: np.ndarray) -> np.ndarray:
    #keep only changes at human pulse speed around 40-180bpm
    band = [LOW_HZ, HIGH_HZ]  #the frequencies in hz
    design = butter(FILTER_ORDER, band, "bandpass", fs=RESAMPLE_HZ, output="sos")  #the filter -> sos is whag butterworth take in suitable form
    return sosfiltfilt(design, values) # forwards then backward -> so no shift in time (sometimes little move error)


def green_pulse(buffer: ColourBuffer) -> np.ndarray:
    #the pulse wave from the green channel: resampled, detrended and band-pass filtered
    green = np.array(buffer.colours)[:, 1] #column 1 = green BGR
    green = resample(buffer.times, green)
    green = green / green.mean() - 1  #change compared w/ average e.g. 0.002 -> +0.2%
    green = detrend(green) #remove a slow straight-line drift (light slowly changing) -> caused errorsw when wasnt there claude suggested
    return -bandpass(green) #blood absorbs green so needa minus sign so beats point up


def pos_pulse(buffer: ColourBuffer) -> np.ndarray:
    #pos uses all colours to determune pulse
    #a light change, shadow or movement makes R, G and B brighter / darker by the same amount
    #but blood changes them by different amounts (mostly green) POS uses this to detect pulse
    colours = np.array(buffer.colours)  # one row per sample, columns B, G, R
    blue = resample(buffer.times, colours[:, 0])
    green = resample(buffer.times, colours[:, 1])
    red = resample(buffer.times, colours[:, 2])
    window = int(POS_WINDOW_SECONDS * RESAMPLE_HZ) #1.6 s -> 48 samples
    pulse = np.zeros(len(green))  #every windows result is added in here (overlap-add)
    count = np.zeros(len(green)) #how many windows covered each sample

    #sliding_window_view gives one row per window start
    red_windows = sliding_window_view(red, window)  # a view of red ->nothing copied
    green_windows = sliding_window_view(green, window)
    blue_windows = sliding_window_view(blue, window)

    red_part = red_windows / red_windows.mean(axis=1, keepdims=True) #1.0 = this windows average
    green_part = green_windows / green_windows.mean(axis=1, keepdims=True)
    blue_part = blue_windows / blue_windows.mean(axis=1, keepdims=True)

    s1 = green_part - blue_part #S1 = G - B -> if same amount change here will give 0 therfore would just be brightness
    s2 = green_part + blue_part - 2 * red_part  # S2 = G + B - 2R -> and here too
    s1_spread = s1.std(axis=1, keepdims=True)  # each window's spread (axis=1 = along each row)
    s2_spread = s2.std(axis=1, keepdims=True)

    balance = np.divide(s1_spread, s2_spread, out=np.zeros_like(s2_spread), where=s2_spread > 0)
    window_pulse = s1 + balance * s2 #scale s2 by ratio of spreads before adding almost cancels noise -> maths formula -> read on wiki
    window_pulse = window_pulse - window_pulse.mean(axis=1, keepdims=True)  # each window's mean
    windows = len(window_pulse)  # how many windows there are

    for offset in range(window):  # overlap-add -> 48 steps (one per position) not one per window
        pulse[offset: offset + windows] += window_pulse[:, offset] #subtract the mean theb add it in
        count[offset : offset + windows] += 1
    pulse = pulse / np.maximum(count, 1) #average the overlaps (ends have fewer windows)
    return -bandpass(pulse) #minus -> blood darkens skin so flip


def estimate_bpm(pulse: np.ndarray) -> float:
    #return strongest frequency in pulse band in bpm
    #periodogram is a FFT. "hann" softens ends of the wave -> so back of wave doesnt drop
    frequencies, power = periodogram(pulse, RESAMPLE_HZ, "hann", FFT_LENGTH) #FFT power
    in_band = (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ)  #true inside 0.7-4 Hz
    peak_frequency = frequencies[in_band][np.argmax(power[in_band])]  #argmax = index of max
    return float(peak_frequency * 60) #secs -> per min


def pulse_snr(pulse: np.ndarray) -> float:
    #how much the strongest beat speed stands out (dB)
    # compares power near the peak w rest of 0.7-4 hz band
    # 0 dB = as strong as everything else put together (bad) a clean pulse scores about 5-10 dB (good)
    frequencies, power = periodogram(pulse, RESAMPLE_HZ, "hann", FFT_LENGTH)  #as in estimate_bpm
    in_band = (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ)  #true inside 0.7-4 Hz
    frequencies, power = frequencies[in_band], power[in_band]  #keep only the pulse band
    if power.sum() <= 0: #no pulse -> flat
        return -100.0
    peak = frequencies[np.argmax(power)] # the strongest frequency (the BPM)
    nearest = (np.abs(frequencies - peak) <= 0.1) | (np.abs(frequencies - 2 * peak) <= 0.1) #6 BPM
    rest = max(power[~nearest].sum(), 1e-30)  # everything else (max -> never divide by 0)
    return float(10 * np.log10(power[nearest].sum() / rest))  #ratio in dB


@dataclass
class RegionReading:
    #one region's result -> @dataclass writes the constructor (like a Java record)
    pulse: np.ndarray # its pulse wave
    bpm: float # its own BPM
    snr_db: float # its quality score
#signal to noise ratio in decibels

class RegionReader:
    #works out each regions pulse wave (POS / green) + own BPM + quality score
    #POS slow so ONE region per frame (instead of all 6 in one)
    #copy all buffers at same time so can combine without any being further along

    def __init__(self) -> None:
        self.readings: dict[str, RegionReading] = {} #the last finished round use these
        self.todo: list[tuple[str, ColourBuffer]] = []  # (name, copied buffer) -> a to do list
        self.done: dict[str, RegionReading] = {} # this rounds readings so far

    def step(self, buffers: dict[str, ColourBuffer], use_pos: bool) -> None:
        #call once a frame -> does one region or finishes the round and  starts next one
        if self.todo: #regions left in this round -> do the next one
            name, buffer = self.todo.pop()  #takes last item off list
            pulse = pos_pulse(buffer) if use_pos else green_pulse(buffer)
            self.done[name] = RegionReading(pulse, estimate_bpm(pulse), pulse_snr(pulse))
            return
        self.readings = self.done  #round finished so these are now the readings to use
        self.done = {}
        for name, buffer in buffers.items():  #.items() gives (key, value) pairs
            if buffer.seconds() >= MIN_SECONDS: #skip regions still filling up
                copy = ColourBuffer()
                copy.times, copy.colours = list(buffer.times), list(buffer.colours)  # copies
                self.todo.append((name, copy)) #add to todo list


def combine_regions(readings: dict[str, RegionReading]) -> float | None:
    #the BPM of all the good regions together -> weighted by quality
    #returns None (shows "Hold still") if no region is good enough or if most good regions BPMs disagree
    good = [reading for reading in readings.values() if reading.snr_db >= MIN_SNR_DB]
    if not good: #no region has clear enough pulse
        return None
    length = min(len(reading.pulse) for reading in good) #waves can differ by a sample or two
    combined = np.zeros(length)
    for reading in good:
        wave = reading.pulse[-length:]  #the newest samples so all the waves line up at "now"
        weight = 10 ** (reading.snr_db / 10)  # dB -> times: 10 dB counts 10x as much as 0 dB
        combined += weight * wave / max(wave.std(), 1e-12)  #combined as will beat everywhree at same time!
    bpm = estimate_bpm(combined)
    agreeing = [reading for reading in good if abs(reading.bpm - bpm) <= AGREE_BPM]
    if len(agreeing) * 2 < len(good):  #fewer than half the good regions agree then not trusted
        return None
    return bpm
