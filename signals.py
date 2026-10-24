# pulse signal maths -> the last few seconds of  skins mean colour. turn it into an
#                   evenly timed, filtered pulse wave and find the heart rate in BPM

import cv2
import numpy as np
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
    #a light change, shadow or movement makes R, G and B brighter / darker by the same amount
    #but blood changes them by different amounts (mostly green) POS uses this to detect pulse
    colours = np.array(buffer.colours)  # one row per sample, columns B, G, R
    blue = resample(buffer.times, colours[:, 0])
    green = resample(buffer.times, colours[:, 1])
    red = resample(buffer.times, colours[:, 2])
    window = int(POS_WINDOW_SECONDS * RESAMPLE_HZ) #1.6 s -> 48 samples
    pulse = np.zeros(len(green))  #every window's result is added in here (overlap-add)
    count = np.zeros(len(green)) #how many windows covered each sample
    for start in range(len(green) - window + 1): #slide the window along, 1 sample at a time
        end = start + window
        red_part = red[start:end] / red[start:end].mean() #1.0 = this window's average
        green_part = green[start:end] / green[start:end].mean()
        blue_part = blue[start:end] / blue[start:end].mean()
        s1 = green_part - blue_part #S1 = G - B -> if same amount change here will give 0
        s2 = green_part + blue_part - 2 * red_part  # S2 = G + B - 2R -> and here too
        balance = s1.std() / s2.std() if s2.std() > 0 else 0.0
        window_pulse = s1 + balance * s2 #h: mixed so leftover shine (glare) cancels
        pulse[start:end] += window_pulse - window_pulse.mean()  #subtract the mean, add it in
        count[start:end] += 1
    pulse = pulse / np.maximum(count, 1)  # average the overlaps (the ends have fewer windows)
    return -bandpass(pulse)  # minus: blood darkens skin, so flip it so beats point up


def estimate_bpm(pulse: np.ndarray) -> float:
    #return strongest frequency in pulse band in bpm
    #periodogram is a FFT. "hann" softens ends of the wave -> so back of wave doesnt drop
    frequencies, power = periodogram(pulse, RESAMPLE_HZ, "hann", FFT_LENGTH) #FFT power
    in_band = (frequencies >= LOW_HZ) & (frequencies <= HIGH_HZ)  #true inside 0.7-4 Hz
    peak_frequency = frequencies[in_band][np.argmax(power[in_band])]  #argmax = index of max
    return float(peak_frequency * 60) #secs -> per min
