"""Trailing-window SSVEP detector shared by Cyton acquisition and file replay.

Inputs are samples and transport coordinates only. Indices are zero-based and
``end_sample`` is inclusive. Every window and its debounce history are causal
at emission time (the in-window filter only sees already received samples).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

from neurostation_contract import CONTINUOUS_SSVEP_METHOD as METHOD

FS = 250
FREQUENCIES = (10, 12, 15, 20)
WINDOW, HOP = 500, 50
MAX_CLOCK_STEP_S = 0.25  # USB bursts shorter than this alone do not establish sample loss.
# For flat Gaussian noise, one FFT bin divided by the mean of six independent
# neighboring bins has an F(2,12) reference. Adjust for 8 channels x 4 targets.
# This fixed engineering reference is not a false-alarm guarantee on EEG.
SIDE_BINS = np.array([-3, -2, -1, 1, 2, 3])
TARGET_BINS = np.array(FREQUENCIES) * WINDOW // FS
SPECTRAL_RATIO = len(SIDE_BINS) * ((.001 / (8 * len(FREQUENCIES))) ** (-1 / len(SIDE_BINS)) - 1)

# Fixed before evaluation. The long path integrates phase-coherent energy per
# channel. Its floor is a local noise estimate, not a clinical confidence.
LONG_WINDOW = 4 * FS
LONG_RATIO = 10.0
RECENT_GAIN = 0.65
RECENT_SECOND_GAIN = 0.55
NOISE_RADIUS = 20
STRONG_SCORE = 0.35
# Use both independent halves to identify a line already present at startup.
START_RATIO = 10.0
START_PHASE_COS = 0.85
START_MIN_BALANCE = 0.65


def evidence_settings():
    return dict(long_window=LONG_WINDOW, long_ratio=LONG_RATIO, recent_gain=RECENT_GAIN,
                recent_second_gain=RECENT_SECOND_GAIN, noise_radius_bins=NOISE_RADIUS,
                strong_score=STRONG_SCORE, startup_half_ratio=START_RATIO, startup_phase_cos=START_PHASE_COS,
                startup_min_balance=START_MIN_BALANCE,
                background_policy='remove_phase_aligned_startup_component')


@dataclass(frozen=True)
class Gate:
    min_score: float = 0.35
    min_margin: float = 0.04
    confirmations: int = 2

    def __post_init__(self):
        if not (0 <= self.min_score <= 1 and 0 <= self.min_margin <= 1
                and type(self.confirmations) is int and self.confirmations >= 1):
            raise ValueError('Invalid SSVEP gate')


@dataclass(frozen=True)
class Window:
    end_sample: int
    quality: str
    scores: tuple[float, ...] = ()
    spectral_support: tuple[bool, ...] = ()


@dataclass(frozen=True)
class Decision:
    window: Window
    hz: int | None
    state: str


class DecisionGate:
    def __init__(self, config: Gate):
        self.config = config
        self.candidate = None
        self.streak = 0

    def apply(self, w: Window) -> Decision:
        hz = None
        if w.scores:
            first, second = sorted(range(len(w.scores)), key=w.scores.__getitem__, reverse=True)[:2]
            if (w.spectral_support and w.spectral_support[first]
                    and w.scores[first] >= self.config.min_score
                    and w.scores[first] - w.scores[second] >= self.config.min_margin):
                hz = FREQUENCIES[first]
        self.streak = self.streak + 1 if hz is not None and hz == self.candidate else int(hz is not None)
        self.candidate = hz
        if hz is not None and self.streak >= self.config.confirmations:
            return Decision(w, hz, 'target')
        return Decision(w, None, 'no_target' if w.quality == 'ok' else 'unavailable')


def sample_quality(index, packet, clock, eeg, previous=None):
    """Flag a break on the first sample after it. Never synthesize EEG."""
    issues = np.full(len(index), '', dtype='U28')
    bad = ~np.isfinite(eeg).all(axis=1)
    issues[bad] = 'nonfinite_eeg'
    if previous is None:
        previous = (index[0] - 1, packet[0] - 1, clock[0] - 1 / FS)
    ts_step = np.diff(clock, prepend=previous[2])
    issues[(ts_step <= 0) | (ts_step > MAX_CLOCK_STEP_S) | ~np.isfinite(ts_step)] = 'clock_discontinuity'
    with np.errstate(invalid='ignore'):
        bad_packet = ((np.diff(packet, prepend=previous[1]) % 256 != 1) | (packet != np.floor(packet))
                      | (packet < 0) | (packet > 255) | ~np.isfinite(packet))
    issues[bad_packet] = 'packet_discontinuity'
    issues[np.diff(index, prepend=previous[0]) != 1] = 'missing_sample_index'
    return issues


class ContinuousDetector:
    def __init__(self, config: Gate = Gate(), *, gpu_graph=None):
        self.gpu_graph = gpu_graph
        self.gate = DecisionGate(config)
        self.sos = butter(4, [6, 35], btype='bandpass', fs=FS, output='sos')
        t = np.arange(WINDOW) / FS
        self.references = []
        self.phase_fits = []
        for f in FREQUENCIES:
            phase = 2 * np.pi * f * t
            carrier = np.column_stack((np.sin(phase), np.cos(phase)))
            fit_part = carrier[:-HOP] - carrier[:-HOP].mean(axis=0)
            test_part = carrier[-HOP:] - carrier[-HOP:].mean(axis=0)
            self.phase_fits.append((np.linalg.pinv(fit_part), test_part))
            self.references.append(np.linalg.qr(carrier)[0])
        self.tail = np.empty((0, 8))
        self.previous = None
        self.valid_since = 0
        self.break_reason = 'warmup'

        self.background = None
        self.background_start = 0
        self.long_since = 0
        self.last_strong = np.full(len(FREQUENCIES), -LONG_WINDOW)
        self.freq = np.asarray(FREQUENCIES)
        # Absolute phase recurs every FS samples for these integer frequencies.
        self.carriers = np.exp(2j * np.pi * np.arange(FS)[:, None] * self.freq / FS)
        # Exclude other target bins from the local noise reference.
        bins = self.freq * LONG_WINDOW // FS
        self.noise_bins = [np.array([b for b in range(f - NOISE_RADIUS, f + NOISE_RADIUS + 1) if b not in bins]) for f in bins]

    def _startup_lines(self, window):
        halves = window.reshape(2, WINDOW // 2, 8)
        spec = np.fft.rfft(halves - halves.mean(axis=1, keepdims=True), axis=1)
        lines = spec[:, self.freq, :]
        # One-second halves span integer cycles. Compare against non-target bins.
        noise = np.mean(np.abs(spec[:, [i for i in range(5, 31) if i not in FREQUENCIES]]) ** 2, axis=1)
        mag = np.abs(lines)
        keep = np.all(mag ** 2 > START_RATIO * noise[:, None, :], axis=0)
        keep &= (lines[0].conj() * lines[1]).real > START_PHASE_COS * mag.prod(axis=0)
        keep &= mag.min(axis=0) >= START_MIN_BALANCE * mag.max(axis=0)
        return np.where(keep, lines.mean(axis=0) * (2 / FS), 0j)

    def _remove_startup(self, samples, end):
        if self.background is None:
            self.background = self._startup_lines(samples[-WINDOW:])
            self.background_start = end - WINDOW + 1
        if not self.background.any():
            return samples
        phase = (np.arange(len(samples)) + end + 1 - len(samples) - self.background_start) % FS
        carrier = self.carriers[phase]
        # Remove only the component still present in the same direction. Never
        # inject the negative of a disappeared line. Use an integer-cycle span.
        n = len(samples) // FS * FS
        observed = carrier[-n:].conj().T @ samples[-n:] * (2 / n)
        power = np.abs(self.background) ** 2
        scale = np.divide((observed * self.background.conj()).real, power,
                          out=np.zeros_like(power), where=power > 0).clip(0, 1)
        return samples - (carrier @ (self.background * scale)).real

    def _weak_evidence(self, samples):
        # Fit on the whole trailing window, require independent sign agreement
        # on its newest hop with a prediction fitted exclusively before it.
        data = detrend(samples, axis=0)
        bins = self.freq * LONG_WINDOW // FS
        if self.gpu_graph is None:
            spec = np.fft.rfft(data, axis=0)
            power = np.abs(spec) ** 2
            lines = spec[bins]
            floor = np.stack([power[side].mean(axis=0) for side in self.noise_bins])
            recent_coeff = np.fft.rfft(data[-FS:], axis=0)[self.freq] / FS
        else:
            lines, floor, recent_coeff = self.gpu_graph.long_evidence(data, bins, self.noise_bins)
        ratio = np.divide(np.abs(lines) ** 2, floor, out=np.zeros_like(floor), where=floor > 0)
        tail = data[-HOP:] - data[-HOP:].mean(axis=0)
        carrier = self.carriers[np.arange(-HOP, 0) % FS]
        # The centered newest hop can have fractional cycles. Its uncentered
        # contribution is removed from the DFT before extrapolating the tone.
        past = 2 * (lines - carrier.conj().T @ data[-HOP:]) / (LONG_WINDOW - HOP)
        predicted = (carrier[:, :, None] * past[None, :, :]).real
        predicted -= predicted.mean(axis=0)
        energy = np.sum(predicted ** 2, axis=0)
        projection = np.sum(predicted * tail[:, None, :], axis=0)
        # Template estimates use just WINDOW samples. Residual coherent energy
        # on those specific channels must exceed the extra estimation noise.
        threshold = LONG_RATIO * (1 + (LONG_WINDOW / WINDOW) * (self.background != 0))
        # Independent past/recent one-second phase agreement limits detections
        # driven by a line that ended earlier in the integration window.
        past_coeff = (lines - FS * recent_coeff) / (LONG_WINDOW - FS)
        agrees = (recent_coeff * past_coeff.conj()).real > RECENT_SECOND_GAIN * np.abs(past_coeff) ** 2
        supported = (ratio >= threshold) & agrees & (projection > RECENT_GAIN * energy)
        best = np.max(np.where(supported, ratio, 0.), axis=1)
        # Bounded engineering score on the same gate scale as the short path.
        return best / (best + LONG_RATIO), supported.any(axis=1)

    def _score(self, end, buffer):
        flat = np.flatnonzero(np.ptp(buffer[-WINDOW:], axis=0) == 0)
        if flat.size:
            self.background = None
            self.long_since = end + 1
            self.last_strong.fill(-LONG_WINDOW)
            return Window(end, 'flat_channel_' + '_'.join(str(i + 1) for i in flat))
        residual = self._remove_startup(buffer, end)
        weak = None
        if len(residual) == LONG_WINDOW and end - self.long_since + 1 >= LONG_WINDOW:
            weak = self._weak_evidence(residual)
        samples = residual[-WINDOW:]
        detrended = detrend(samples, axis=0)
        x = sosfiltfilt(self.sos, detrended, axis=0)
        x -= x.mean(axis=0)
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        # Retain resolved signal dimensions only, including rank-deficient EEG.
        basis = u[:, s > s[0] * 1e-6]
        if not basis.size:
            return Window(end, 'degenerate_eeg')
        if self.gpu_graph is not None:
            scores, peaks_power, noise = self.gpu_graph.score(basis, detrended)
        else:
            scores = tuple(float(np.linalg.svd(basis.T @ ref, compute_uv=False)[0]) for ref in self.references)
            power = np.abs(np.fft.rfft(detrended, axis=0)) ** 2
            peaks_power = power[TARGET_BINS]
            noise = power[TARGET_BINS[:, None] + SIDE_BINS].mean(axis=1)
        peaks = peaks_power > SPECTRAL_RATIO * noise
        # A strong earlier tone can dominate the trailing spectrum after it ends.
        # Fit phase/amplitude on the preceding samples. Require that subtracting
        # this prediction reduces energy on the newest hop in a supported channel.
        history = samples[:-HOP] - samples[:-HOP].mean(axis=0)
        recent = samples[-HOP:] - samples[-HOP:].mean(axis=0)
        support = np.zeros(len(FREQUENCIES), dtype=bool)
        for j, (fit, test_part) in enumerate(self.phase_fits):
            channels = peaks[j]
            if np.any(channels):
                predicted = test_part @ (fit @ history[:, channels])
                gain = np.sum(predicted * (2 * recent[:, channels] - predicted), axis=0)
                support[j] = np.any(gain > 0)
        # Do not re-acquire a recently strong line from its stale long-window
        # energy. The two-second fast path remains active at every hop.
        strong = support & (np.asarray(scores) >= STRONG_SCORE)
        self.last_strong[strong] = end
        if weak is not None:
            weak_scores, weak_support = weak
            weak_support &= end - self.last_strong >= LONG_WINDOW
            weak_scores = np.where(weak_support, weak_scores, 0.)
            scores = tuple(map(float, np.maximum(scores, weak_scores)))
            support |= weak_support
        return Window(end, 'ok', scores, tuple(map(bool, support)))

    def _break(self, index, issue, eeg):
        self.valid_since = int(index) + int(not np.isfinite(eeg).all())
        self.break_reason = str(issue)
        self.background = None
        self.last_strong.fill(-LONG_WINDOW)
        self.gate.candidate, self.gate.streak = None, 0

    def push(self, index, packet, clock, eeg) -> Iterator[Decision]:
        index, packet, clock = map(np.asarray, (index, packet, clock))
        eeg = np.asarray(eeg, dtype=float)
        if (index.ndim != 1 or index.dtype.kind not in 'iu' or eeg.shape != (len(index), 8)
                or packet.shape != index.shape or clock.shape != index.shape):
            raise ValueError('Expected integer sample positions, packet, clock, and N x 8 EEG')
        if not len(index):
            return
        if index[0] < 0 or np.any(index[1:] <= index[:-1]) or (self.previous and index[0] <= self.previous[0]):
            raise ValueError('Sample positions must strictly increase')
        issues = sample_quality(index, packet, clock, eeg, self.previous)
        if self.previous is None and index[0] > 0:
            issues[0] = 'missing_sample_index'
        samples = np.concatenate((self.tail, eeg))
        prefix = len(self.tail)
        breaks = np.flatnonzero(issues)
        k = 0
        # Breaks affect only current and later decisions, regardless of chunk size.
        for offset in np.flatnonzero(index % HOP == HOP - 1):
            while k < len(breaks) and breaks[k] <= offset:
                pos = breaks[k]
                self._break(index[pos], issues[pos], eeg[pos])
                k += 1
            end = int(index[offset])
            if end - self.valid_since + 1 < WINDOW:
                w = Window(end, self.break_reason)
            else:
                stop = prefix + offset + 1
                size = min(LONG_WINDOW, end - self.valid_since + 1)
                w = self._score(end, samples[stop - size:stop])
            yield self.gate.apply(w)
        if k < len(breaks):
            pos = breaks[-1]
            self._break(index[pos], issues[pos], eeg[pos])
        self.previous = (int(index[-1]), float(packet[-1]), float(clock[-1]))
        self.tail = samples[-(LONG_WINDOW - 1):].copy()
