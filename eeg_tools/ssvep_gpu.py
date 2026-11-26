"""GPU-only graph execution for live SSVEP. The desktop shell imports no ORT."""
import hashlib
import sys
from pathlib import Path

import numpy as np

from .continuous_ssvep import ContinuousDetector, Gate, Window, WINDOW
from neurostation_models import BUILTIN, WINDOW_CONTRACT, model_by_id
from .ssvep_models import read_graph


def open_gpu_graph(data):
    # A source checkout can keep the DirectML wheel separate from the CPU wheel
    # used by its training tools. Packaged workers ship their own DirectML DLLs.
    private_runtime = Path(sys.prefix) / 'ssvep-directml'
    if not getattr(sys, 'frozen', False) and private_runtime.is_dir() and str(private_runtime) not in sys.path:
        sys.path.insert(0, str(private_runtime))
    try:
        import onnxruntime as ort
        gpus = [d for d in ort.get_ep_devices() if d.ep_name == 'DmlExecutionProvider'
                and d.device.type == ort.OrtHardwareDeviceType.GPU]
        if not gpus:
            raise RuntimeError('No DirectML hardware GPU found')
        gpu = min(gpus, key=lambda d: (d.device.metadata.get('Discrete') != '1', int(d.ep_options['device_id'])))
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        session = ort.InferenceSession(data, options, providers=[(gpu.ep_name, gpu.ep_options)], enable_fallback=0)
        session.disable_fallback()
        if session.get_providers()[0] != gpu.ep_name:
            raise RuntimeError('DirectML failed to initialize')
    except Exception as error:
        raise RuntimeError(f'本机 GPU 初始化失败 / Local GPU initialization failed: {error}') from error
    return session, dict(provider=gpu.ep_name, device=gpu.device.metadata.get('Description', 'GPU'),
                         device_id=gpu.ep_options['device_id'], runtime=ort.__version__)


def window_scores(session, eeg):
    scores, = session.run(['scores'], {'eeg': np.ascontiguousarray(eeg.T[None], dtype=np.float32)})
    if scores.shape != (1, 4) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('SSVEP scores must be four finite values between 0 and 1')
    return tuple(map(float, scores[0]))


def check_window_graph(session):
    if session.get_modelmeta().custom_metadata_map.get('neurostation.contract') != WINDOW_CONTRACT:
        raise ValueError(f'Expected ONNX metadata neurostation.contract={WINDOW_CONTRACT}')
    for ports, name, shape in [(session.get_inputs(), 'eeg', [1,8,500]), (session.get_outputs(), 'scores', [1,4])]:
        if len(ports) != 1 or (ports[0].name, ports[0].type, ports[0].shape) != (name, 'tensor(float)', shape):
            raise ValueError(f'Expected float32 {name}{shape}')
    window_scores(session, np.zeros((WINDOW, 8)))


class WindowModelDetector(ContinuousDetector):
    """Reuse the transport/quality checks, causal window cadence and decision gate."""
    def __init__(self, gate, session):
        super().__init__(gate)
        self.session = session

    def _score(self, end, buffer):
        eeg = buffer[-WINDOW:]
        flat = np.flatnonzero(np.ptp(eeg, axis=0) == 0)
        if flat.size:
            return Window(end, 'flat_channel_' + '_'.join(str(i+1) for i in flat))
        scores = window_scores(self.session, eeg)
        # Imported models supply their own frequency evidence. The shared gate
        # still requires a minimum score, margin and successive confirmations.
        return Window(end, 'ok', scores, (True,) * len(scores))


class CcaGraph:
    def __init__(self, session):
        self.session = session
        self.blank = {key: np.zeros((n,8), np.float32) for key, n in
                      [('basis',500), ('detrended',500), ('history',1000), ('recent',250)]}

    def long_evidence(self, data, bins, noise_bins):
        arrays = self.session.run(['long_real','long_imag','recent_real','recent_imag'],
                                  {**self.blank, 'history': data.astype(np.float32), 'recent': data[-250:].astype(np.float32)})
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('Non-finite GPU spectral output')
        spec = arrays[0].astype(float) + 1j * arrays[1].astype(float)
        keys = np.unique(np.concatenate([bins, *noise_bins]))
        floor = np.stack([(np.abs(spec[np.searchsorted(keys, side)]) ** 2).mean(axis=0) for side in noise_bins])
        return spec[np.searchsorted(keys, bins)], floor, (arrays[2] + 1j * arrays[3]) / 250

    def score(self, basis, detrended):
        padded = np.zeros((WINDOW, 8), dtype=np.float32)
        padded[:, :basis.shape[1]] = basis
        scores, power = self.session.run(['scores','band_power'],
                                         {**self.blank, 'basis': padded, 'detrended': np.asarray(detrended, dtype=np.float32)})
        if not np.isfinite(scores).all() or not np.isfinite(power).all():
            raise ValueError('Non-finite GPU CCA output')
        return tuple(map(float, scores.clip(0, 1))), power[:, 0], power[:, 1:].mean(axis=1)


def load_detector(key, gate=Gate()):
    model = model_by_id(key)
    data = read_graph(model.path)
    if hashlib.sha256(data).hexdigest() != model.sha256:
        raise ValueError('SSVEP model hash mismatch')
    session, gpu = open_gpu_graph(data)
    if key == BUILTIN:
        graph = CcaGraph(session)
        graph.score(np.zeros((WINDOW, 1)), np.zeros((WINDOW, 8)))  # actual dispatch before acquisition
        detector = ContinuousDetector(gate, gpu_graph=graph)
    else:
        check_window_graph(session)
        detector = WindowModelDetector(gate, session)
    return detector, {**model.identity(), **gpu}


def model_command(args):
    """CLI used by the desktop model page and the packaged GPU smoke check."""
    import argparse
    import json
    from .ssvep_models import import_model
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('import', 'check'))
    parser.add_argument('value')
    command = parser.parse_args(args)
    try:
        if command.action == 'import':
            model, device = import_model(Path(command.value))
            result = {**model.identity(), **device}
        else:
            detector, result = load_detector(command.value)
            index = np.arange(1500)
            eeg = np.random.default_rng(11).normal(0, .1, (len(index), 8))
            eeg[500:, 0] += np.sin(2 * np.pi * 12 * index[500:] / 250)
            decisions = list(detector.push(index, index % 256, index / 250, eeg))
            if len(decisions) != 30 or decisions[-1].window.quality != 'ok':
                raise RuntimeError('Synthetic stream produced incomplete output')
            if command.value == BUILTIN and decisions[-1].hz != 12:
                raise RuntimeError('Synthetic 12 Hz GPU model check failed')
            result.update(windows=len(decisions), last_frequency_hz=decisions[-1].hz)
        code = 0
    except Exception as exc:
        result, code = {'error': str(exc)}, 1
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return code
