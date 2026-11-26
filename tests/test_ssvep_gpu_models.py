"""Model library and DirectML integration on synthetic EEG. No board or stimulus."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from neurostation_models import BUILTIN, WINDOW_CONTRACT

ROOT = Path(__file__).resolve().parents[1]


def make_fixture(path, contract=WINDOW_CONTRACT):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    weights = numpy_helper.from_array(np.eye(8, 4, dtype=np.float32), 'weights')
    graph = helper.make_graph([
        helper.make_node('ReduceMean', ['eeg'], ['mean'], axes=[2], keepdims=0),
        helper.make_node('MatMul', ['mean', 'weights'], ['raw']),
        helper.make_node('Sigmoid', ['raw'], ['scores']),
    ], 'fixture', [helper.make_tensor_value_info('eeg', TensorProto.FLOAT, [1, 8, 500])],
       [helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 4])], [weights])
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 17)])
    helper.set_model_props(model, {'neurostation.name': 'Fixture model', 'neurostation.version': 'v2',
                                   'neurostation.contract': contract})
    onnx.save(model, path)


@unittest.skipUnless(os.name == 'nt', 'DirectML integration requires Windows')
class LocalModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Inspect the same runtime as the worker in a fresh process, even if
        # another suite has loaded the training-only CPU wheel in this process.
        env = dict(os.environ)
        private = Path(sys.prefix) / 'ssvep-directml'
        if private.exists():
            env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(private), env.get('PYTHONPATH'))))
        result = subprocess.run([sys.executable, '-c',
            "import importlib.util as u; import sys; "
            "sys.exit(77) if u.find_spec('onnxruntime') is None else None; "
            "import onnxruntime as o; sys.exit(0 if any(d.ep_name=='DmlExecutionProvider' "
            "and d.device.type==o.OrtHardwareDeviceType.GPU for d in getattr(o,'get_ep_devices',lambda:[])()) else 77)"],
            env=env, timeout=45)
        if result.returncode == 77:
            raise unittest.SkipTest('DirectML hardware unavailable')
        result.check_returncode()

    def test_reference_parity_and_discontinuity(self):
        check = '''
import numpy as np
from neurostation_models import BUILTIN
from eeg_tools.ssvep_gpu import load_detector
from eeg_tools.continuous_ssvep import ContinuousDetector
idx = np.arange(7500)
eeg = np.random.default_rng(350).normal(0, .2, (len(idx), 8))
for j, hz in enumerate((10,12,15,20), start=1):
    t = idx[j*1500:(j+1)*1500] / 250
    eeg[j*1500:(j+1)*1500, j] += 2*np.sin(2*np.pi*t*hz)
packet = idx % 256
packet[4100:] = (packet[4100:]+1) % 256
cpu = ContinuousDetector(); gpu, info = load_detector(BUILTIN)
assert info['provider'] == 'DmlExecutionProvider'
states = set()
for start in range(0, len(idx), 91):
    s = slice(start, start + 91)
    args = idx[s], packet[s], idx[s]/250, eeg[s]
    for a,b in zip(cpu.push(*args), gpu.push(*args), strict=True):
        assert (a.window.end_sample,a.window.quality,a.hz,a.state)==(b.window.end_sample,b.window.quality,b.hz,b.state), (a,b)
        np.testing.assert_allclose(a.window.scores,b.window.scores,atol=3e-5,rtol=3e-5)
        states.add((b.hz,b.window.quality))
assert all((f,'ok') in states for f in [10,12,15,20,None]) and (None,'packet_discontinuity') in states
'''
        subprocess.run([sys.executable, '-B', '-c', check], cwd=ROOT, check=True, timeout=90)

    @unittest.skipUnless(importlib.util.find_spec('onnx'), 'building fixture requires onnx')
    def test_import_select_reimport_and_integrity(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = dict(os.environ, NEUROSTATION_SETTINGS=tmp)
            def cli(*args, success=True):
                command = [os.environ['NEUROSTATION_TEST_MODEL_EXE']] if os.environ.get('NEUROSTATION_TEST_MODEL_EXE') else [sys.executable, str(ROOT/'workstation.py')]
                done = subprocess.run([*command,'--ssvep-model',*args], cwd=root, env=env, capture_output=True, timeout=60)
                self.assertEqual(done.returncode == 0, success, (done.stdout,done.stderr))
                return json.loads(done.stdout)
            file = root/'import.onnx'
            make_fixture(file)
            key = hashlib.sha256(file.read_bytes()).hexdigest()
            for _ in range(2):
                imported = cli('import',str(file))
                self.assertEqual((imported['id'],imported['name'],imported['version']), (key,'Fixture model','v2'))
            self.assertEqual(cli('check',key)['windows'], 30)
            make_fixture(file,'wrong contract')
            self.assertIn('contract',cli('import',str(file),success=False)['error'])
            self.assertEqual([p.name for p in (root/'SSVEPModels').iterdir()],[key])
            (root/'SSVEPModels'/key/'model.onnx').write_bytes(b'modified')
            self.assertIn('hash mismatch',cli('check',key,success=False)['error'])
