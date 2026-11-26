"""Rebuild the shipped CCA projection and short-window spectral evidence graph."""
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build():
    n, fs = 500, 250
    hz = np.array([10, 12, 15, 20])
    t = np.arange(n)[:, None] / fs
    angle = 2 * np.pi * t * hz
    refs = np.stack([np.linalg.qr(np.stack((np.sin(a), np.cos(a)), axis=1))[0] for a in angle.T])
    bins = 2 * hz[:, None] + np.array([0, -3, -2, -1, 1, 2, 3])
    dft = 2 * np.pi * t * (bins.flatten() * fs / n)
    nodes, constants = [], []

    def const(name, value):
        constants.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def op(kind, name, *inputs, **attrs):
        nodes.append(helper.make_node(kind, list(inputs), [name], **attrs))
        return name

    refs = const('refs', refs.transpose(0, 2, 1).reshape(8, n).astype('float32'))
    coeff = op('Reshape', 'coeff', op('MatMul', 'project', refs, 'basis'), const('shape', np.array([4, 2, 8], np.int64)))
    a, b = [op('Gather', name, coeff, const(name + '_i', np.int64(i)), axis=1) for i, name in enumerate(('a','b'))]
    def dot(name, x, y):
        return op('ReduceSum', name, op('Mul', name+'prod', x, y), const(name+'axis', np.array([1],np.int64)), keepdims=0)
    aa, bb, ab = dot('aa',a,a), dot('bb',b,b), dot('ab',a,b)
    # Largest singular value of a 2 x rank projection via its 2 x 2 Gram matrix.
    diff = op('Sub','diff',aa,bb)
    trace = op('Add','trace',aa,bb)
    disc = op('Add','disc',op('Mul','delta2',diff,diff),op('Mul','fourab2',const('four',np.float32(4)),op('Mul','ab2',ab,ab)))
    op('Sqrt','scores',op('Mul','eig',const('half',np.float32(.5)),op('Add','sum',trace,op('Sqrt','radius',disc))))
    re, im = [op('MatMul', name, const(name+'dft', fun(dft).T.astype('float32')), 'detrended') for name,fun in [('re',np.cos),('im',np.sin)]]
    powers = op('Add','power',op('Mul','re2',re,re),op('Mul','im2',im,im))
    op('Reshape','band_power',powers,const('band_shape',np.array([4,7,8],np.int64)))
    # The long-window branch uses complex DFT coefficients at target and noise
    # bins; projecting on these frequencies is equivalent to slicing rfft.
    long_bins = np.unique(np.concatenate([np.arange(4*f - 20, 4*f + 21) for f in hz]))
    long_phase = 2 * np.pi * np.arange(1000)[:, None] * long_bins / 1000
    recent_phase = 2 * np.pi * np.arange(250)[:, None] * hz / 250
    for key, phase, input_name in [('long', long_phase, 'history'), ('recent', recent_phase, 'recent')]:
        for component, weights in [('real', np.cos(phase)), ('imag', -np.sin(phase))]:
            out = key + '_' + component
            op('MatMul', out, const(out + '_weights', weights.T.astype('float32')), input_name)

    def tensor(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    graph = helper.make_graph(nodes,'Continuous CCA v3', [tensor('basis',[n,8]),tensor('detrended',[n,8]),
                               tensor('history',[1000,8]), tensor('recent',[250,8])],
                              [tensor('scores',[4]),tensor('band_power',[4,7,8]),
                               *[tensor('long_'+c,[len(long_bins),8]) for c in ('real','imag')],
                               *[tensor('recent_'+c,[4,8]) for c in ('real','imag')]], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',17)], ir_version=8)
    onnx.checker.check_model(model)
    return model.SerializeToString()


if __name__ == '__main__':
    dest = Path(__file__).resolve().parents[1] / 'assets/ssvep_models/cca-v3.onnx'
    dest.parent.mkdir(exist_ok=True)
    dest.write_bytes(build())
