"""Validate and publish content-addressed local ONNX model files."""
from pathlib import Path
import hashlib
import json
import tempfile

from neurostation_models import Model, _label, library, model_by_id

MAX_BYTES = 128 * 1024 * 1024


def read_graph(path):
    with Path(path).open('rb') as stream:
        data = stream.read(MAX_BYTES + 1)
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Expected a self-contained ONNX file up to 128 MiB')
    return data


def import_model(path):
    from .ssvep_gpu import open_gpu_graph, check_window_graph
    data = read_graph(path)
    digest = hashlib.sha256(data).hexdigest()
    session, device = open_gpu_graph(data)
    check_window_graph(session)
    metadata = session.get_modelmeta().custom_metadata_map
    model = Model(digest, _label(metadata.get('neurostation.name')), _label(metadata.get('neurostation.version')),
                  digest, library() / digest / 'model.onnx')
    if model.path.exists():
        saved = model_by_id(digest)
        if hashlib.sha256(read_graph(saved.path)).hexdigest() != digest:
            raise ValueError('Stored model hash mismatch')
        return saved, device
    library().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=library()) as scratch:
        stage = Path(scratch) / 'model'
        stage.mkdir()
        (stage / 'model.onnx').write_bytes(data)
        (stage / 'model.json').write_text(json.dumps(model.identity(), ensure_ascii=False), encoding='utf-8')
        try:
            stage.rename(model.path.parent)
        except FileExistsError:  # Another importer published this content-addressed model.
            return model_by_id(digest), device
    return model, device
