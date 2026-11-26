"""Local SSVEP model inventory. Listing models never loads a numerical runtime."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re

from neurostation_contract import CONTINUOUS_SSVEP_METHOD, default_user_settings_directory

BUILTIN = CONTINUOUS_SSVEP_METHOD
WINDOW_CONTRACT = 'neurostation.ssvep.eeg-uv.8x500.250hz.10-12-15-20.v1'


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    version: str
    sha256: str
    path: Path

    @property
    def label(self):
        return f'{self.name} · {self.version}'

    def identity(self):
        return {key: val for key, val in asdict(self).items() if key != 'path'}


_BUILTIN = Model(BUILTIN, 'Continuous CCA', 'v3',
                 '0d829cb931dab253afe3baaf13be29f161e4e09b17599e772f22431f1f026100',
                 Path(__file__).resolve().parent / 'assets/ssvep_models/cca-v3.onnx')


def library():
    return default_user_settings_directory() / 'SSVEPModels'


def model_by_id(key):
    if key == BUILTIN:
        return _BUILTIN
    if not isinstance(key, str) or not re.fullmatch('[0-9a-f]{64}', key):
        raise ValueError('Unknown SSVEP model ID')
    folder = library() / key
    info = json.loads((folder / 'model.json').read_text(encoding='utf-8'))
    if not isinstance(info, dict) or info.get('sha256') != key or info.get('id') != key:
        raise ValueError('Invalid SSVEP model record')
    return Model(key, _label(info.get('name')), _label(info.get('version')), key, folder / 'model.onnx')


def _label(value):
    if not isinstance(value, str) or not value.strip() or not value.isprintable() or len(value) > 100:
        raise ValueError('Invalid SSVEP model name/version')
    return value.strip()


def list_models():
    models = [_BUILTIN]
    for entry in sorted(library().glob('*')):
        try:
            model = model_by_id(entry.name)
            if model.id != BUILTIN and model.path.is_file():
                models.append(model)
        except (ValueError, OSError):
            continue
    return models
