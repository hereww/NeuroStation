"""Offline harmonic-spectrum MLP for Wang2016 three-frequency task decoding."""
from __future__ import annotations
import numpy as np
from scipy import signal
import torch
from torch import nn

FREQUENCIES = (10, 12, 15)
COMMANDS = {10: "MOVE_LEFT", 12: "MOVE_RIGHT", 15: "SELECT"}
FLAGS = {"status":"formal", "exploratory_only":True, "runtime_enabled":False,
 "production_validated":False, "default_deployment_authorized":False, "executed":False,
 "three_frequency_only":True, "missing20":True, "missing_target_frequencies_hz":[20],
 "personal_transfer_supported":False}
MODEL_SPEC = {"name":"harmonic_spectrum_mlp_three_frequency_v1", "input_features":72,
 "hidden_features":24, "output_classes":3, "activation":"GELU", "dropout":0.15}
PREPROCESSING = {"sample_rate_hz":250, "training_epoch_shape":[8,1250],
 "channel_order":["Fp1","Fp2","C3","C4","P7","P8","O1","O2"],
 "detrend":"linear per trial/channel", "window":"Hann symmetric",
 "spectrum":"abs(rfft(window*detrend(eeg)))**2; actual n; no padding",
 "feature":"log((mean harmonic power +/-0.4Hz + eps)/(mean 6..90Hz power + eps))",
 "harmonics":[1,2,3], "feature_order":"channel/frequency/harmonic", "eps":1e-20,
 "standardization":"train-only population mean/std; std floor 1e-6", "resampling":False,
 "rest_or_zero_padding":False}
GATE = {"min_probability":0.65, "min_probability_margin":0.15,
 "selection":"fixed before training; no threshold tuning"}

def require(ok, message):
    if not ok: raise ValueError(message)

def spectral_features(eeg: np.ndarray) -> np.ndarray:
    x=np.asarray(eeg)
    require(x.ndim==2 and x.shape[0]==8 and x.shape[1]>=250, "features require actual EEG (8,n>=250)")
    require(x.dtype.kind in "fiu" and np.isfinite(x).all(), "EEG must be finite real data")
    x=signal.detrend(x.astype(np.float64),axis=-1,type="linear")
    n=x.shape[-1]; power=np.abs(np.fft.rfft(x*np.hanning(n),axis=-1))**2
    bins=np.fft.rfftfreq(n,1/250); band=power[:,(bins>=6)&(bins<=90)].mean(axis=-1)
    values=np.empty((8,3,3),dtype=np.float64)
    for i,f in enumerate(FREQUENCIES):
        for j,h in enumerate((1,2,3)):
            selected=np.abs(bins-f*h)<=0.4+1e-12
            require(selected.any(), "input duration cannot resolve locked harmonic band")
            values[:,i,j]=np.log((power[:,selected].mean(axis=-1)+1e-20)/(band+1e-20))
    result=values.reshape(72).astype(np.float32)
    require(np.isfinite(result).all(), "nonfinite spectral features")
    return result

class HarmonicSpectrumMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("feature_mean",torch.zeros(72))
        self.register_buffer("feature_std",torch.ones(72))
        self.layers=nn.Sequential(nn.Linear(72,24),nn.GELU(),nn.Dropout(0.15),nn.Linear(24,3))
    def forward(self,features): return self.layers((features-self.feature_mean)/self.feature_std)

def command_prediction(probabilities, gate=None):
    gate=GATE if gate is None else gate; values=np.asarray(probabilities,dtype=float)
    require(values.shape==(3,) and np.isfinite(values).all() and (values>=0).all() and
            (values<=1).all() and np.isclose(values.sum(),1,atol=1e-6), "invalid three-class probabilities")
    require(gate==GATE,"gate differs from fixed preregistered thresholds")
    order=np.argsort(-values,kind="stable"); best,second=order[:2]
    accepted=bool(values[best]>=gate["min_probability"] and values[best]-values[second]>=gate["min_probability_margin"])
    frequency=FREQUENCIES[best]
    return {"predicted_frequency_hz":frequency,"probabilities":values.tolist(),
            "margin":float(values[best]-values[second]),"accepted":accepted,
            "command":COMMANDS[frequency] if accepted else "UNKNOWN","executed":False}
