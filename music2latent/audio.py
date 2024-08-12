import torch
import torch.nn.functional as F
import numpy as np

from .hparams import *


def wv2spec(wv, hop_size=256, fac=4):
    X = stft(wv, hop_size=hop_size, fac=fac, device=wv.device)
    X = power2db(torch.abs(X)**2)
    X = normalize(X)
    return X

def spec2wv(S,P, hop_size=256, fac=4):
    S = denormalize(S)
    S = torch.sqrt(db2power(S))
    P = P * np.pi
    SP = torch.complex(S * torch.cos(P), S * torch.sin(P))
    return istft(SP, fac=fac, hop_size=hop_size, device=SP.device)

def denormalize_realimag(x):
    x = x/beta_rescale
    return torch.sign(x)*(x.abs()**(1./alpha_rescale))

def normalize_complex(x):
    return beta_rescale*(x.abs()**alpha_rescale).to(torch.complex64)*torch.exp(1j*torch.angle(x).to(torch.complex64))

def denormalize_complex(x):
    x = x/beta_rescale
    return (x.abs()**(1./alpha_rescale)).to(torch.complex64)*torch.exp(1j*torch.angle(x).to(torch.complex64))

def wv2complex(wv, hop_size=256, fac=4):
    X = stft(wv, hop_size=hop_size, fac=fac, device=wv.device)
    return X[:,:hop_size*2,:]

def wv2realimag(wv, hop_size=256, fac=4):
    X = wv2complex(wv, hop_size, fac)
    X = normalize_complex(X)
    return torch.stack((torch.real(X),torch.imag(X)), -3)

def realimag2wv(x, hop_size=256, fac=4):
    x = torch.nn.functional.pad(x, (0,0,0,1))
    real,imag = torch.chunk(x, 2, -3)
    X = torch.complex(real.squeeze(-3),imag.squeeze(-3))
    X = denormalize_complex(X)
    return istft(X, fac=fac, hop_size=hop_size, device=X.device).clamp(-1.,1.)

def to_representation_encoder(x):
    return wv2realimag(x, hop)

def to_representation(x):
    return wv2realimag(x, hop)

def to_waveform(x):
    return realimag2wv(x, hop)

def overlap_and_add(signal, frame_step):

    outer_dimensions = signal.shape[:-2]
    outer_rank = torch.numel(torch.tensor(outer_dimensions))

    def full_shape(inner_shape):
      s = torch.cat([torch.tensor(outer_dimensions), torch.tensor(inner_shape)], 0)
      s = list(s)
      s = [int(el) for el in s]
      return s

    frame_length = signal.shape[-1]
    frames = signal.shape[-2]

    # Compute output length.
    output_length = frame_length + frame_step * (frames - 1)

    # Compute the number of segments, per frame.
    segments = -(-frame_length // frame_step)  # Divide and round up.

    signal = torch.nn.functional.pad(signal, (0, segments * frame_step - frame_length, 0, segments))

    shape = full_shape([frames + segments, segments, frame_step])
    signal = torch.reshape(signal, shape)

    perm = torch.cat([torch.arange(0, outer_rank), torch.tensor([el+outer_rank for el in [1, 0, 2]])], 0)
    perm = list(perm)
    perm = [int(el) for el in perm]
    signal = torch.permute(signal, perm)

    shape = full_shape([(frames + segments) * segments, frame_step])
    signal = torch.reshape(signal, shape)

    signal = signal[..., :(frames + segments - 1) * segments, :]

    shape = full_shape([segments, (frames + segments - 1), frame_step])
    signal = torch.reshape(signal, shape)

    signal = signal.sum(-3)

    # Flatten the array.
    shape = full_shape([(frames + segments - 1) * frame_step])
    signal = torch.reshape(signal, shape)

    # Truncate to final length.
    signal = signal[..., :output_length]

    return signal

def inverse_stft_window(frame_length, frame_step, forward_window):
    denom = forward_window**2
    overlaps = -(-frame_length // frame_step)
    denom = F.pad(denom, (0, overlaps * frame_step - frame_length))
    denom = torch.reshape(denom, [overlaps, frame_step])
    denom = torch.sum(denom, 0, keepdim=True)
    denom = torch.tile(denom, [overlaps, 1])
    denom = torch.reshape(denom, [overlaps * frame_step])
    return forward_window / denom[:frame_length]

def istft(SP, fac=4, hop_size=256, device='cuda'):
    x = torch.fft.irfft(SP, dim=-2)
    window = torch.hann_window(fac*hop_size).to(device)
    window = inverse_stft_window(fac*hop_size, hop_size, window)
    x = x*window.unsqueeze(-1)
    return overlap_and_add(x.permute(0,2,1), hop_size)

def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """
    equivalent of tf.signal.frame
    """
    signal_length = signal.shape[axis]
    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
        pad_size = int(frame_length - rest_samples)
        if pad_size != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = F.pad(signal, pad_axis, "constant", pad_value)
    frames = signal.unfold(axis, frame_length, frame_step)
    return frames

def stft(wv, fac=4, hop_size=256, device='cuda'):
    window = torch.hann_window(fac*hop_size).to(device)
    framed_signals = frame(wv, fac*hop_size, hop_size)
    framed_signals = framed_signals*window
    return torch.fft.rfft(framed_signals, n=None, dim=- 1, norm=None).permute(0,2,1)

def normalize(S, mu_rescale=-25., sigma_rescale=75.):
    return (S - mu_rescale) / sigma_rescale

def denormalize(S, mu_rescale=-25., sigma_rescale=75.):
    return (S * sigma_rescale) + mu_rescale

def db2power(S_db, ref=1.0):
    return ref * torch.pow(10.0, 0.1 * S_db)

def power2db(power, ref_value=1.0, amin=1e-10):
    log_spec = 10.0 * torch.log10(torch.maximum(torch.tensor(amin), power))
    log_spec -= 10.0 * torch.log10(torch.maximum(torch.tensor(amin), torch.tensor(ref_value)))
    return log_spec