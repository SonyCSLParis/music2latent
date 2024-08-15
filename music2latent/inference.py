import soundfile as sf
import torch
import numpy as np

from .hparams import *
from .hparams_inference import *
from .utils import *
from .models import *
from .audio import *


class EncoderDecoder:
    def __init__(self, load_path_inference=None, device=None):
        download_model()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.load_path_inference = load_path_inference
        if load_path_inference is None:
            self.load_path_inference = load_path_inference_default
        self.get_models()
        
    def get_models(self):
        gen = UNet().to(self.device)
        checkpoint = torch.load(self.load_path_inference, map_location=self.device)
        gen.load_state_dict(checkpoint['gen_state_dict'], strict=False)
        self.gen = gen

    def encode(self, path_or_audio, max_waveform_length=None, max_batch_size=None, extract_features=False):
        '''
        path_or_audio: path of audio sample to encode or numpy array of waveform to encode
        max_waveform_length: maximum length of waveforms in the batch for encoding: tune it depending on the available GPU memory
        max_batch_size: maximum inference batch size for encoding: tune it depending on the available GPU memory

        WARNING! if input is numpy array of stereo waveform, it must have shape [waveform_samples, audio_channels]

        Returns latents with shape [audio_channels, dim, length]
        '''
        if max_waveform_length is None:
            max_waveform_length = max_waveform_length_encode
        if max_batch_size is None:
            max_batch_size = max_batch_size_encode
        return encode_audio_inference(path_or_audio, self, max_waveform_length, max_batch_size, device=self.device, extract_features=extract_features)
    
    def decode(self, latent, denoising_steps=1, max_waveform_length=None, max_batch_size=None):
        '''
        latent: numpy array of latents to decode with shape [audio_channels, dim, length]
        denoising_steps: number of denoising steps to use for decoding
        max_waveform_length: maximum length of waveforms in the batch for decoding: tune it depending on the available GPU memory
        max_batch_size: maximum inference batch size for decoding: tune it depending on the available GPU memory

        Returns numpy array of decoded waveform with shape [waveform_samples, audio_channels]
        '''
        if max_waveform_length is None:
            max_waveform_length = max_waveform_length_decode
        if max_batch_size is None: 
            max_batch_size = max_batch_size_decode
        return decode_latent_inference(latent, self, max_waveform_length, max_batch_size, diffusion_steps=denoising_steps, device=self.device)






# decode samples with consistency model to real/imag STFT spectrograms
# Parameters:
#   model: trained consistency model 
#   latents: latent representation with shape [audio_channels/batch_size, dim, length]
#   diffusion_steps: number of steps
# Returns:
#   decoded_spectrograms with shape [audio_channels/batch_size, data_channels, hop*2, length*downscaling_factor]
def decode_to_representation(model, latents, diffusion_steps=1):
    num_samples = latents.shape[0]
    downscaling_factor = 2**freq_downsample_list.count(0)
    sample_length = int(latents.shape[-1]*downscaling_factor)
    initial_noise = torch.randn((num_samples, data_channels, hop*2, sample_length)).cuda()*sigma_max
    decoded_spectrograms = reverse_diffusion(model, initial_noise, diffusion_steps, latents=latents)
    return decoded_spectrograms




# Encode audio sample for inference
# Parameters:
#   audio_path: path of audio sample
#   model: trained consistency model
#   device: device to run the model on
# Returns:
#   latent: compressed latent representation with shape [audio_channels, dim, latent_length]
@torch.no_grad()
def encode_audio_inference(audio_path, trainer, max_waveform_length_encode, max_batch_size_encode, device='cuda', extract_features=False):
    trainer.gen = trainer.gen.to(device)
    trainer.gen.eval()
    downscaling_factor = 2**freq_downsample_list.count(0)
    if is_path(audio_path):
        audio, sr = sf.read(audio_path, dtype='float32', always_2d=True)
        audio = np.transpose(audio, [1,0])
    else:
        audio = audio_path
        sr = None
        if len(audio.shape)==1:
            # check if audio is numpy array, then use np.expand_dims, if it is a pytorch tensor, then use torch.unsqueeze
            if isinstance(audio, np.ndarray):
                audio = np.expand_dims(audio, 0)
            else:
                audio = torch.unsqueeze(audio, 0)
    audio_channels = audio.shape[0]
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).to(device)
    else:
        # check if audio tensor is on cpu. if it is, move it to the device
        if audio.device.type=='cpu':
            audio = audio.to(device)

    # EXPERIMENTAL: crop audio to be divisible by downscaling_factor
    cropped_length = ((((audio.shape[-1]-3*hop)//hop)//downscaling_factor)*hop*downscaling_factor)+3*hop
    audio = audio[:,:cropped_length]

    repr_encoder = to_representation_encoder(audio)
    sample_length = repr_encoder.shape[-1]
    max_sample_length = (int(max_waveform_length_encode/hop)//downscaling_factor)*downscaling_factor

    # if repr_encoder is longer than max_sample_length, chunk it into max_sample_length chunks, the last chunk will be zero-padded, then concatenate the chunks into the batch dimension (before encoding them)
    pad_size = 0
    if sample_length > max_sample_length:
        # pad repr_encoder with copies of the sample to be divisible by max_sample_length
        pad_size = max_sample_length - (sample_length % max_sample_length)
        # repeat repr_encoder such that repr_encoder.shape[-1] is higher than pad_size, then crop it such that repr_encoder.shape[-1]=pad_size
        repr_encoder_pad = torch.cat([repr_encoder for _ in range(1+(pad_size//repr_encoder.shape[-1]))], dim=-1)[:,:,:,:pad_size]
        repr_encoder = torch.cat([repr_encoder, repr_encoder_pad], dim=-1)
        repr_encoder = torch.split(repr_encoder, max_sample_length, dim=-1)
        repr_encoder = torch.cat(repr_encoder, dim=0)
    # encode repr_encoder using a maximum batch size (dimension 0) of max_batch_size_inference, if repr_encoder is longer than max_batch_size_inference, chunk it into max_batch_size_inference chunks, the last chunk will maybe have less samples in the batch, then encode the chunks and concatenate the results into the batch dimension
    max_batch_size = max_batch_size_encode
    if repr_encoder.shape[0] > max_batch_size:
        repr_encoder_ls = torch.split(repr_encoder, max_batch_size, dim=0)
        latent_ls = []
        for i in range(len(repr_encoder_ls)):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=mixed_precision): # disable float16 for encoding (can cause nans)
                latent = trainer.gen.encoder(repr_encoder_ls[i], extract_features=extract_features)
            latent_ls.append(latent)
        latent = torch.cat(latent_ls, dim=0)
    else:
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=mixed_precision): # disable float16 for encoding (can cause nans)
            latent = trainer.gen.encoder(repr_encoder, extract_features=extract_features)
    # split samples
    if latent.shape[0]>1:
        latent_ls = torch.split(latent, audio_channels, 0)
        latent = torch.cat(latent_ls, -1)
    latent = latent[:,:,:latent.shape[-1]-(pad_size//downscaling_factor)]
    if extract_features:
        return latent
    else:
        return latent/sigma_rescale



# Decode latent representation for inference, use the same framework as in encode_audio_inference, but in reverse order for decoding
# Parameters:
#   latent: compressed latent representation with shape [audio_channels, dim, length]
#   model: trained consistency model
#   diffusion_steps: number of diffusion steps to use for decoding
#   device: device to run the model on
# Returns:
#   audio: numpy array of decoded waveform with shape [waveform_samples, audio_channels]
@torch.no_grad()
def decode_latent_inference(latent, trainer, max_waveform_length_decode, max_batch_size_decode, diffusion_steps=1, device='cuda'):
    trainer.gen = trainer.gen.to(device)
    trainer.gen.eval()
    downscaling_factor = 2**freq_downsample_list.count(0)
    latent = latent*sigma_rescale
    # check if latent is numpy array, then convert to tensor
    if isinstance(latent, np.ndarray):
        latent = torch.from_numpy(latent)
    # check if latent tensor is on cpu. if it is, move it to the device
    if latent.device.type=='cpu':
        latent = latent.to(device)
    # if latent has only 2 dimensions, add a third dimension as axis 0
    if len(latent.shape)==2:
        latent = torch.unsqueeze(latent, 0)
    audio_channels = latent.shape[0]
    latent_length = latent.shape[-1]
    max_latent_length = int(max_waveform_length_decode/hop)//downscaling_factor

    # if latent is longer than max_latent_length, chunk it into max_latent_length chunks, the last chunk will be zero-padded, then concatenate the chunks into the batch dimension (before decoding them)
    pad_size = 0
    if latent_length > max_latent_length:
        # pad latent with copies of itself to be divisible by max_latent_length
        pad_size = max_latent_length - (latent_length % max_latent_length)
        # repeat latent such that latent.shape[-1] is higher than pad_size, then crop it such that latent.shape[-1]=pad_size
        latent_pad = torch.cat([latent for _ in range(1+(pad_size//latent.shape[-1]))], dim=-1)[:,:,:pad_size]
        latent = torch.cat([latent, latent_pad], dim=-1)
        latent = torch.split(latent, max_latent_length, dim=-1)
        latent = torch.cat(latent, dim=0)
    # decode latent using a maximum batch size (dimension 0) of max_batch_size_inference, if latent is longer than max_batch_size_inference, chunk it into max_batch_size_inference chunks, the last chunk will maybe have less samples in the batch, then decode the chunks and concatenate the results into the batch dimension
    max_batch_size = max_batch_size_decode
    if latent.shape[0] > max_batch_size:
        latent_ls = torch.split(latent, max_batch_size, dim=0)
        repr_ls = []
        for i in range(len(latent_ls)):
            repr = decode_to_representation(trainer.gen, latent_ls[i], diffusion_steps=diffusion_steps)
            repr_ls.append(repr)
        repr = torch.cat(repr_ls, dim=0)
    else:
        repr = decode_to_representation(trainer.gen, latent, diffusion_steps=diffusion_steps)
    # split samples
    if repr.shape[0]>1:
        repr_ls = torch.split(repr, audio_channels, 0)
        repr = torch.cat(repr_ls, -1)
    repr = repr[:,:,:,:repr.shape[-1]-(pad_size*downscaling_factor)]
    return to_waveform(repr)