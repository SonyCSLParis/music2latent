import os
from os import path as ospath
from huggingface_hub import hf_hub_download
import torch

from .hparams import *

# Get scaling coefficients c_skip, c_out, c_in based on noise sigma
# These are used to scale the input and output of the consistency model, while satisfying the boundary condition for consistency models
# Parameters:
# sigma: noise level
# Returns:
# c_skip, c_out, c_in: scaling coefficients
def get_c(sigma):
    sigma_correct = sigma_min
    c_skip = (sigma_data**2.)/(((sigma-sigma_correct)**2.) + (sigma_data**2.))
    c_out = (sigma_data*(sigma-sigma_correct))/(((sigma_data**2.) + (sigma**2.))**0.5)
    c_in = 1./(((sigma**2.)+(sigma_data**2.))**0.5)
    return c_skip.reshape(-1,1,1,1), c_out.reshape(-1,1,1,1), c_in.reshape(-1,1,1,1)

# Get noise level sigma_i based on index i and number of discretization steps k
# Parameters:
# i: index
# k: number of discretization steps
# Returns:
# sigma_i: noise level corresponding to index i
def get_sigma(i, k):
    return (sigma_min**(1./rho) + ((i-1)/(k-1))*(sigma_max**(1./rho)-sigma_min**(1./rho)))**rho

# Get noise level sigma for a continuous index i in [0, 1]
# Follows parameterization in https://openreview.net/pdf?id=FmqFfMTNnv
# Parameters:
# i: continuous index in [0, 1]
# Returns:
# sigma: corresponding noise level
def get_sigma_continuous(i):
    return (sigma_min**(1./rho) + i*(sigma_max**(1./rho)-sigma_min**(1./rho)))**rho


# Add Gaussian noise to input x based on given noise and sigma
# Parameters:
# x: input tensor
# noise: tensor containing Gaussian noise
# sigma: noise level
# Returns:
# x_noisy: x with noise added
def add_noise(x, noise, sigma):
    return x + sigma.reshape(-1,1,1,1)*noise


# Reverse the probability flow ODE by one step
# Parameters:
#   x: input
#   noise: Gaussian noise 
#   sigma: noise level
# Returns:
#   x: x after reversing ODE by one step
def reverse_step(x, noise, sigma):
    return x + ((sigma**2 - sigma_min**2)**0.5)*noise


# Denoise samples at a given noise level
# Parameters:
#   model: consistency model
#   noisy_samples: input noisy samples
#   sigma: noise level
# Returns: 
#   pred_noises: predicted noise
#   pred_samples: denoised samples
def denoise(model, noisy_samples, sigma, latents=None):
    # Denoise samples
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=mixed_precision):
            if latents is not None:
                pred_samples = model(latents, noisy_samples, sigma)
            else:
                pred_samples = model(noisy_samples, sigma)
    # Sample noise
    pred_noises = torch.randn_like(pred_samples)
    return pred_noises, pred_samples

# Reverse the diffusion process to generate samples
# Parameters:
#   model: trained consistency model
#   initial_noise: initial noise to start from 
#   diffusion_steps: number of steps to reverse
# Returns:
#   final_samples: generated samples
def reverse_diffusion(model, initial_noise, diffusion_steps, latents=None):
    next_noisy_samples = initial_noise
    # Reverse process step-by-step
    for k in range(diffusion_steps):

        # Get sigma values
        sigma = get_sigma(diffusion_steps+1-k, diffusion_steps+1)
        next_sigma = get_sigma(diffusion_steps-k, diffusion_steps+1)

        # Denoise 
        noisy_samples = next_noisy_samples
        pred_noises, pred_samples = denoise(model, noisy_samples, sigma, latents)

        # Step to next (lower) noise level
        next_noisy_samples = reverse_step(pred_samples, pred_noises, next_sigma)

    return pred_samples.detach().cpu()


def is_path(variable):
    return isinstance(variable, str) and os.path.exists(variable)



def download_model():
    filepath = os.path.abspath(__file__)
    lib_root = os.path.dirname(filepath)

    if not ospath.exists(lib_root + "/models/music2latent.pt"):
        print("Downloading model...")
        os.makedirs(lib_root + "/models", exist_ok=True)
        _ = hf_hub_download(repo_id="SonyCSLParis/music2latent", filename="music2latent.pt", cache_dir=lib_root + "/models", local_dir=lib_root + "/models")
        print("Model was downloaded successfully!")