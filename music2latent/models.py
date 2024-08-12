from .utils import *
from .audio import *

import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_init(module):
    if init_as_zero:
        for p in module.parameters():
            p.detach().zero_()
    return module

def upsample_1d(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")

def downsample_1d(x):
    return F.avg_pool1d(x, kernel_size=2, stride=2)

def upsample_2d(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")

def downsample_2d(x):
    return F.avg_pool2d(x, kernel_size=2, stride=2)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.ln = torch.nn.LayerNorm(dim)

    def forward(self, input):
        x = input.permute(0,2,3,1)
        x = self.ln(x)
        x = x.permute(0,3,1,2)
        return x

class FreqGain(nn.Module):
    def __init__(self, freq_dim):
        super(FreqGain, self).__init__()
        self.scale = nn.Parameter(torch.ones((1,1,freq_dim,1)))

    def forward(self, input):
        return input*self.scale
    

class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_2d=False, normalize=False):
        super(UpsampleConv, self).__init__()
        self.normalize = normalize

        self.use_2d = use_2d

        if out_channels is None:
            out_channels = in_channels
        
        if normalize:
            self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels)

        if use_2d:
            self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        else:
            self.c = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')

    def forward(self, x):

        if self.normalize:
            x = self.norm(x)
        
        if self.use_2d:
            x = upsample_2d(x)
        else:
            x = upsample_1d(x)
        x = self.c(x)

        return x
    
class DownsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_2d=False, normalize=False):
        super(DownsampleConv, self).__init__()
        self.normalize = normalize

        if out_channels is None:
            out_channels = in_channels
        
        if normalize:
            self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels)

        if use_2d:
            self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.c = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        
        if self.normalize:
            x = self.norm(x)
        x = self.c(x)

        return x

class UpsampleFreqConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, normalize=False):
        super(UpsampleFreqConv, self).__init__()
        self.normalize = normalize

        if out_channels is None:
            out_channels = in_channels

        if normalize:
            self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels)
        
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=(5,1), stride=1, padding='same')

    def forward(self, x):
        if self.normalize:
            x = self.norm(x)
        x = F.interpolate(x, scale_factor=(4,1), mode="nearest")
        x = self.c(x)
        return x

class DownsampleFreqConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, normalize=False):
        super(DownsampleFreqConv, self).__init__()
        self.normalize = normalize

        if out_channels is None:
            out_channels = in_channels

        if normalize:
            self.norm = nn.GroupNorm(min(in_channels//4, 32), in_channels)

        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=(5,1), stride=(4,1), padding=(2,0))

    def forward(self, x):
        if self.normalize:
            x = self.norm(x)
        x = self.c(x)
        return x
    
class MultiheadAttention(nn.MultiheadAttention):
    def _reset_parameters(self):
        super()._reset_parameters()
        self.out_proj = zero_init(self.out_proj)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, normalize=True, use_2d=False):
        super(Attention, self).__init__()
        
        self.normalize = normalize
        self.use_2d = use_2d
        
        self.mha = MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=0.0, add_zero_attn=False, batch_first=True)
        if normalize:
            self.norm = nn.GroupNorm(min(dim//4, 32), dim)

    def forward(self, x):
        
        inp = x
        
        if self.normalize:
            x = self.norm(x)
        
        if self.use_2d:
            x = x.permute(0,3,2,1) # shape: [bs,len,freq,channels]
            bs,len,freq,channels = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
            x = x.reshape(bs*len,freq,channels) # shape: [bs*len,freq,channels]
        else:
            x = x.permute(0,2,1) # shape: [bs,len,channels]
        
        x = self.mha(x, x, x, need_weights=False)[0]
        
        if self.use_2d:
            x = x.reshape(bs,len,freq,channels).permute(0,3,2,1)
        else:
            x = x.permute(0,2,1)
        x = x+inp

        return x



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels=None, kernel_size=3, downsample=False, upsample=False, normalize=True, leaky=False, attention=False, heads=4, use_2d=False, normalize_residual=False):
        super(ResBlock, self).__init__()
        self.normalize = normalize
        self.attention = attention
        self.upsample = upsample
        self.downsample = downsample
        self.leaky = leaky
        self.kernel_size = kernel_size
        self.normalize_residual = normalize_residual
        self.use_2d = use_2d
        if use_2d:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv1d
        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same')
        self.conv2 = zero_init(Conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same'))
        if in_channels!=out_channels:
            self.res_conv = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.res_conv = nn.Identity()
        if normalize:
            self.norm1 = nn.GroupNorm(min(in_channels//4, 32), in_channels)
            self.norm2 = nn.GroupNorm(min(out_channels//4, 32), out_channels)
        if leaky:
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.activation = nn.SiLU()
        if cond_channels is not None:
            self.proj_emb = zero_init(nn.Linear(cond_channels, out_channels))
        self.dropout = nn.Dropout(dropout_rate)
        if attention:
            self.att = Attention(out_channels, heads, use_2d=use_2d)
            

    def forward(self, x, time_emb=None):
        if not self.normalize_residual:
            y = x.clone()
        if self.normalize:
            x = self.norm1(x)
        if self.normalize_residual:
            y = x.clone()
        x = self.activation(x)
        if self.downsample:
            if self.use_2d:
                x = downsample_2d(x)
                y = downsample_2d(y)
            else:
                x = downsample_1d(x)
                y = downsample_1d(y)
        if self.upsample:
            if self.use_2d:
                x = upsample_2d(x)
                y = upsample_2d(y)
            else:
                x = upsample_1d(x)
                y = upsample_1d(y)
        x = self.conv1(x)
        if time_emb is not None:
            if self.use_2d:
                x = x+self.proj_emb(time_emb)[:,:,None,None]
            else:
                x = x+self.proj_emb(time_emb)[:,:,None]
        if self.normalize:
            x = self.norm2(x)
        x = self.activation(x)
        if x.shape[-1]<=min_res_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        y = self.res_conv(y)
        x = x+y
        if self.attention:
            x = self.att(x)
        return x


# adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/layerspp.py
class GaussianFourierProjection(torch.nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=128, scale=0.02):
    super().__init__()
    self.W = torch.nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2. * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, embedding_size=128, max_positions=10000):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_positions = max_positions

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.embedding_size//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.embedding_size // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        layers_list = layers_list_encoder
        attention_list = attention_list_encoder
        self.layers_list = layers_list
        self.multipliers_list = multipliers_list
        input_channels = base_channels*multipliers_list[0]
        Conv = nn.Conv2d
        self.gain = FreqGain(freq_dim=hop*2)

        channels = data_channels
        self.conv_inp = Conv(channels, input_channels, kernel_size=3, stride=1, padding=1)

        self.freq_dim = (hop*2)//(4**freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim//(2**freq_downsample_list.count(0))
        
        # DOWNSAMPLING
        down_layers = []
        for i, (num_layers,multiplier) in enumerate(zip(layers_list,multipliers_list)):
            output_channels = base_channels*multiplier
            for num in range(num_layers):
                down_layers.append(ResBlock(input_channels, output_channels, normalize=normalization, attention=attention_list[i]==1, heads=heads, use_2d=True))
                input_channels = output_channels
            if i!=(len(layers_list)-1):
                if freq_downsample_list[i]==1:
                    down_layers.append(DownsampleFreqConv(input_channels, normalize=pre_normalize_downsampling_encoder))
                else:
                    down_layers.append(DownsampleConv(input_channels, use_2d=True, normalize=pre_normalize_downsampling_encoder))

        if pre_normalize_2d_to_1d:
            self.prenorm_1d_to_2d = nn.GroupNorm(min(input_channels//4, 32), input_channels)

        bottleneck_layers = []
        output_channels = bottleneck_base_channels
        bottleneck_layers.append(nn.Conv1d(input_channels*self.freq_dim, output_channels, kernel_size=1, stride=1, padding='same'))
        for i in range(num_bottleneck_layers):
            bottleneck_layers.append(ResBlock(output_channels, output_channels, normalize=normalization, use_2d=False))
        self.bottleneck_layers = nn.ModuleList(bottleneck_layers)

        self.norm_out = nn.GroupNorm(min(output_channels//4, 32), output_channels)
        self.activation_out = nn.SiLU()
        self.conv_out = nn.Conv1d(output_channels, bottleneck_channels, kernel_size=1, stride=1, padding='same')
        self.activation_bottleneck = nn.Tanh()
            
        self.down_layers = nn.ModuleList(down_layers)


    def forward(self, x, extract_features=False):

        x = self.conv_inp(x)
        if frequency_scaling:
            x = self.gain(x)
        
        # DOWNSAMPLING
        k = 0
        for i,num_layers in enumerate(self.layers_list):
            for num in range(num_layers):
                x = self.down_layers[k](x)
                k = k+1
            if i!=(len(self.layers_list)-1):
                x = self.down_layers[k](x)
                k = k+1

        if pre_normalize_2d_to_1d:
            x = self.prenorm_1d_to_2d(x)

        x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))
        if extract_features:
            return x

        for layer in self.bottleneck_layers:
            x = layer(x)
                
        x = self.norm_out(x)
        x = self.activation_out(x)
        x = self.conv_out(x)
        x = self.activation_bottleneck(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        layers_list = layers_list_encoder
        attention_list = attention_list_encoder
        self.layers_list = layers_list_encoder
        self.multipliers_list = multipliers_list
        input_channels = base_channels*multipliers_list[-1]

        output_channels = bottleneck_base_channels
        self.conv_inp = nn.Conv1d(bottleneck_channels, output_channels, kernel_size=1, stride=1, padding='same')
        
        self.freq_dim = (hop*2)//(4**freq_downsample_list.count(1))
        self.freq_dim = self.freq_dim//(2**freq_downsample_list.count(0))

        bottleneck_layers = []
        for i in range(num_bottleneck_layers):
            bottleneck_layers.append(ResBlock(output_channels, output_channels, cond_channels, normalize=normalization, use_2d=False))

        self.conv_out_bottleneck = nn.Conv1d(output_channels, input_channels*self.freq_dim, kernel_size=1, stride=1, padding='same')
        self.bottleneck_layers = nn.ModuleList(bottleneck_layers)

        # UPSAMPLING
        multipliers_list_upsampling = list(reversed(multipliers_list))[1:]+list(reversed(multipliers_list))[:1]
        freq_upsample_list = list(reversed(freq_downsample_list))
        up_layers = []      
        for i, (num_layers,multiplier) in enumerate(zip(reversed(layers_list),multipliers_list_upsampling)):
            for num in range(num_layers):
                up_layers.append(ResBlock(input_channels, input_channels, normalize=normalization, attention=list(reversed(attention_list))[i]==1, heads=heads, use_2d=True))
            if i!=(len(layers_list)-1):
                output_channels = base_channels*multiplier
                if freq_upsample_list[i]==1:
                    up_layers.append(UpsampleFreqConv(input_channels, output_channels))
                else:
                    up_layers.append(UpsampleConv(input_channels, output_channels, use_2d=True))
                input_channels = output_channels
            
        self.up_layers = nn.ModuleList(up_layers)


    def forward(self, x):

        x = self.conv_inp(x)
        
        for layer in self.bottleneck_layers:
            x = layer(x)
        x = self.conv_out_bottleneck(x)

        x_ls = torch.chunk(x.unsqueeze(-2), self.freq_dim, -3)
        x = torch.cat(x_ls, -2)
        
        # UPSAMPLING
        k = 0
        pyramid_list = []
        for i,num_layers in enumerate(reversed(self.layers_list)):
            for num in range(num_layers):
                x = self.up_layers[k](x)
                k = k+1
            pyramid_list.append(x)
            if i!=(len(self.layers_list)-1):
                x = self.up_layers[k](x)
                k = k+1

        pyramid_list = pyramid_list[::-1]

        return pyramid_list


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.layers_list = layers_list
        self.multipliers_list = multipliers_list
        input_channels = base_channels*multipliers_list[0]
        Conv = nn.Conv2d

        self.encoder = Encoder()
        self.decoder = Decoder()

        if use_fourier:
            self.emb = GaussianFourierProjection(embedding_size=cond_channels, scale=fourier_scale)
        else:
            self.emb = PositionalEmbedding(embedding_size=cond_channels)

        self.emb_proj = nn.Sequential(nn.Linear(cond_channels, cond_channels), nn.SiLU(), nn.Linear(cond_channels, cond_channels), nn.SiLU())

        self.scale_inp = nn.Sequential(nn.Linear(cond_channels, cond_channels), nn.SiLU(), nn.Linear(cond_channels, cond_channels), nn.SiLU(), zero_init(nn.Linear(cond_channels, hop*2)))
        self.scale_out = nn.Sequential(nn.Linear(cond_channels, cond_channels), nn.SiLU(), nn.Linear(cond_channels, cond_channels), nn.SiLU(), zero_init(nn.Linear(cond_channels, hop*2)))

        self.conv_inp = Conv(data_channels, input_channels, kernel_size=3, stride=1, padding=1)
        
        # DOWNSAMPLING
        down_layers = []
        for i, (num_layers,multiplier) in enumerate(zip(layers_list,multipliers_list)):
            output_channels = base_channels*multiplier
            for num in range(num_layers):
                down_layers.append(Conv(output_channels, output_channels, kernel_size=1, stride=1, padding=0))
                down_layers.append(ResBlock(output_channels, output_channels, cond_channels, normalize=normalization, attention=attention_list[i]==1, heads=heads, use_2d=True))                
                input_channels = output_channels
            if i!=(len(layers_list)-1):
                output_channels = base_channels*multipliers_list[i+1]
                if freq_downsample_list[i]==1:
                    down_layers.append(DownsampleFreqConv(input_channels, output_channels))
                else:
                    down_layers.append(DownsampleConv(input_channels, output_channels, use_2d=True))

        # UPSAMPLING
        multipliers_list_upsampling = list(reversed(multipliers_list))[1:]+list(reversed(multipliers_list))[:1]
        freq_upsample_list = list(reversed(freq_downsample_list))
        up_layers = []      
        for i, (num_layers,multiplier) in enumerate(zip(reversed(layers_list),multipliers_list_upsampling)):
            for num in range(num_layers):
                up_layers.append(Conv(input_channels, input_channels, kernel_size=1, stride=1, padding=0))
                up_layers.append(ResBlock(input_channels, input_channels, cond_channels, normalize=normalization, attention=list(reversed(attention_list))[i]==1, heads=heads, use_2d=True))
            if i!=(len(layers_list)-1):
                output_channels = base_channels*multiplier
                if freq_upsample_list[i]==1:
                    up_layers.append(UpsampleFreqConv(input_channels, output_channels))
                else:
                    up_layers.append(UpsampleConv(input_channels, output_channels, use_2d=True))
                input_channels = output_channels
                
        self.conv_decoded = Conv(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.norm_out = nn.GroupNorm(min(input_channels//4, 32), input_channels)
        self.activation_out = nn.SiLU()
        self.conv_out = zero_init(Conv(input_channels, data_channels, kernel_size=3, stride=1, padding=1))
            
        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)


    def forward(self, latents, x, sigma=None, pyramid_latents=None):

        if sigma is None:
            sigma = sigma_max
        
        inp = x
        
        # CONDITIONING
        sigma = torch.ones((x.shape[0],), dtype=torch.float32).to(x.device)*sigma
        sigma_log = torch.log(sigma)/4.
        emb_sigma_log = self.emb(sigma_log)
        time_emb = self.emb_proj(emb_sigma_log)

        scale_w_inp = self.scale_inp(emb_sigma_log).reshape(x.shape[0],1,-1,1)
        scale_w_out = self.scale_out(emb_sigma_log).reshape(x.shape[0],1,-1,1)
            
        c_skip, c_out, c_in = get_c(sigma)
        
        x = c_in*x

        if latents.shape == x.shape:
            latents = self.encoder(latents)

        if pyramid_latents is None:
            pyramid_latents = self.decoder(latents)

        x = self.conv_inp(x)
        if frequency_scaling:
            x = (1.+scale_w_inp)*x
        
        skip_list = []
        
        # DOWNSAMPLING
        k = 0
        r = 0
        for i,num_layers in enumerate(self.layers_list):
            for num in range(num_layers):
                d = self.down_layers[k](pyramid_latents[i])
                k = k+1
                x = (x+d)/np.sqrt(2.)
                x = self.down_layers[k](x, time_emb)
                skip_list.append(x)
                k = k+1
            if i!=(len(self.layers_list)-1):
                x = self.down_layers[k](x)
                k = k+1
              
        # UPSAMPLING
        k = 0
        for i,num_layers in enumerate(reversed(self.layers_list)):
            for num in range(num_layers):
                d = self.up_layers[k](pyramid_latents[-i-1])
                k = k+1
                x = (x+skip_list.pop()+d)/np.sqrt(3.)
                x = self.up_layers[k](x, time_emb)
                k = k+1
            if i!=(len(self.layers_list)-1):
                x = self.up_layers[k](x)
                k = k+1
                
        d = self.conv_decoded(pyramid_latents[0])
        x = (x+d)/np.sqrt(2.)

        x = self.norm_out(x)
        x = self.activation_out(x)
        if frequency_scaling:
            x = (1.+scale_w_out)*x
        x = self.conv_out(x)

        out = c_skip*inp + c_out*x 

        return out