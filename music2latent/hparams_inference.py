import os

filepath = os.path.abspath(__file__)
lib_root = os.path.dirname(filepath)

load_path_inference_default = os.path.join(lib_root, 'models/music2latent.pt')

max_batch_size_encode = 1                            # maximum inference batch size for encoding: tune it depending on the available GPU memory  
max_waveform_length_encode = 44100*60                # maximum length of waveforms in the batch for encoding: tune it depending on the available GPU memory
max_batch_size_decode = 1                            # maximum inference batch size for decoding: tune it depending on the available GPU memory
max_waveform_length_decode = 44100*60                # maximum length of waveforms in the batch for decoding: tune it depending on the available GPU memory


sigma_rescale = 0.06                                 # rescale sigma for inference