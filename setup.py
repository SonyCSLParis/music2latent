from setuptools import setup, find_packages

setup(
    name='music2latent',
    version='0.1.1',
    packages=find_packages(), 
    description='Encode and decode audio samples to compressed representations! Useful for generative modelling applications.',
    author='Sony Computer Science Laboratories Paris',
    author_email='music@csl.sony.fr', 
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown', 
    install_requires=[
        'numpy',
        'soundfile',
        'huggingface_hub',
        'torch',
    ],
    license='CC BY-NC 4.0',
    url='https://github.com/SonyCSLParis/music2latent',
    keywords='audio speech music compression generative-model autoencoder diffusion consistency',
)