ExoGAN (Exoplanets Generative Adversarial Network)

The first DCGAN able to analyse exoplanetary atmospheres.

Inputs:
The input file can be either a python dictionary or a .dat file

Training phase:
The training set is available at this link: https://osf.io/6dxps/
The parameter bounds used to generate the training set are shown in Table 1 of Zingales & Waldmann, 2018 (https://arxiv.org/abs/1806.02906)

Prediction:
The input file for a prediction is usually a .dat file with three columns:
1) wavelength
2) spectrum
3) error bars

For a comparison with a known set of data it is possible to use as optional input the .par file
An example of input spectrum and parameters file are provided within this repository.

