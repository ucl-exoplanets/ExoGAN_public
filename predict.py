# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT
# 
# [2017-11-02] Modifications for Exoplanetary science
# contributors: Tiziano Zingales (1, 2), Ingo Waldmann (1)
#   + License: (1) UCL, (2) INAF/OaPa


import matplotlib
matplotlib.use('pdf')
font = {'size'   : 22}

matplotlib.rc('font', **font)

import argparse
import sys
import copy
import os
import glob
import scipy as sp
import numpy as np
import tensorflow as tf
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import cPickle as pickle
import time
from util import *
from ops import *
import multiprocessing
from multiprocessing import Pool
from model import DCGAN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.set_printoptions(threshold='nan')





if __name__ == '__main__':

   
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod', type=str, default='predict')
    parser.add_argument('--errorbar', type=float, default=0)
    parser.add_argument('--spec_num', type=int, default=0)
    parser.add_argument('--approach', type=str,
                        choices=['adam', 'hmc'],
                        default='adam')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--hmcBeta', type=float, default=0.2)
    parser.add_argument('--hmcEps', type=float, default=0.001)
    parser.add_argument('--hmcL', type=int, default=100)
    parser.add_argument('--hmcAnneal', type=float, default=1)
    parser.add_argument('--nIter', type=int, default=1001)
    parser.add_argument('--imgSize', type=int, default=33)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--checkpointDir', type=str, default='checkpoint_test')
    parser.add_argument('--outDir', type=str, default='exogan_output/')
    parser.add_argument('--outInterval', type=int, default=50)
    parser.add_argument('--maskType', type=str,
                        choices=['random', 'center', 'left', 'full', 
                                 'grid', 'lowres', 'parameters', 'wfc3'],
                        default='parameters')
    parser.add_argument('--input_spectrum', type=str, default='./input_spectrum.dat')
    parser.add_argument('--centerScale', type=float, default=0.25)
    parser.add_argument('--make_corner', type=bool, default=False)
    parser.add_argument('--spectra_int_norm', type=bool, default=False)
    parser.add_argument('--spectra_norm', type=bool, default=False)
    parser.add_argument('--spectra_real_norm', type=bool, default=True)
    
    args = parser.parse_args()
    assert(os.path.exists(args.checkpointDir))
    tf.reset_default_graph()
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    spectrum = args.input_spectrum
    
    dcgan = DCGAN(sess, 
                  image_size=args.imgSize,
                  z_dim=100,
                  batch_size=64,
                  checkpoint_dir=args.checkpointDir, 
                  c_dim=1,
                  lam=args.lam)
    dcgan.complete(args, spectrum)
