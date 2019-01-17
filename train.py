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
  
    # setup gan
    # note: assume square images, so only need 1 dim
    flags = tf.app.flags
    flags.DEFINE_string("mod", 'train', "choose working modality")
    flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
    flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
    flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
    flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
    flags.DEFINE_integer("image_size", 33, "The size of image to use")
    flags.DEFINE_string("training_dir", 'training_set', "Training set directory")
    flags.DEFINE_integer("training_init", 0, "Starting chunk of the training set" )
    flags.DEFINE_string("checkpoint_dir", "checkpoint_test", "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
    FLAGS = flags.FLAGS
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
        
    """
    Load the training set
    """
    
    all_spec = glob(FLAGS.training_dir+'/chunck_*.pkgz')
    X = []
    for i in range(FLAGS.training_init, len(all_spec)):
      s = load(all_spec[i])
      for j in s.keys():
        X.append(s[j])
    X = np.array(X)
    np.random.shuffle(X)
    
    
    tf.reset_default_graph()
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    dcgan = DCGAN(sess,
                  image_size=FLAGS.image_size, 
                  is_crop=False,
                  batch_size=FLAGS.batch_size, 
                  sample_size=64, 
                  lowres=4,
                  z_dim=100, 
                  gf_dim=64, 
                  df_dim=64,
                  gfc_dim=1024, 
                  dfc_dim=1024, 
                  c_dim=1,
                  checkpoint_dir=FLAGS.checkpoint_dir, 
                  lam=0.1)
    dcgan.train(FLAGS, X)