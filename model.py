import scipy as sp
import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import cPickle as pickle
import time
from util import *
from ops import *
from corner import *



def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, image_size=64, is_crop=False,
               batch_size=64, sample_size=64, lowres=8,
               z_dim=100, gf_dim=64, df_dim=64,
               gfc_dim=1024, dfc_dim=1024, c_dim=3,
               checkpoint_dir=None, lam=0.1):
    """
    Args:
        sess: TensorFlow session
        batch_size: The size of batch. Should be specified before training.
        lowres: (optional) Low resolution image/mask shrink factor. [8]
        z_dim: (optional) Dimension of dim for Z. [100]
        gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
        df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
        gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
        dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
        c_dim: (optional) Dimension of image color. [3]
    """
    # Currently, image size must be a (power of 2) and (8 or higher).
    ## assert (image_size & (image_size - 1) == 0 and image_size >= 8)

    self.sess = sess
    self.is_crop = is_crop
    self.batch_size = batch_size
    self.image_size = image_size
    self.sample_size = sample_size
    self.image_shape = [image_size, image_size, c_dim]


    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.lam = lam

    self.c_dim = c_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bns = [
      batch_norm(name='d_bn{}'.format(i, )) for i in range(4)]

    log_size = int(math.log(image_size) / math.log(2))
    self.g_bns = [
      batch_norm(name='g_bn{}'.format(i, )) for i in range(log_size)]

    self.checkpoint_dir = checkpoint_dir
    self.build_model()

    self.model_name = "DCGAN.model"

  def build_model(self):
    self.is_training = tf.placeholder(tf.bool, name='is_training')
    self.images = tf.placeholder(
      tf.float32, [None] + self.image_shape, name='real_images')
    
    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    self.z_sum = tf.summary.histogram("z", self.z)

    self.G = self.generator(self.z)
    
    self.D, self.D_logits = self.discriminator(self.images)

    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    self.d_sum = tf.summary.histogram("d", self.D)
    self.d__sum = tf.summary.histogram("d_", self.D_)
    self.G_sum = tf.summary.image("G", self.G)

    self.d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                              labels=tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                              labels=tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                              labels=tf.ones_like(self.D_)))

    self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
    self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver(max_to_keep=1)

    # Completion.
    self.mask = tf.placeholder(tf.float32, self.image_shape, name='mask')
    self.contextual_loss = tf.reduce_sum(
      tf.contrib.layers.flatten(
        tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images))), 1)
    self.perceptual_loss = self.g_loss
    self.complete_loss = self.contextual_loss + self.lam * self.perceptual_loss
    self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

  def train(self, config, X):
    data = X
    np.random.shuffle(data)
    assert (len(data) > 0)

    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
      .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
      .minimize(self.g_loss, var_list=self.g_vars)
    with self.sess.as_default():
      try:
        tf.global_variables_initializer().run()
      except:
        tf.initialize_all_variables().run()
    self.g_sum = tf.summary.merge(
      [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = tf.summary.merge(
      [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
    

    sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
    sample_files = data[0:self.sample_size]

    sample = [get_spectral_matrix(sample_file, size=self.image_size - 10) for sample_file in sample_files]
    sample_images = np.array(sample).astype(np.float32)
    counter = 1
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print("""
            ======
            An existing model was found in the checkpoint directory.
            If you just cloned this repository, it's a model for exoplanetary spectra 
            creating repackagint 10 millions spectra in groups of 10 thousands.
            If you want to train a new model from scratch,
            delete the checkpoint directory or specify a different
            --checkpoint_dir argument.
            ======
            """)
    else:
      print("""
            ======
            An existing model was not found in the checkpoint directory.
            Initializing a new one.
            ======
            """)

    for epoch in xrange(config.epoch):
      data = X
      batch_idxs = min(len(data), config.train_size) // self.batch_size

      for idx in xrange(0, batch_idxs):
        batch_files = data[idx * config.batch_size:(idx + 1) * config.batch_size]
        batch = [get_spectral_matrix(batch_file, size=self.image_size - 10)
                 for batch_file in batch_files]
        batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
          .astype(np.float32)

        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
                                       feed_dict={self.images: batch_images, self.z: batch_z, self.is_training: True})
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
                                       feed_dict={self.z: batch_z, self.is_training: True})
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
                                       feed_dict={self.z: batch_z, self.is_training: True})
        self.writer.add_summary(summary_str, counter)
        with self.sess.as_default():
          errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
          errD_real = self.d_loss_real.eval({self.images: batch_images, self.is_training: False})
          errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})

        counter += 1
        print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
          epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

#        if np.mod(counter, 1000) == 1:
#          samples, d_loss, g_loss = self.sess.run(
#            [self.G, self.d_loss, self.g_loss],
#            feed_dict={self.z: sample_z, self.images: sample_images, self.is_training: False}
#          )
#          save_images(samples, [8, 8],
#                      './samples/train_{:02d}_{:04d}.pdf'.format(epoch, idx))
#          print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

        if np.mod(counter, 1000) == 2:
          self.save(config.checkpoint_dir, counter)

  def complete(self, config, X, path="", sigma=0.0):
    """
    Finds the best representation that can complete any missing 
    part of the ASPA code.

    Input: any spectrum correctly converted into an ASPA code    
    """
    
    if type(X) == dict:
      X_to_split = copy.deepcopy(X)
    elif type(X) == str:
      if X[-3:] == 'dat':
        X_to_split = str(X)
    else:
      X_to_split = np.array(X)
      
    try:    
      if type(X) != np.ndarray:
        true_spectrum = X
      else:
        true_spectrum = None
    except IOError:
      true_spectrum = None
    
    build_directories(config)
    
    with self.sess.as_default():
      try:
        tf.global_variables_initializer().run()
      except:
        tf.initialize_all_variables().run()

    isLoaded = self.load(self.checkpoint_dir)
    assert (isLoaded)
    
    wnw_grid = np.genfromtxt('wnw_grid.txt')
    

    nImgs = self.batch_size

    batch_idxs = int(np.ceil(nImgs / self.batch_size))
    if config.maskType == 'random':
      fraction_masked = 0.2
      mask = np.ones(self.image_shape)
      mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
    elif config.maskType == 'center':
      assert (config.centerScale <= 0.5)
      mask = np.ones(self.image_shape)
      l = int(self.image_size * config.centerScale)
      u = int(self.image_size * (1.0 - config.centerScale))
      mask[l:u, l:u, :] = 0.0
    elif config.maskType == 'left':
      mask = np.ones(self.image_shape)
      c = self.image_size // 2
      mask[:, :c, :] = 0.0
    elif config.maskType == 'full':
      mask = np.ones(self.image_shape)
    elif config.maskType == 'grid':
      mask = np.zeros(self.image_shape)
      mask[::4, ::4, :] = 1.0
    elif config.maskType == 'lowres':
      mask = np.zeros(self.image_shape)
    elif config.maskType == 'parameters':
      assert (config.centerScale <= 0.5)
      mask = np.ones(self.image_shape)
      mask[-3:, :, :] = 0.0
      mask[:, -3:, :] = 0.0
      mask[-10:, -10:, :] = 0.0
    elif config.maskType == 'wfc3':
      assert (config.centerScale <= 0.5)
      m_size = self.image_size - 10
      mask = np.ones(self.image_shape)
      fake_spec = np.ones(m_size**2)
      fake_spec[:334] = 0.0
      fake_spec[384:] = 0.0
      fake_spec = fake_spec.reshape((m_size, m_size))
      mask[:m_size, :m_size, 0] = fake_spec
      mask[-8:, :, :] = 0.0
      mask[:, -10:, :] = 0.0
      mask[-10:, -10:, :] = 0.0
    else:
      assert (False)

    for idx in xrange(0, batch_idxs):
      l = idx * self.batch_size
      u = min((idx + 1) * self.batch_size, nImgs)
      batchSz = u - l
      if type(X) != str:
        Xtrue = get_spectral_matrix(X, size=self.image_size - 10)
        Xt = get_test_image(X, sigma=sigma, size=self.image_size, batch_size=self.batch_size)
      else:
        Xtrue = get_spectral_matrix(X, parfile=X[:-3]+'par', size=self.image_size - 10)
        Xt = get_test_image(X, sigma=sigma, size=self.image_size, batch_size=self.batch_size, parfile=X[:-3]+'par')
      spec_parameters = get_parameters(Xtrue, size=self.image_size)
      
      batch = Xt
      batch_images = np.array(batch).astype(np.float32)
      if batchSz < self.batch_size:
        print(batchSz)
        padSz = ((0, int(self.batch_size - batchSz)), (0, 0), (0, 0), (0, 0))
        batch_images = np.pad(batch_images, padSz, 'constant')
        
        batch_images = batch_images.astype(np.float32)
       

      zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
      m = 0
      v = 0
      
      nImgs = 1
      nRows = int(np.sqrt(nImgs))
      nCols = int(np.sqrt(nImgs))
#      save_images(batch_images[:nImgs, :, :, :], [nRows, nCols],
#                  os.path.join(config.outDir, 'before.pdf'))
      plt.imsave(os.path.join(config.outDir, 'before.png'), Xtrue[:, :, 0], cmap='gist_gray', format='png')
      plt.close()
      resize(os.path.join(config.outDir, 'before.png'))
      
      masked_images = np.multiply(batch_images, mask)
#      save_images(masked_images[:nImgs, :, :, :], [nRows, nCols],
#                  os.path.join(config.outDir, 'masked.pdf'))
      plt.imsave(os.path.join(config.outDir, 'masked.png'), masked_images[0, :, :, 0], cmap='gist_gray', format='png')
      plt.close()
      resize(os.path.join(config.outDir, 'masked.png'))
      
      for img in range(batchSz):
        with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'a') as f:
          f.write('iter loss ' +
                  ' '.join(['z{}'.format(zi) for zi in range(self.z_dim)]) +
                  '\n')
      
      for i in xrange(config.nIter):
        
        fd = {
          self.z: zhats,
          self.mask: mask,
          self.images: batch_images,
          self.is_training: False
        }
        run = [self.complete_loss, self.grad_complete_loss, self.G]
        loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

        for img in range(batchSz):
          with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
            f.write('{} {} '.format(i, loss[img]).encode())
            np.savetxt(f, zhats[img:img + 1])

        if i % config.outInterval == 0:
          prediction_file = open(config.outDir+'predictions/prediction_{:04d}.txt'.format(i), 'w')
          
          ranges = []
          ground_truths = []
          gan_avg = []
          gan_p_err = []
          gan_m_err = []
          
          parser = SafeConfigParser()
          
          if type(X) == str:
            """
            If the input spectrum is sysnthetic, you know the parameters array 
            and you want to compare the real value with the retrieved one, if
            your spectrum does not contain a molecule, the default value is fixed
            to -7.9
            """
            parser.readfp(open(X[:-3] + 'par', 'rb')) # python 2
                          
            real_tp = getpar(parser, 'Atmosphere', 'tp_iso_temp', 'float' )
            real_rp = getpar(parser, 'Planet', 'radius', 'float')
            real_mp = getpar(parser, 'Planet', 'mass', 'float')
            atm_active_gases = np.array([gas.upper() for gas in getpar(parser, 'Atmosphere','active_gases', 'list-str')])
            atm_active_gases_mixratios = np.array(getpar(parser, 'Atmosphere','active_gases_mixratios', 'list-float'))
            real_mol = check_molecule_existence(['CO', 'CO2', 'H2O', 'CH4'], 
                                                atm_active_gases_mixratios,
                                                atm_active_gases,
                                                default=-7.9)
            ground_truths = np.array(real_mol + [real_rp, real_mp, real_tp])
          
          elif true_spectrum != None and type(X) != str:
            h2o = np.log10(true_spectrum['param']['h2o_mixratio'])
            ch4 = np.log10(true_spectrum['param']['ch4_mixratio'])
            co2 = np.log10(true_spectrum['param']['co2_mixratio'])
            co = np.log10(true_spectrum['param']['co_mixratio'])
            rp = true_spectrum['param']['planet_radius']/RJUP
            mp = true_spectrum['param']['planet_mass']/MJUP
            tp = true_spectrum['param']['temperature_profile']
            ground_truths = np.array([co, co2, h2o, ch4, rp, mp, tp])
            real_mol = np.zeros(4)
          else:
            ground_truths = np.array([None]*7)
            real_mol = np.zeros(4)
            
            
          parameters = ['CO', 'CO2', 'H2O', 'CH4', 'Rp', 'Mp', 'Tp']
          labels = ['$\log{CO}$', '$\log{CO_2}$', '$\log{H_2O}$', 
                    '$\log{CH_4}$', '$R_p (R_j)$', '$M_p (M_j)$', '$T_p$']
          
          all_hists = []
          for mol in parameters:
            prediction_file, gan_avg, gan_p_err, gan_m_err, ranges, all_hists = \
                          histogram_par(mol, G_imgs, batchSz, self.image_size,
                                        ground_truths, all_hists,
                                        prediction_file, gan_avg, 
                                        gan_p_err, gan_m_err, ranges)

          
          all_hists = np.array(all_hists).T
          
          if config.make_corner:
            make_corner_plot(all_hists, ranges, labels, ground_truths, config, i)
          
          """
          Plot histograms
          """
          hist_dict = {}
          f, ax = plt.subplots(2, 4, figsize = (21, 15))
          all_hists = all_hists.T
          ii = 0
          jj = 0
          for his in range(len(all_hists)):
            if his == 4:
              ii = 1
              jj = 4
            hist_dict[labels[his]] = {}
            weights = np.ones_like(all_hists[his])/float(len(all_hists[his]))
            hist_dict[labels[his]]['histogram'] = all_hists[his]
            hist_dict[labels[his]]['weights'] = weights
            hist_dict[labels[his]]['bins'] = ranges[his]
            ax[ii, his-jj].hist(all_hists[his], bins=np.linspace(min(ranges[his]), max(ranges[his]), 20), 
              color='firebrick', weights=weights, normed=0)
#            ax[his].set_ylim(0, 1)
            ax[ii, his-jj].set_xlim(min(ranges[his]), max(ranges[his]))
            ax[ii, his-jj].axvline(gan_avg[his], c='g', label='ExoGAN mean')
            ax[ii, his-jj].axvline(ground_truths[his], c='b', label='Input value')
            ax[ii, his-jj].set_xlabel(labels[his] + \
              ' = $%1.2f_{-%1.2f}^{%1.2f}$' % (gan_avg[his], gan_m_err[his], gan_p_err[his]))
            if his == 3:
              ax[ii, his-jj].legend()
#            ax[his].annotate('$%1.2f_{-%1.2f}^{%1.2f}$' % (gan_avg[his], gan_p_err[his], gan_m_err[his]), 
#               bbox=dict(boxstyle="round4", fc="w", alpha=0.5),
#               xy=(gan_avg[his], max(weights)*(0.9)), 
#               xycoords='data')
            ax[ii, his-jj].axvline(gan_avg[his] + gan_p_err[his], c='k', linestyle='--')
            ax[ii, his-jj].axvline(gan_avg[his] - gan_m_err[his], c='k', linestyle='--')
          ax[-1, -1].axis('off')
          plt.subplots_adjust(right=1.2)
          
          histName = os.path.join(config.outDir,
                                   'histograms/all_par/{:04d}.pdf'.format(i))
          plt.savefig(histName, bbox_inches='tight')
          plt.close()
          histpickle = os.path.join(config.outDir,
                                   'histograms/all_par/histogram.pickle')
          with open(histpickle,'wb') as fp:
            pickle.dump(hist_dict,fp)
            
          real_spec = Xtrue[:self.image_size, :self.image_size, :]
          real_spec = real_spec[:23, :23, 0].flatten()
          
          chi_square = []
          spectra = []
          f, ax = plt.subplots(sharey=True, figsize=(12, 6))
          for k in range(batchSz):
            spectrum = G_imgs[k, :self.image_size, :self.image_size, :]
            spectrum = spectrum[:23, :23, 0].flatten()
            spectra.append(spectrum)
            chi_square.append(chisquare(spectrum[:440], f_exp=real_spec[:440])[0])
          best_ind = chi_square.index(min(chi_square))
          
          
          print(i, np.mean(loss[0:batchSz]))
          imgName = os.path.join(config.outDir,
                                 'hats_imgs/{:04d}.png'.format(i))
          
#          save_images(G_imgs[:nImgs, :, :, :], [nRows, nCols], imgName)
          plt.imsave(imgName, G_imgs[best_ind, :, :, 0], cmap='gist_gray', format='png')
          plt.close()
          resize(imgName)

          inv_masked_hat_images = np.multiply(G_imgs, 1.0 - mask)
          completed = masked_images + inv_masked_hat_images
          imgName = os.path.join(config.outDir,
                                 'completed/{:04d}.png'.format(i))
#          save_images(completed[:nImgs, :, :, :], [nRows, nCols], imgName)
          plt.imsave(imgName, completed[best_ind, :, :, 0], cmap='gist_gray', format='png')
          plt.close()
          resize(imgName)
        
          
          
          
          if config.spectra_int_norm:
            # Compared real spectrum with the generated one
            spectra_int_norm(Xtrue, self.image_size, wnw_grid, 
                             batchSz, G_imgs, config, i)
            
          if config.spectra_norm:
            # Compare spectra with original normalisation between 0 and 1
            spectra_norm(Xtrue, self.image_size, wnw_grid, 
                             batchSz, G_imgs, config, i)
          
          if config.spectra_real_norm:
            # Compare spectra with the normalisation factor from the real spectrum
            spectra_real_norm(Xtrue, self.image_size, wnw_grid, 
                             batchSz, G_imgs, config, i)
            
      
          
          
        if config.approach == 'adam':
          # Optimize single completion with Adam
          m_prev = np.copy(m)
          v_prev = np.copy(v)
          m = config.beta1 * m_prev + (1 - config.beta1) * g[0]
          v = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
          m_hat = m / (1 - config.beta1 ** (i + 1))
          v_hat = v / (1 - config.beta2 ** (i + 1))
          zhats += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
          zhats = np.clip(zhats, -1, 1)

        elif config.approach == 'hmc':
          # Sample example completions with HMC (not in paper)
          zhats_old = np.copy(zhats)
          loss_old = np.copy(loss)
          v = np.random.randn(self.batch_size, self.z_dim)
          v_old = np.copy(v)

          for steps in range(config.hmcL):
            v -= config.hmcEps / 2 * config.hmcBeta * g[0]
            zhats += config.hmcEps * v
            np.copyto(zhats, np.clip(zhats, -1, 1))
            loss, g, _, _ = self.sess.run(run, feed_dict=fd)
            v -= config.hmcEps / 2 * config.hmcBeta * g[0]

          for img in range(batchSz):
            logprob_old = config.hmcBeta * loss_old[img] + np.sum(v_old[img] ** 2) / 2
            logprob = config.hmcBeta * loss[img] + np.sum(v[img] ** 2) / 2
            accept = np.exp(logprob_old - logprob)
            if accept < 1 and np.random.uniform() > accept:
              np.copyto(zhats[img], zhats_old[img])

          config.hmcBeta *= config.hmcAnneal

        else:
          assert (False)

  def discriminator(self, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      # TODO: Investigate how to parameterise discriminator based off image size.
      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim * 2, name='d_h1_conv'), self.is_training))
      h2 = lrelu(self.d_bns[1](conv2d(h1, self.df_dim * 4, name='d_h2_conv'), self.is_training))
      h3 = lrelu(self.d_bns[2](conv2d(h2, self.df_dim * 8, name='d_h3_conv'), self.is_training))
      h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')
      
      
      return tf.nn.sigmoid(h4), h4
  
  def generator(self, z):
    with tf.variable_scope("generator") as scope:
      s_h, s_w = self.image_size, self.image_size
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
      
      hs = [None]
      hs[0] = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
      hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))
      
      hs.append(None)
      hs[1], _, _ = conv2d_transpose(hs[0], [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
      hs[1] = tf.nn.relu(self.g_bns[1](hs[1], self.is_training))
      
      hs.append(None)
      hs[2], _, _ = conv2d_transpose(hs[1], [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
      hs[2] = tf.nn.relu(self.g_bns[2](hs[2], self.is_training))
            
      hs.append(None)
      hs[3], _, _ = conv2d_transpose(hs[2], [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
      hs[3] = tf.nn.relu(self.g_bns[3](hs[3], self.is_training))
      
      hs.append(None)
      hs[4], _, _ = conv2d_transpose(hs[3], [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
      
      # for normalisations between 0 and 1 use 'tf.nn.sigmoid(hs[4])'
      
      return tf.nn.sigmoid(hs[4])

  def save(self, checkpoint_dir, step):
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, self.model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      return True
    else:
      return False
