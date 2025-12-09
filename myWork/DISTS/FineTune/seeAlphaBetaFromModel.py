import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as scio

tf.disable_eager_execution()

class DISTS():
    def __init__(self):
        # Φόρτωση των παραμέτρων από το αποθηκευμένο μοντέλο
        self.parameters = scio.loadmat(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\weights\net_param.mat')
        self.chns = [3, 64, 128, 256, 512, 512]
        self.mean = tf.constant(self.parameters['vgg_mean'], dtype=tf.float32, shape=(1, 1, 1, 3), name="img_mean")
        self.std = tf.constant(self.parameters['vgg_std'], dtype=tf.float32, shape=(1, 1, 1, 3), name="img_std")
        # Φόρτωση του καλύτερου μοντέλου alpha και beta
        #self.weights = scio.loadmat(r"D:\Diploma\DISTS_FINE_TUNE\Mat_files_similar_imgs\epoch_10_alpha_beta.mat")
        self.weights = scio.loadmat(r"D:\Diploma\DISTS_FINE_TUNE\SecondFineTune\Mat_files_second_training\best_model_alpha_beta.mat")

        self.alpha = tf.Variable(np.reshape(self.weights['alpha'], (1, 1, 1, sum(self.chns))), dtype=tf.float32, name="alpha", trainable=False)
        self.beta = tf.Variable(np.reshape(self.weights['beta'], (1, 1, 1, sum(self.chns))), dtype=tf.float32, name="beta", trainable=False)

# Συνάρτηση για να φορτώνει τις τιμές alpha και beta και να τις τυπώνει
def print_alpha_beta():
    model = DISTS()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        alpha_values, beta_values = sess.run([model.alpha, model.beta])

        print("Alpha values:")
        print(alpha_values)

        print("\nBeta values:")
        print(beta_values)

# Κλήση της συνάρτησης για εμφάνιση των τιμών alpha και beta
print_alpha_beta()
