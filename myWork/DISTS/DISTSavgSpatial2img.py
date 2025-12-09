import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as scio
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# tf.enable_eager_execution()
tf.disable_eager_execution()

class DISTS():
    def __init__(self):
        self.parameters = scio.loadmat(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\weights\net_param.mat')
        self.chns = [3, 64, 128, 256, 512, 512]
        self.mean = tf.constant(self.parameters['vgg_mean'], dtype=tf.float32, shape=(1, 1, 1, 3), name="img_mean")
        self.std = tf.constant(self.parameters['vgg_std'], dtype=tf.float32, shape=(1, 1, 1, 3), name="img_std")
        self.weights = scio.loadmat(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\weights\alpha_beta.mat')
        self.alpha = tf.constant(np.reshape(self.weights['alpha'], (1, 1, 1, sum(self.chns))), name="alpha")
        self.beta = tf.constant(np.reshape(self.weights['beta'], (1, 1, 1, sum(self.chns))), name="beta")

    def get_features(self, img):
        x = (img - self.mean) / self.std
        self.conv1_1 = self.conv_layer(x, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.pool_layer(self.conv1_2, name="pool_1")

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.pool_layer(self.conv2_2, name="pool_2")

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.pool_layer(self.conv3_3, name="pool_3")

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.pool_layer(self.conv4_3, name="pool_4")

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")

        return [img, self.conv1_2, self.conv2_2, self.conv3_3, self.conv4_3, self.conv5_3]

    def conv_layer(self, input, name):
        with tf.variable_scope(name) as _:
            filter = self.get_conv_filter(name)
            conv = tf.nn.conv2d(input, filter, strides=1, padding="SAME")
            bias = self.get_bias(name)
            conv = tf.nn.relu(tf.nn.bias_add(conv, bias))
            return conv

    def pool_layer(self, input, name):
        with tf.variable_scope(name) as _:
            filter = tf.squeeze(tf.constant(self.parameters['L2' + name], name="filter"), 3)
            conv = tf.nn.conv2d(input**2, filter, strides=2, padding=[[0, 0], [1, 0], [1, 0], [0, 0]])
            return tf.sqrt(tf.maximum(conv, 1e-12))

    def get_conv_filter(self, name):
        return tf.constant(self.parameters[name + '_weight'], name="filter")

    def get_bias(self, name):
        return tf.constant(np.squeeze(self.parameters[name + '_bias']), name="bias")

    def calculate_cosine_similarity(self, feature_maps1, feature_maps2):
        assert feature_maps1.shape == feature_maps2.shape, "Τα feature maps πρέπει να έχουν τις ίδιες διαστάσεις"        
        cosine_similarities = []

        # Διατρέχει όλα τα στοιχεία των feature maps
        for i in range(feature_maps1.shape[1]):
            for j in range(feature_maps1.shape[2]):
                # Δημιουργία των διανυσμάτων από τα feature maps
                v1 = feature_maps1[:, i, j]
                v2 = feature_maps2[:, i, j]

                # Υπολογισμός του cosine similarity για το ζεύγος διανυσμάτων
                cos_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

                # Προσθήκη του cosine similarity στη λίστα
                cosine_similarities.append(cos_sim)

        # Υπολογισμός του μέσου όρου των cosine similarities
        avg_cos_sim = np.mean(cosine_similarities)
        
        return avg_cos_sim

    def get_score(self, img1, img2, session):
        feats0 = self.get_features(img1)
        feats1 = self.get_features(img2) 

        # Υπολογισμός cosine similarity για κάθε layer
        cosine_sims = []
        for k in range(len(feats0)):
            # Μετατροπή των χαρακτηριστικών σε numpy arrays
            feat0_np = session.run(feats0[k], feed_dict={img1: ref, img2: dist})  # Παροχή των τιμών μέσω του feed_dict
            feat1_np = session.run(feats1[k], feed_dict={img1: ref, img2: dist})

            # Υπολογισμός cosine similarity
            cos_sim = self.calculate_cosine_similarity(feat0_np, feat1_np)
            print(f"k: {k}, cos_sim: {cos_sim}")
            cosine_sims.append(cos_sim)

        # Υπολογισμός της τελικής μέσης απόστασης (αντί του dist1, dist2)
        avg_distance = 1 - np.mean(cosine_sims)
        
        return avg_distance


if __name__ == '__main__':
    # Τροποποίηση για τις συγκεκριμένες εικόνες
    ref_image_path = r"C:\Users\steli\DIPLOMA\bcc\CASE040\0.jpg"
    dist_image_path = r"C:\Users\steli\DIPLOMA\bcc\CASE040\2.jpg"
    model = DISTS()

    # Αλλαγή του μεγέθους των εικόνων σε 224x224
    ref = Image.open(ref_image_path).convert("RGB").resize((224, 224))
    dist = Image.open(dist_image_path).convert("RGB").resize((224, 224))
    
    # Μετατροπή σε numpy array και κανονικοποίηση
    ref = np.expand_dims(np.array(ref) / 255.0, axis=0)
    dist = np.expand_dims(np.array(dist) / 255.0, axis=0)

    x = tf.placeholder(dtype=tf.float32, shape=ref.shape, name="ref")
    y = tf.placeholder(dtype=tf.float32, shape=dist.shape, name="dist")
    #score = model.get_score(x, y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        distance = model.get_score(x, y, sess)  # Χρήση των ref και dist ως feed_dict
        print(f"distance (TensorFlow): {distance}")

