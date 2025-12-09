import sys
sys.path.append(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master')

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from DISTS_tensorflow import DISTS  # Κάνουμε import την κλάση DISTS

# Απενεργοποιούμε το eager execution στο TensorFlow 1.x
tf.disable_eager_execution()

# Δημιουργία ενός αντικειμένου DISTS
model = DISTS()

# Διαδρομές για τις εικόνες που θέλουμε να συγκρίνουμε
ref_image_path = r"C:\Users\steli\DIPLOMA\bcc\CASE001\0.jpg"
dist_image_path = r"C:\Users\steli\DIPLOMA\bcc\CASE001\1.jpg"
#"C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\images\r1.png"

# Φόρτωση και προετοιμασία των εικόνων
ref = Image.open(ref_image_path).convert("RGB").resize((224, 224))
dist = Image.open(dist_image_path).convert("RGB").resize((224, 224))

# Μετατροπή σε numpy array και κανονικοποίηση
ref = np.expand_dims(np.array(ref) / 255.0, axis=0)
dist = np.expand_dims(np.array(dist) / 255.0, axis=0)

# Υπολογισμός της απόστασης μεταξύ των δύο εικόνων
x = tf.placeholder(dtype=tf.float32, shape=ref.shape, name="ref")
y = tf.placeholder(dtype=tf.float32, shape=dist.shape, name="dist")
score = model.get_score(x, y)

# Δημιουργία TensorFlow συνεδρίας και εκτέλεση υπολογισμού
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    distance = sess.run(score, feed_dict={x: ref, y: dist})
    print(f"DISTS distance (TensorFlow): {distance}")
