import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as scio
from PIL import Image
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gc

# Απενεργοποίηση του Eager Execution
tf.disable_eager_execution()

class DISTS():
    def __init__(self):
        self.parameters = scio.loadmat(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\weights\net_param.mat')
        self.chns = [3, 64, 128, 256, 512, 512]
        self.mean_value = self.parameters['vgg_mean']
        self.std_value = self.parameters['vgg_std']
        self.weights = scio.loadmat(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\weights\alpha_beta.mat')
        self.alpha_value = np.reshape(self.weights['alpha'], (1, 1, 1, sum(self.chns)))
        self.beta_value = np.reshape(self.weights['beta'], (1, 1, 1, sum(self.chns)))

    def get_features(self, img):
        x = (img - self.mean) / self.std
        conv1_1 = self.conv_layer(x, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.pool_layer(conv1_2, name="pool_1")

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.pool_layer(conv2_2, name="pool_2")

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.pool_layer(conv3_3, name="pool_3")

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.pool_layer(conv4_3, name="pool_4")

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")

        return [img, conv1_2, conv2_2, conv3_3, conv4_3, conv5_3]

    def conv_layer(self, input, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            filter = self.get_conv_filter(name)
            conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME")
            bias = self.get_bias(name)
            conv = tf.nn.relu(tf.nn.bias_add(conv, bias))
            return conv

    def pool_layer(self, input, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            filter = tf.squeeze(tf.constant(self.parameters['L2' + name], dtype=tf.float32, name="filter"), 3)
            conv = tf.nn.conv2d(input**2, filter, strides=[1, 2, 2, 1], padding="SAME")
            return tf.sqrt(tf.maximum(conv, 1e-12))

    def get_conv_filter(self, name):
        return tf.constant(self.parameters[name + '_weight'], dtype=tf.float32, name="filter")

    def get_bias(self, name):
        return tf.constant(np.squeeze(self.parameters[name + '_bias']), dtype=tf.float32, name="bias")

    def calculate_cosine_similarity(self, feature_maps1, feature_maps2):
        assert feature_maps1.shape == feature_maps2.shape, "Τα feature maps πρέπει να έχουν τις ίδιες διαστάσεις"
        
        feature_maps1 = feature_maps1[0]
        feature_maps2 = feature_maps2[0]
        cosine_similarities = []

        for i in range(feature_maps1.shape[0]):  
            for j in range(feature_maps1.shape[1]):  
                v1 = feature_maps1[i, j]
                v2 = feature_maps2[i, j]
                cos_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
                cosine_similarities.append(cos_sim)

        avg_cos_sim = np.mean(cosine_similarities)
        return avg_cos_sim

    def get_score(self, img1, img2, session, weights):
        # Create placeholders for images
        x = tf.placeholder(dtype=tf.float32, shape=img1.shape, name="img1")
        y = tf.placeholder(dtype=tf.float32, shape=img2.shape, name="img2")

        # Get features for both images
        self.mean = tf.constant(self.mean_value, dtype=tf.float32, shape=(1, 1, 1, 3), name="img_mean")
        self.std = tf.constant(self.std_value, dtype=tf.float32, shape=(1, 1, 1, 3), name="img_std")

        feats0 = self.get_features(x)
        feats1 = self.get_features(y)

        cosine_sims = []
        for k in range(len(feats0)):
            # Μετατροπή των χαρακτηριστικών σε numpy arrays
            feat0_np = session.run(feats0[k], feed_dict={x: img1})
            feat1_np = session.run(feats1[k], feed_dict={y: img2})
            
            # Υπολογισμός cosine similarity
            cos_sim = self.calculate_cosine_similarity(feat0_np, feat1_np)
            
            # Προσθήκη της cosine similarity στη λίστα χωρίς πολλαπλασιασμό με το βάρος
            cosine_sims.append(cos_sim)
        
        # Υπολογισμός του σταθμισμένου μέσου όρου
        weighted_mean_sim = np.sum(np.array(cosine_sims) * np.array(weights))
        
        # Υπολογισμός της τελικής μέσης απόστασης με βάση τα βάρη
        avg_distance = 1 - weighted_mean_sim
        
        return avg_distance

def process_all_cases(base_path, output_file, log_file):
    # Define weights for both methods
    weights1 = [0.16259184534886537, 0.3108558526442156, 0.3038336238220734, 0.16412432895316986, 0.058594349231675676, 0]
    weights2 = [0.21585427272327534, 0.16563675663580021, 0.4992801282012849, 0.11922884243963962, 0, 0]

    results = []

    # Iterate over all cases
    for case_num in range(53, 177):  # Assuming 176 cases
        case_path = os.path.join(base_path, f"CASE{str(case_num).zfill(3)}")
        img_paths = [os.path.join(case_path, f"{i}.jpg") for i in range(3)]

        print(f"Processing Case: CASE{str(case_num).zfill(3)}")

        if all(os.path.exists(p) for p in img_paths):
            try:
                # Φόρτωση και προετοιμασία των εικόνων
                img1 = np.expand_dims(np.array(Image.open(img_paths[0]).convert("RGB").resize((224, 224))) / 255.0, axis=0)
                img2 = np.expand_dims(np.array(Image.open(img_paths[1]).convert("RGB").resize((224, 224))) / 255.0, axis=0)
                img3 = np.expand_dims(np.array(Image.open(img_paths[2]).convert("RGB").resize((224, 224))) / 255.0, axis=0)

                # Ρύθμιση για σταδιακή κατανάλωση μνήμης GPU
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                # Δημιουργία νέου γραφήματος και session για κάθε περίπτωση
                tf.reset_default_graph()
                model = DISTS()

                with tf.Session(config=config) as sess:
                    sess.run(tf.global_variables_initializer())

                    # Method 1 (weights1)
                    distance1_method1 = model.get_score(img1, img2, sess, weights1)
                    distance2_method1 = model.get_score(img1, img3, sess, weights1)
                    avg_distance_method1 = (distance1_method1 + distance2_method1) / 2

                    # Method 2 (weights2)
                    distance1_method2 = model.get_score(img1, img2, sess, weights2)
                    distance2_method2 = model.get_score(img1, img3, sess, weights2)
                    avg_distance_method2 = (distance1_method2 + distance2_method2) / 2

                # Εκτύπωση αποστάσεων
                print(f"Case {str(case_num).zfill(3)} - Method 1 Distance 1: {distance1_method1}")
                print(f"Case {str(case_num).zfill(3)} - Method 1 Distance 2: {distance2_method1}")
                print(f"Case {str(case_num).zfill(3)} - Method 1 Average Distance: {avg_distance_method1}")
                
                print(f"Case {str(case_num).zfill(3)} - Method 2 Distance 1: {distance1_method2}")
                print(f"Case {str(case_num).zfill(3)} - Method 2 Distance 2: {distance2_method2}")
                print(f"Case {str(case_num).zfill(3)} - Method 2 Average Distance: {avg_distance_method2}")

                # Log the distances to the log file
                with open(log_file, 'a') as log:
                    log.write(f"Case {str(case_num).zfill(3)} - Method 1 Distance 1: {distance1_method1}\n")
                    log.write(f"Case {str(case_num).zfill(3)} - Method 1 Distance 2: {distance2_method1}\n")
                    log.write(f"Case {str(case_num).zfill(3)} - Method 1 Average Distance: {avg_distance_method1}\n")
                    log.write(f"Case {str(case_num).zfill(3)} - Method 2 Distance 1: {distance1_method2}\n")
                    log.write(f"Case {str(case_num).zfill(3)} - Method 2 Distance 2: {distance2_method2}\n")
                    log.write(f"Case {str(case_num).zfill(3)} - Method 2 Average Distance: {avg_distance_method2}\n\n")

                results.append({
                    'Case': f"CASE{str(case_num).zfill(3)}",
                    'Method1_Distance1': distance1_method1,
                    'Method1_Distance2': distance2_method1,
                    'Method1_AvgDistance': avg_distance_method1,
                    'Method2_Distance1': distance1_method2,
                    'Method2_Distance2': distance2_method2,
                    'Method2_AvgDistance': avg_distance_method2
                })

            except tf.errors.ResourceExhaustedError as e:
                print(f"OOM Error encountered at Case {str(case_num).zfill(3)}. Skipping this case.")
                with open(log_file, 'a') as log:
                    log.write(f"Case {str(case_num).zfill(3)} encountered OOM and was skipped.\n\n")
                continue  # Προχωράμε στην επόμενη περίπτωση

            finally:
                # Καθαρισμός μνήμης στο τέλος κάθε επανάληψης
                img1 = None
                img2 = None
                img3 = None
                gc.collect()

    # Save the results after all cases are processed
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)

if __name__ == '__main__':
    base_path = r"C:\Users\steli\DIPLOMA\bcc"
    output_file = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DISTS_tf_avg_spatial_IoU_weights.xlsx"
    log_file = r"C:\Users\steli\DIPLOMA\myProgramms\DISTS\distances_IoU_weights.txt"

    # Δημιουργία του log αρχείου ή εκκαθάριση του αν υπάρχει ήδη
    with open(log_file, 'w') as log:
        log.write("Case Distances Log\n\n")

    process_all_cases(base_path, output_file, log_file)

    print(f"Results saved to {output_file}")
    print(f"Log saved to {log_file}")