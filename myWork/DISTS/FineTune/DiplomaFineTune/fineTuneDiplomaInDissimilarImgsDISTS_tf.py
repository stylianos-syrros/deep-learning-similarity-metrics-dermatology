import tensorflow.compat.v1 as tf
import numpy as np
import os
from PIL import Image
import scipy.io as scio
import glob
import logging
import pandas as pd 

tf.disable_eager_execution()

# Set up logging
logging.basicConfig(filename="D:\\Diploma\\DISTS_FINE_TUNE\\DiplomaFineTune\\diploma_training_log.txt", level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def check_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return "Running on GPU"
    else:
        return "Running on CPU"

print(check_device())

# Load the Excel file with the normalized scores
file_path = r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-MEAN.xlsx"
scores_df = pd.read_excel(file_path)

# Convert 'CASES' to integer after removing the prefix 'CASE'
scores_df['CASES'] = scores_df['CASES'].str.replace('CASE', '').astype(int)
scores_df.set_index('CASES', inplace=True)

# Select only the column 'normalizedScore'
doctor_scores = scores_df['normalizedScore']

class DISTS():
    def __init__(self):
        self.parameters = scio.loadmat(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\weights\net_param.mat')
        self.chns = [3, 64, 128, 256, 512, 512]
        self.mean = tf.constant(self.parameters['vgg_mean'], dtype=tf.float32, shape=(1, 1, 1, 3), name="img_mean")
        self.std = tf.constant(self.parameters['vgg_std'], dtype=tf.float32, shape=(1, 1, 1, 3), name="img_std")
        self.weights = scio.loadmat(r"D:\Diploma\DISTS_FINE_TUNE\Mat_files_similar_imgs\epoch_10_alpha_beta.mat")
        # Alpha and Beta as trainable variables
        self.alpha = tf.Variable(np.reshape(self.weights['alpha'], (1, 1, 1, sum(self.chns))), dtype=tf.float32, name="alpha", trainable=True)
        self.beta = tf.Variable(np.reshape(self.weights['beta'], (1, 1, 1, sum(self.chns))), dtype=tf.float32, name="beta", trainable=True)

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
            filter = tf.squeeze(tf.constant(self.parameters['L2'+name], name="filter"), 3)
            conv = tf.nn.conv2d(input**2, filter, strides=2, padding=[[0, 0], [1, 0], [1, 0], [0, 0]])
            return tf.sqrt(tf.maximum(conv, 1e-12))

    def get_conv_filter(self, name):
        return tf.constant(self.parameters[name + '_weight'], name="filter")

    def get_bias(self, name):
        return tf.constant(np.squeeze(self.parameters[name + '_bias']), name="bias")

    def get_score(self, img1, img2):
        feats0 = self.get_features(img1)
        feats1 = self.get_features(img2)
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = tf.reduce_sum(self.alpha) + tf.reduce_sum(self.beta)
        alpha = tf.split(self.alpha / w_sum, self.chns, axis=3)
        beta = tf.split(self.beta / w_sum, self.chns, axis=3)
        for k in range(len(self.chns)):
            x_mean = tf.reduce_mean(feats0[k], [1, 2], keepdims=True)
            y_mean = tf.reduce_mean(feats1[k], [1, 2], keepdims=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + tf.reduce_sum(alpha[k] * S1, 3, keepdims=True)
            x_var = tf.reduce_mean((feats0[k] - x_mean)**2, [1, 2], keepdims=True)
            y_var = tf.reduce_mean((feats1[k] - y_mean)**2, [1, 2], keepdims=True)
            xy_cov = tf.reduce_mean(feats0[k] * feats1[k], [1, 2], keepdims=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + tf.reduce_sum(beta[k] * S2, 3, keepdims=True)

        dist = 1 - tf.squeeze(dist1 + dist2)
        return dist

def process_case_folder(folder):
    img_paths = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    imgs = [Image.open(img_path).convert('RGB').resize((224, 224)) for img_path in img_paths]
    return [np.array(img) / 255.0 for img in imgs]

def fine_tune_dists_multiple_epochs(train_dir, doctor_scores, epochs=25):
    model = DISTS()
    x = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name="img1")
    y = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name="img2")
    doctor_scores_placeholder = tf.placeholder(dtype=tf.float32, name="doctor_scores")
    distances = model.get_score(x, y)

    # Συνάρτηση απώλειας και optimizer
    loss = tf.reduce_mean(tf.square(distances - doctor_scores_placeholder))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    best_loss = float('inf')
    best_alpha = None
    best_beta = None

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Get initial values of alpha and beta
        initial_alpha, initial_beta = sess.run([model.alpha, model.beta])
        logging.info(f"Initial Alpha: {initial_alpha}, Initial Beta: {initial_beta}")
        print(f"Initial Alpha: {initial_alpha}")
        print(f"Initial Beta: {initial_beta}")

        case_folders_train = sorted(glob.glob(os.path.join(train_dir, 'CASE*')))

        for epoch in range(epochs):
            print(f"Processing epoch {epoch + 1}")
            epoch_loss = 0

            for case_folder in case_folders_train:
                case_number = int(os.path.basename(case_folder).replace('CASE', ''))  # Extract case number

                if case_number in doctor_scores.index:
                    imgs = process_case_folder(case_folder)
                    img0 = np.expand_dims(imgs[0], axis=0)  # Add batch dimension
                    imgs1 = np.expand_dims(imgs[1], axis=0)  # Add batch dimension
                    imgs2 = np.expand_dims(imgs[2], axis=0)  # Add batch dimension

                    mean_dist = np.mean([
                        sess.run(distances, feed_dict={x: img0, y: imgs1}),
                        sess.run(distances, feed_dict={x: img0, y: imgs2})
                    ])

                    # Get the normalized score from the Excel file
                    normalized_score = doctor_scores.loc[case_number]
                    loss_value = np.square(mean_dist - normalized_score)
                    epoch_loss += loss_value

                    # Update the model weights
                    sess.run(optimizer, feed_dict={x: img0, y: imgs1, doctor_scores_placeholder: [normalized_score]})
                    sess.run(optimizer, feed_dict={x: img0, y: imgs2, doctor_scores_placeholder: [normalized_score]})

                    print(f"Case {case_number}: Mean DISTS: {mean_dist}, Normalized Score: {normalized_score}, Loss: {loss_value}")

            # Save epoch data
            current_alpha, current_beta = sess.run([model.alpha, model.beta])
            scio.savemat(f"D:\\Diploma\\DISTS_FINE_TUNE\\DiplomaFineTune\\Mat_files_diploma_training\\epoch_{epoch + 1}_alpha_beta.mat", {'alpha': current_alpha, 'beta': current_beta})

            epoch_loss /= len(case_folders_train)
            logging.info(f"Epoch {epoch + 1}: Training Loss: {epoch_loss}")
            logging.info(f"Alpha: {current_alpha}, Beta: {current_beta}")

            print(f"Epoch {epoch + 1}: Training Loss: {epoch_loss}")
            print(f"Epoch {epoch + 1} values of alpha:", current_alpha)
            print(f"Epoch {epoch + 1} values of beta:", current_beta)

            # Check for best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_alpha, best_beta = current_alpha, current_beta
                # Save best model
                scio.savemat("D:\\Diploma\\DISTS_FINE_TUNE\\DiplomaFineTune\\Mat_files_diploma_training\\best_model_alpha_beta.mat", {'alpha': best_alpha, 'beta': best_beta})
                logging.info("Best model updated")

        # Save the best model
        scio.savemat("D:\\Diploma\\DISTS_FINE_TUNE\\DiplomaFineTune\\Mat_files_diploma_training\\best_model_alpha_beta.mat", {'alpha': best_alpha, 'beta': best_beta})
        print("Best model saved with Epoch Loss:", best_loss)
        logging.info(f"Final Best Epoch Loss: {best_loss}")

train_dir = r"C:\Users\steli\DIPLOMA\bcc\final_train_diploma"
fine_tune_dists_multiple_epochs(train_dir, doctor_scores, epochs=25)

def evaluate_best_model(test_dir, doctor_scores):
    # Φόρτωση του μοντέλου DISTS με τις τιμές alpha και beta από το καλύτερο μοντέλο
    model = DISTS()
    x = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name="img1")
    y = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name="img2")
    doctor_scores_placeholder = tf.placeholder(dtype=tf.float32, name="doctor_scores")
    
    # Υπολογισμός της απόστασης DISTS
    distances = model.get_score(x, y)
    
    # Συνάρτηση απώλειας
    loss = tf.reduce_mean(tf.square(distances - doctor_scores_placeholder))

    case_folders_test = sorted(glob.glob(os.path.join(test_dir, 'CASE*')))
    total_loss = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for case_folder in case_folders_test:
            case_number = int(os.path.basename(case_folder).replace('CASE', ''))
            
            if case_number in doctor_scores.index:
                imgs = process_case_folder(case_folder)
                img0 = np.expand_dims(imgs[0], axis=0)  # Προσθήκη batch διάστασης
                imgs1 = np.expand_dims(imgs[1], axis=0)
                imgs2 = np.expand_dims(imgs[2], axis=0)

                # Υπολογισμός της μέσης τιμής DISTS
                mean_dist = np.mean([
                    sess.run(distances, feed_dict={x: img0, y: imgs1}),
                    sess.run(distances, feed_dict={x: img0, y: imgs2})
                ])

                # Πάρε το normalized score από το αρχείο Excel
                normalized_score = doctor_scores.loc[case_number]
                loss_value = sess.run(loss, feed_dict={x: img0, y: imgs1, doctor_scores_placeholder: normalized_score})
                total_loss += loss_value

                print(f"Case {case_number}: Mean DISTS: {mean_dist}, Normalized Score: {normalized_score}, Loss: {loss_value}")

    # Υπολογισμός του μέσου test loss
    average_test_loss = total_loss / len(case_folders_test)
    print(f"Average Test Loss: {average_test_loss}")
    return average_test_loss

test_dir = r"C:\Users\steli\DIPLOMA\bcc\final_test_diploma"
#evaluate_best_model(test_dir, doctor_scores)