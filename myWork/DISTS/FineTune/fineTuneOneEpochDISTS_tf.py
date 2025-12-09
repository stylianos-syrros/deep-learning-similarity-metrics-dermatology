import tensorflow.compat.v1 as tf
import numpy as np
import os
from PIL import Image
import scipy.io as scio
import glob
import logging
import pandas as pd 

tf.disable_eager_execution()

def check_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return "Running on GPU"
    else:
        return "Running on CPU"

print(check_device())

# Φορτώστε το αρχείο Excel
scores_df = pd.read_excel(r"C:\Users\steli\DIPLOMA\bcc\SCARS-SCORES-IN-DETAIL.xlsx")

# Αφαίρεση του 'CASE' και μετατροπή σε ακέραιο
scores_df['CASES'] = scores_df['CASES'].str.replace('CASE', '').astype(int)
scores_df.set_index('CASES', inplace=True)

# Επιλογή μόνο των στηλών για το overall score των γιατρών
doctor_scores_columns = ['IB_overall', 'GG_overall', 'MOS_overall', 'SER_overall', 'F_overall', 'H_overall']
doctor_scores = scores_df[doctor_scores_columns]

class DISTS():
    def __init__(self):
        self.parameters = scio.loadmat(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\weights\net_param.mat')
        self.chns = [3,64,128,256,512,512]
        self.mean = tf.constant(self.parameters['vgg_mean'], dtype=tf.float32, shape=(1,1,1,3),name="img_mean")
        self.std = tf.constant(self.parameters['vgg_std'], dtype=tf.float32, shape=(1,1,1,3),name="img_std")
        self.weights = scio.loadmat(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\weights\alpha_beta.mat')
        # Alpha and Beta as trainable variables
        self.alpha = tf.Variable(np.reshape(self.weights['alpha'],(1,1,1,sum(self.chns))), dtype=tf.float32, name="alpha", trainable=True)
        self.beta = tf.Variable(np.reshape(self.weights['beta'],(1,1,1,sum(self.chns))), dtype=tf.float32, name="beta", trainable=True)

    def get_features(self, img):
        x = (img - self.mean)/self.std 
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

        return [img, self.conv1_2,self.conv2_2,self.conv3_3,self.conv4_3,self.conv5_3]

    def conv_layer(self, input, name):
        with tf.variable_scope(name) as _:
            filter = self.get_conv_filter(name)
            conv = tf.nn.conv2d(input, filter, strides=1, padding="SAME")
            bias = self.get_bias(name)
            conv = tf.nn.relu(tf.nn.bias_add(conv, bias))
            return conv

    def pool_layer(self, input, name):
        with tf.variable_scope(name) as _:
            filter = tf.squeeze(tf.constant(self.parameters['L2'+name], name = "filter"),3)
            conv = tf.nn.conv2d(input**2, filter, strides=2, padding=[[0, 0], [1, 0], [1, 0], [0, 0]])
            return tf.sqrt(tf.maximum(conv, 1e-12))

    def get_conv_filter(self, name):
        return tf.constant(self.parameters[name+'_weight'], name = "filter")

    def get_bias(self, name):
        return tf.constant(np.squeeze(self.parameters[name+'_bias']), name = "bias")

    def get_score(self, img1, img2):
        feats0 = self.get_features(img1)
        feats1 = self.get_features(img2)
        dist1 = 0 
        dist2 = 0 
        c1 = 1e-6
        c2 = 1e-6
        w_sum = tf.reduce_sum(self.alpha) + tf.reduce_sum(self.beta)
        alpha = tf.split(self.alpha/w_sum, self.chns, axis=3)
        beta = tf.split(self.beta/w_sum, self.chns, axis=3)
        for k in range(len(self.chns)):
            x_mean = tf.reduce_mean(feats0[k],[1,2], keepdims=True)
            y_mean = tf.reduce_mean(feats1[k],[1,2], keepdims=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+tf.reduce_sum(alpha[k]*S1, 3, keepdims=True)
            x_var = tf.reduce_mean((feats0[k]-x_mean)**2,[1,2], keepdims=True)
            y_var = tf.reduce_mean((feats1[k]-y_mean)**2,[1,2], keepdims=True)
            xy_cov = tf.reduce_mean(feats0[k]*feats1[k],[1,2], keepdims=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+tf.reduce_sum(beta[k]*S2, 3, keepdims=True)

        dist = 1-tf.squeeze(dist1+dist2)
        return dist

def process_case_folder(folder):
    img_paths = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    imgs = [Image.open(img_path).convert('RGB').resize((224, 224)) for img_path in img_paths]
    return [np.array(img) / 255.0 for img in imgs]

def fine_tune_dists_for_one_epoch(train_dir, validation_dir, doctor_scores):
    model = DISTS()
    x = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name="img1")
    y = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name="img2")
    doctor_scores_placeholder = tf.placeholder(dtype=tf.float32, name="doctor_scores")
    distances = model.get_score(x, y)

    # Συνάρτηση απώλειας και optimizer
    loss = tf.reduce_mean(tf.square(distances - doctor_scores_placeholder))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        initial_alpha, initial_beta = sess.run([model.alpha, model.beta])
        print("Starting values of alpha:", initial_alpha)
        print("Starting values of beta:", initial_beta)
        epoch_loss = 0
        case_folders_train = sorted(glob.glob(os.path.join(train_dir, 'CASE*')))
        case_folders_val = sorted(glob.glob(os.path.join(validation_dir, 'CASE*')))

        #for case_folder in sorted(glob.glob(os.path.join(train_dir, 'CASE*'))):
        for case_folder in case_folders_train:
            case_number = int(os.path.basename(case_folder).replace('CASE', ''))
            if case_number in doctor_scores.index:
                imgs = process_case_folder(case_folder)
                img0 = np.expand_dims(imgs[0], axis=0)  # Add batch dimension
                imgs1 = np.expand_dims(imgs[1], axis=0)  # Add batch dimension
                imgs2 = np.expand_dims(imgs[2], axis=0)  # Add batch dimension
                
                mean_dist = np.mean([
                    sess.run(distances, feed_dict={x: img0, y: imgs1}),
                    sess.run(distances, feed_dict={x: img0, y: imgs2})
                ])

                scaled_dist = mean_dist * 6 + 1  # Scale the distance to 1-7
                case_scores = doctor_scores.loc[case_number].values
                mean_doctor_score = np.mean(case_scores)
                loss_value = np.square(scaled_dist - mean_doctor_score)
                epoch_loss += loss_value

                # Update the model weights
                sess.run(optimizer, feed_dict={x: img0, y: imgs1, doctor_scores_placeholder: [mean_doctor_score]})
                sess.run(optimizer, feed_dict={x: img0, y: imgs2, doctor_scores_placeholder: [mean_doctor_score]})

                print(f"Case {case_number}: Mean DISTS: {mean_dist}, Scaled DISTS: {scaled_dist}, Loss: {loss_value}")
                print(f"Doctor Scores: {case_scores}")
                #current_alpha, current_beta = sess.run([model.alpha, model.beta])

        # Validate
        total_validation_loss = 0
        #for case_folder in sorted(glob.glob(os.path.join(validation_dir, 'CASE*'))):
        for case_folder in case_folders_val:
            case_number = int(os.path.basename(case_folder).replace('CASE', ''))
            if case_number in doctor_scores.index:
                imgs = process_case_folder(case_folder)
                img0 = np.expand_dims(imgs[0], axis=0)  # Add batch dimension
                imgs1 = np.expand_dims(imgs[1], axis=0)  # Add batch dimension
                imgs2 = np.expand_dims(imgs[2], axis=0)  # Add batch dimension
                mean_dist = np.mean([
                    sess.run(distances, feed_dict={x: img0, y: imgs1}),
                    sess.run(distances, feed_dict={x: img0, y: imgs2})
                ])
                scaled_dist = mean_dist * 6 + 1  # Scale the distance to 1-7
                case_scores = doctor_scores.loc[case_number].values
                mean_doctor_score = np.mean(case_scores)
                validation_loss = np.square(scaled_dist - mean_doctor_score)  # Compute the squared error
                total_validation_loss += validation_loss

                print(f"Case {case_number}: Mean DISTS: {mean_dist}, Scaled DISTS: {scaled_dist}, Loss: {loss_value}")
                print(f"Doctor Scores: {case_scores}")

        epoch_loss /= len(case_folders_train)
        print(f"Epoch completed. Average Training Loss: {epoch_loss}")

        average_validation_loss = total_validation_loss / len(case_folders_val) 
        print(f"Average Validation Loss: {average_validation_loss}")

        final_alpha, final_beta = sess.run([model.alpha, model.beta])
        print("Final values of alpha:", final_alpha)
        print("Final values of beta:", final_beta)

        # Αποθήκευση σε MAT αρχείο
        scio.savemat(r"D:\Diploma\DISTS_FINE_TUNE\Mat_files\testing_alpha_beta.mat", {'alpha': final_alpha, 'beta': final_beta})

train_dir = r"C:\Users\steli\DIPLOMA\bcc\train"
validation_dir = r"C:\Users\steli\DIPLOMA\bcc\val"
fine_tune_dists_for_one_epoch(train_dir, validation_dir, doctor_scores)
