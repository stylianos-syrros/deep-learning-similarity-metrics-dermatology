import tensorflow.compat.v1 as tf
import numpy as np
import time
import scipy.io as scio
from PIL import Image
import argparse
import os
import pandas as pd
# tf.enable_eager_execution()
tf.disable_eager_execution()

class DISTS():
    def __init__(self, alpha_beta_path):
        self.parameters = scio.loadmat(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\weights\net_param.mat')
        self.chns = [3,64,128,256,512,512]
        self.mean = tf.constant(self.parameters['vgg_mean'], dtype=tf.float32, shape=(1,1,1,3),name="img_mean")
        self.std = tf.constant(self.parameters['vgg_std'], dtype=tf.float32, shape=(1,1,1,3),name="img_std")
        #self.weights = scio.loadmat(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\weights\alpha_beta.mat')
        self.weights = scio.loadmat(alpha_beta_path)
        self.alpha = tf.constant(np.reshape(self.weights['alpha'],(1,1,1,sum(self.chns))),name="alpha")
        self.beta = tf.constant(np.reshape(self.weights['beta'],(1,1,1,sum(self.chns))),name="beta")

    def get_features(self, img):

        x = (img - self.mean)/self.std # normaliazation of input image
        # pass the image through the convolutional layers
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

        # returns list with the features that got extracted and the input image
        return [img, self.conv1_2,self.conv2_2,self.conv3_3,self.conv4_3,self.conv5_3]

    def conv_layer(self, input, name): # implements a convolutional layer
        with tf.variable_scope(name) as _:
            filter = self.get_conv_filter(name) # takes as input the output of last layer and the name of the current layer 
            conv = tf.nn.conv2d(input, filter, strides=1, padding="SAME")
            bias = self.get_bias(name)
            conv = tf.nn.relu(tf.nn.bias_add(conv, bias)) # applies a ReLU activation function 
            return conv

    def pool_layer(self, input, name): # implements a pooling layer 
        # return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
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

def process_all_cases_for_epochs(base_path, output_file, alpha_beta_dir):
    # Initialize results DataFrame with cases
    results_df = pd.DataFrame({'Case': [f"CASE{str(i).zfill(3)}" for i in range(1, 177)]})

    for epoch in range(1, 26):  # Loop through epochs 1 to 25
        tf.reset_default_graph()  # Clear the default graph
        alpha_beta_path = os.path.join(alpha_beta_dir, f"epoch_{epoch}_alpha_beta.mat")
        model = DISTS(alpha_beta_path)

        # Create a TensorFlow session
        x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="ref")
        y = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="dist")
        score = model.get_score(x, y)
        print(f"Processing epoch: {epoch}")

        avg_distances = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Iterate over all cases
            for case_num in range(1, 177):  # Assuming 176 cases
                case_path = os.path.join(base_path, f"CASE{str(case_num).zfill(3)}")
                img_paths = [os.path.join(case_path, f"{i}.jpg") for i in range(3)]

                # Check if all images exist
                if all(os.path.exists(p) for p in img_paths):
                    # Load and preprocess images
                    img1 = np.expand_dims(np.array(Image.open(img_paths[0]).convert("RGB").resize((224, 224))) / 255.0, axis=0)
                    img2 = np.expand_dims(np.array(Image.open(img_paths[1]).convert("RGB").resize((224, 224))) / 255.0, axis=0)
                    img3 = np.expand_dims(np.array(Image.open(img_paths[2]).convert("RGB").resize((224, 224))) / 255.0, axis=0)

                    # Compute distances
                    distance1 = sess.run(score, feed_dict={x: img1, y: img2})
                    distance2 = sess.run(score, feed_dict={x: img1, y: img3})
                    avg_distance = (distance1 + distance2) / 2

                    # Store the average distance for this case
                    avg_distances.append(avg_distance)

                    print(f"Case {case_num}, Distance1: {distance1}, Distance2: {distance2}, Average Distance: {avg_distance}")

            # Add the average distances to the DataFrame under the appropriate epoch column
            results_df[f'epoch_{epoch}_dists'] = avg_distances

        # Clear the current session to free up GPU memory
        sess.close()

    # Save the complete results DataFrame to Excel
    results_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    # Path to the directory containing the cases
    base_path = r"C:\Users\steli\DIPLOMA\bcc"

    # Path to the directory containing alpha and beta files for each epoch
    alpha_beta_dir = r"D:\Diploma\DISTS_FINE_TUNE\SecondFineTune\Mat_files_second_training"

    # Output file path
    output_file = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\Second_Fine_Tuned_Model_DISTS_tf_all_epochs.xlsx" 

    # Process all cases for all epochs and save to Excel
    process_all_cases_for_epochs(base_path, output_file, alpha_beta_dir)

