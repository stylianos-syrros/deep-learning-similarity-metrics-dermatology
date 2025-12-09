import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as scio
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd

# tf.enable_eager_execution()
tf.disable_eager_execution()

# Check if a GPU is available
device_name = tf.test.gpu_device_name()
if device_name:
    print(f"Found GPU at: {device_name}")
else:
    print("No GPU found. Using CPU.")

class DISTS():
    def __init__(self):
        self.parameters = scio.loadmat(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\weights\net_param.mat')
        self.chns = [3, 64, 128, 256, 512, 512]
        self.weights = scio.loadmat(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master\weights\alpha_beta.mat')
        self.alpha = tf.constant(np.reshape(self.weights['alpha'], (1, 1, 1, sum(self.chns))), name="alpha")
        self.beta = tf.constant(np.reshape(self.weights['beta'], (1, 1, 1, sum(self.chns))), name="beta")

    def get_features(self, img):
        # Define mean and std within the method to avoid graph-related errors
        mean = tf.constant(self.parameters['vgg_mean'], dtype=tf.float32, shape=(1, 1, 1, 3), name="img_mean")
        std = tf.constant(self.parameters['vgg_std'], dtype=tf.float32, shape=(1, 1, 1, 3), name="img_std")

        x = (img - mean) / std
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

        # Iterate over all feature map elements
        for i in range(feature_maps1.shape[1]):
            for j in range(feature_maps1.shape[2]):
                # Create vectors from the feature maps
                v1 = feature_maps1[:, i, j]
                v2 = feature_maps2[:, i, j]

                # Calculate cosine similarity for the vector pair
                cos_sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

                # Add cosine similarity to the list
                cosine_similarities.append(cos_sim)

        # Calculate the mean of cosine similarities
        avg_cos_sim = np.mean(cosine_similarities)
        
        return avg_cos_sim

    def get_score(self, img1, img2, session, weights, skip_first_layer=False):
        x = tf.placeholder(dtype=tf.float32, shape=img1.shape, name="x")
        y = tf.placeholder(dtype=tf.float32, shape=img2.shape, name="y")

        feats0 = self.get_features(x)
        feats1 = self.get_features(y)

        # Adjust range of layers to be considered
        if skip_first_layer:
            layer_range = range(1, len(feats0))  # Skip the first layer
            used_weights = weights  # Use weights3 with 5 values
        else:
            layer_range = range(len(feats0))
            used_weights = weights  # Use weights1 or weights2 with 6 values

        cosine_sims = []
        for k in layer_range:
            # Convert features to numpy arrays
            feat0_np = session.run(feats0[k], feed_dict={x: img1})  # Provide values through feed_dict
            feat1_np = session.run(feats1[k], feed_dict={y: img2})
            
            # Calculate cosine similarity
            cos_sim = self.calculate_cosine_similarity(feat0_np, feat1_np)
            
            # Add cosine similarity to the list without multiplying by the weight
            cosine_sims.append(cos_sim)
        
        # Calculate the weighted mean of cosine similarities
        weighted_mean_sim = np.sum(np.array(cosine_sims) * np.array(used_weights))
        
        # Calculate the final average distance
        avg_distance = 1 - weighted_mean_sim
        
        return avg_distance

def process_all_cases(base_path, output_file):
    model = DISTS()

    # Define weights for each method
    weights1 = [0.05, 0.1, 0.15, 0.2, 0.25, 0.25]
    weights2 = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
    weights3 = [0.15, 0.15, 0.2, 0.25, 0.25]

    # List to store results
    results = []

    # Iterate over all cases
    for case_num in range(150, 177):  # Assuming 176 cases
        case_path = os.path.join(base_path, f"CASE{str(case_num).zfill(3)}")
        img_paths = [os.path.join(case_path, f"{i}.jpg") for i in range(3)]

        # Print the start of processing for the current case
        print(f"Starting processing for Case: CASE{str(case_num).zfill(3)}")

        # Check if all images exist
        if all(os.path.exists(p) for p in img_paths):
            # Load and preprocess images
            img1 = np.expand_dims(np.array(Image.open(img_paths[0]).convert("RGB").resize((224, 224))) / 255.0, axis=0)
            img2 = np.expand_dims(np.array(Image.open(img_paths[1]).convert("RGB").resize((224, 224))) / 255.0, axis=0)
            img3 = np.expand_dims(np.array(Image.open(img_paths[2]).convert("RGB").resize((224, 224))) / 255.0, axis=0)

            # Create a new session and graph for each case to avoid memory issues
            tf.reset_default_graph()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                # Method 1: Using weights [0.05, 0.1, 0.15, 0.2, 0.25, 0.25]
                distance1_method1 = model.get_score(img1, img2, sess, weights1, skip_first_layer=False)
                distance2_method1 = model.get_score(img1, img3, sess, weights1, skip_first_layer=False)
                avg_distance_method1 = (distance1_method1 + distance2_method1) / 2

                # Method 2: Using weights [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
                distance1_method2 = model.get_score(img1, img2, sess, weights2, skip_first_layer=False)
                distance2_method2 = model.get_score(img1, img3, sess, weights2, skip_first_layer=False)
                avg_distance_method2 = (distance1_method2 + distance2_method2) / 2

                # Method 3: Skip the first layer, use weights3 (which has 5 values)
                distance1_method3 = model.get_score(img1, img2, sess, weights3, skip_first_layer=True)
                distance2_method3 = model.get_score(img1, img3, sess, weights3, skip_first_layer=True)
                avg_distance_method3 = (distance1_method3 + distance2_method3) / 2


                # Print current case and distances for all methods
                print(f"Case: CASE{str(case_num).zfill(3)}")
                print(f"Method 1 - Distance1: {distance1_method1}, Distance2: {distance2_method1}, Average Distance: {avg_distance_method1}")
                print(f"Method 2 - Distance1: {distance1_method2}, Distance2: {distance2_method2}, Average Distance: {avg_distance_method2}")
                print(f"Method 3 - Distance1: {distance1_method3}, Distance2: {distance2_method3}, Average Distance: {avg_distance_method3}")

                # Store results
                results.append({
                    'Case': f"CASE{str(case_num).zfill(3)}",
                    'Method1_Distance1': distance1_method1,
                    'Method1_Distance2': distance2_method1,
                    'Method1_AvgDistance': avg_distance_method1,
                    'Method2_Distance1': distance1_method2,
                    'Method2_Distance2': distance2_method2,
                    'Method2_AvgDistance': avg_distance_method2,
                    'Method3_Distance1': distance1_method3,
                    'Method3_Distance2': distance2_method3,
                    'Method3_AvgDistance': avg_distance_method3
                })

    # Save results to Excel file
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)

if __name__ == '__main__':
    # Path to the directory containing the cases
    base_path = r"C:\Users\steli\DIPLOMA\bcc"

    # Output file path
    output_file = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DISTS_tf_avg_spatial_weights.xlsx"

    # Process all cases and save to Excel
    process_all_cases(base_path, output_file)

    print(f"Results saved to {output_file}")
