import sys
import tensorflow.compat.v1 as tf
import numpy as np
import os
import scipy.io as scio
from PIL import Image
import pandas as pd

# Προσθέτουμε το path στο sys.path για να κάνουμε import το module
sys.path.append(r'C:\Users\steli\DIPLOMA\myProgramms\DISTS\DISTS-master')

# Κάνουμε import το DISTS από το module DISTS_tensorflow
from DISTS_tensorflow import DISTS

# tf.enable_eager_execution()
tf.disable_eager_execution()

def process_all_cases(base_path, output_file):
    # Initialize DISTS model
    model = DISTS()

    # Create a TensorFlow session
    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="ref")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="dist")
    score = model.get_score(x, y)

    # List to store results
    results = []

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

                # Store results
                results.append({
                    'Case': f"CASE{str(case_num).zfill(3)}",
                    'Distance1': distance1,
                    'Distance2': distance2,
                    'Average Distance': avg_distance
                })

                print(f"Case {case_num}, Distance1: {distance1} , Distance2: {distance2} , Average Distance: {avg_distance}")

    # Save results to Excel file
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)

if __name__ == '__main__':
    # Path to the directory containing the cases
    base_path = r"C:\Users\steli\DIPLOMA\bcc"

    # Output file path
    output_file = r"C:\Users\steli\DIPLOMA\myProgramms\XLSX\DISTS_tf.xlsx"

    # Process all cases and save to Excel
    process_all_cases(base_path, output_file)

    print(f"Results saved to {output_file}")
