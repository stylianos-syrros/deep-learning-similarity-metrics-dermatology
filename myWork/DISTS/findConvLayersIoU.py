import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as scio
from PIL import Image

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

    def create_binary_masks(self, feature_maps, threshold=0.2):
        # Δημιουργεί δυαδικές μάσκες για κάθε κανάλι στο feature map
        binary_masks = []
        for fmap in feature_maps:
            # Πάρε το μέσο όρο του feature map και δημιούργησε μια μάσκα
            mean_activation = np.mean(fmap, axis=-1)
            binary_mask = (mean_activation > threshold).astype(np.int32)  # Δυαδική μάσκα
            binary_masks.append(binary_mask)
        return binary_masks

    def process_images(self, img1, img2, img3, session):
        x = tf.placeholder(dtype=tf.float32, shape=img1.shape, name="x")

        # Get features for the images
        feats_img1 = session.run(self.get_features(x), feed_dict={x: img1})
        feats_img2 = session.run(self.get_features(x), feed_dict={x: img2})
        feats_img3 = session.run(self.get_features(x), feed_dict={x: img3})
        
        # Create binary masks for each feature map
        masks_img1 = self.create_binary_masks(feats_img1)
        masks_img2 = self.create_binary_masks(feats_img2)
        masks_img3 = self.create_binary_masks(feats_img3)

        return masks_img1, masks_img2, masks_img3

    def calculate_iou(self, mask1, mask2):
        # Υπολογίζει το IoU μεταξύ δύο δυαδικών μασκών
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0  # Αποφυγή διαίρεσης με το μηδέν
        iou = intersection / union
        return iou

    def process_and_calculate_iou(self, img1, img2, img3, session):
        # Υπολογίζει το IoU μεταξύ των εικόνων του μη φυσιολογικού δέρματος και των δύο φυσιολογικών εικόνων.
        masks_img1, masks_img2, masks_img3 = self.process_images(img1, img2, img3, session)

        iou_scores = []

        # Για κάθε επίπεδο (κανάλι) υπολογίζουμε το IoU
        for i in range(len(masks_img1)):
            iou1 = self.calculate_iou(masks_img1[i], masks_img2[i])
            iou2 = self.calculate_iou(masks_img1[i], masks_img3[i])

            # Παίρνουμε τον μέσο όρο του IoU για τις δύο φυσιολογικές εικόνες
            avg_iou = (iou1 + iou2) / 2
            iou_scores.append(avg_iou)
            print(f"Layer {i+1} - IoU Score: {avg_iou}")

        return iou_scores

# Πώς να χρησιμοποιήσετε τον παραπάνω κώδικα
if __name__ == '__main__':
    # Τροποποίηση για τις συγκεκριμένες εικόνες
    ref_image_path = r"C:\Users\steli\DIPLOMA\bcc\CASE004\0.jpg"
    norm_image1_path = r"C:\Users\steli\DIPLOMA\bcc\CASE004\1.jpg"
    norm_image2_path = r"C:\Users\steli\DIPLOMA\bcc\CASE004\2.jpg"

    model = DISTS()

    # Φόρτωση και προεπεξεργασία των εικόνων
    ref = np.expand_dims(np.array(Image.open(ref_image_path).convert("RGB").resize((224, 224))) / 255.0, axis=0)
    norm1 = np.expand_dims(np.array(Image.open(norm_image1_path).convert("RGB").resize((224, 224))) / 255.0, axis=0)
    norm2 = np.expand_dims(np.array(Image.open(norm_image2_path).convert("RGB").resize((224, 224))) / 255.0, axis=0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Υπολογισμός των δυαδικών μασκών
        masks_img1, masks_img2, masks_img3 = model.process_images(ref, norm1, norm2, sess)
        
        # Εκτυπώστε τις μάσκες για έλεγχο
        for i, (mask1, mask2, mask3) in enumerate(zip(masks_img1, masks_img2, masks_img3)):
            # Εκτύπωση των πρώτων 10 στοιχείων από την πρώτη μάσκα της μη φυσιολογικής εικόνας
            print(f"First 10 elements of the binary mask for the abnormal image from the {i+1} feature map:")
            print(masks_img1[i].flatten()[:10])

            # Εκτύπωση των πρώτων 10 στοιχείων από την πρώτη μάσκα της πρώτης φυσιολογικής εικόνας
            print(f"First 10 elements of the binary mask for the first normal image from the {i+1} feature map:")
            print(masks_img2[i].flatten()[:10])

            # Εκτύπωση των πρώτων 10 στοιχείων από την πρώτη μάσκα της δεύτερης φυσιολογικής εικόνας
            print(f"First 10 elements of the binary mask for the second normal image from the {i+1} feature map:")
            print(masks_img3[i].flatten()[:10])

            # Υπολογισμός του αριθμού των άσσων και των μηδενικών σε κάθε μάσκα
            ones_mask1 = np.sum(masks_img1[i])
            zeros_mask1 = masks_img1[i].size - ones_mask1
            ones_mask2 = np.sum(masks_img2[i])
            zeros_mask2 = masks_img2[i].size - ones_mask2
            ones_mask3 = np.sum(masks_img3[i])
            zeros_mask3 = masks_img3[i].size - ones_mask3

            # Εκτύπωση του αριθμού των άσσων και των μηδενικών
            print(f"Abnormal image mask - Layer {i+1}: Ones: {ones_mask1}, Zeros: {zeros_mask1}")
            print(f"First normal image mask - Layer {i+1}: Ones: {ones_mask2}, Zeros: {zeros_mask2}")
            print(f"Second normal image mask - Layer {i+1}: Ones: {ones_mask3}, Zeros: {zeros_mask3}")

        # Υπολογισμός των IoU scores
        iou_scores = model.process_and_calculate_iou(ref, norm1, norm2, sess)
