import argparse
import lpips
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

# Ορισμός των διαδρομών για τις εικόνες
#img_path_0 = r"C:\Users\steli\DIPLOMA\images\yeezyCARBON.png"
#img_path_1 = r"C:\Users\steli\DIPLOMA\images\yeezyGRANITE.png"
img_path_0 = r"C:\Users\steli\DIPLOMA\bcc\CASE176\0.jpg"
img_path_1 = r"C:\Users\steli\DIPLOMA\bcc\CASE176\2.jpg"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default=img_path_0)
parser.add_argument('-p1','--path1', type=str, default=img_path_1)
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')


opt = parser.parse_args()

# Φορτώνουμε τις εικόνες με χρήση του torch
img0 = lpips.im2tensor(lpips.load_image(img_path_0))
img1 = lpips.im2tensor(lpips.load_image(img_path_1))
img0 = TF.resize(img0, (224, 224))
img1 = TF.resize(img1, (224, 224))

# Initialize the model
loss_fn = lpips.LPIPS(net='alex', version=opt.version) #0.6303163766860962
#loss_fn = lpips.LPIPS(net='vgg', version=opt.version) #0.6341116428375244
#loss_fn = lpips.LPIPS(net='squeeze', version=opt.version) #0.4868687689304352
#loss_fn = lpips.LPIPS(net='alex', version='0.0') #0.3255026340484619
#loss_fn = lpips.LPIPS(net='vgg', version='0.0') #0.1793443113565445
#loss_fn = lpips.LPIPS(net='squeeze', version='0.0') #0.2656324803829193


if(opt.use_gpu):
    loss_fn.cuda()

if(opt.use_gpu):
    img0 = img0.cuda()
    img1 = img1.cuda()

# Compute distance
dist01 = loss_fn.forward(img0, img1)
#print('Distance: %.3f' % dist01)
print(f'Distance: {dist01.item()}')

