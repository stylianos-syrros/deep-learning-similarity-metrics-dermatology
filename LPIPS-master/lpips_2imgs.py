import argparse
import lpips
import torch
import warnings
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='C:\\Users\\steli\\DIPLOMA\\myProgramms\\GUI\\testImages\\img0LPIPS.png')
parser.add_argument('-p1','--path1', type=str, default='C:\\Users\\steli\\DIPLOMA\\myProgramms\\GUI\\testImages\\img1LPIPS.png')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn_alex = lpips.LPIPS(net='alex', version=opt.version)
loss_fn_vgg = lpips.LPIPS(net='vgg', version=opt.version)

if(opt.use_gpu):
    loss_fn_alex.cuda()
    loss_fn_vgg.cuda()

# Load images
img0 = lpips.im2tensor(lpips.load_image(opt.path0)) # RGB image from [-1,1]
img1 = lpips.im2tensor(lpips.load_image(opt.path1))

# Resize images to a common size
img0 = TF.resize(img0, (256, 256))
img1 = TF.resize(img1, (256, 256))

if(opt.use_gpu):
    img0 = img0.cuda()
    img1 = img1.cuda()

# Compute distance
dist_alex = loss_fn_alex.forward(img0, img1)
dist_vgg = loss_fn_vgg.forward(img0, img1)
print('LPIPS Distance (Alex): %.3f'%dist_alex)
print('LPIPS Distance (VGG): %.3f'%dist_vgg)
