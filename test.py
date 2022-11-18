import argparse
import os
import sys

import numpy as np
import PIL.Image as Image
import skimage.metrics as metrics
import torch
import torch.autograd as autograd
import torch.utils

import model
import utils


parser = argparse.ArgumentParser('cone')
parser.add_argument('--dataset', type=str, default='mit', help='dataset') # 'mit' or 'lsrw'
parser.add_argument('--cuda', default=True, type=bool, help='use cuda or not')
parser.add_argument('--gpu', type=str, default='0', help='gpu id')
parser.add_argument('--save', type=str, default='exp', help='experiment dir')
parser.add_argument('--model', type=str, default='train-20221108-052232', help='target model')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

epochs_dic = {'mit': 500, 'lsrw': 200}
epochs = epochs_dic[args.dataset] # no. of epochs: 500 for mit and 200 for lsrw


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


save = os.path.join(args.save, args.dataset, args.model)
model_path = os.path.join(save, 'models')
image_path = os.path.join(save, 'results')
os.makedirs(image_path, exist_ok=True)
cem = ''
with open(os.path.join(save, 'cem.txt')) as f:
    cem = f.read()


def main():
    if not torch.cuda.is_available():
        print("no gpu device available")
        sys.exit(1)

    network = model.inference(os.path.join(model_path, 'model_%03d.pt' % epochs), cem=cem)
    network = network.cuda()


    test_path = os.path.join('./data', args.dataset, 'test/input')
    test_data = utils.MemoryFriendlyLoader(img_dir=test_path, task='test')

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=True)


    network.eval()
    mean_psnr = 0
    mean_ssim = 0
    with torch.no_grad():
        for _, (input, img_name) in enumerate(test_queue):
            input = autograd.Variable(input).cuda()
            img_name = img_name[0]

            print('processing {}'.format(img_name))

            y = network(input)
            y = y[0].cpu().float().numpy()
            y = np.transpose(y, (1, 2, 0))
            y = np.clip(y * 255.0, 0, 255.0).astype('uint8')

            y_gt = Image.open(os.path.join('./data', args.dataset, 'test/gt', img_name))
            y_gt = y_gt.convert('RGB')
            y_gt = np.asarray(y_gt)

            mean_psnr += metrics.peak_signal_noise_ratio(y, y_gt)
            mean_ssim += metrics.structural_similarity(y, y_gt, multichannel=True)
            # In this study, we set 'multichannel=True' for all the compared
            # methods, which computes the SSIM for each channel independently,
            # then averages them together. A larger (better) SSIM score may be
            # obtained if the image is converted to a grayscale image before
            # computing the SSIM.

            y = Image.fromarray(y)
            y.save(os.path.join(image_path, img_name), 'png')

    n = len(test_queue.dataset)
    mean_psnr /= n
    mean_ssim /= n

    print("psnr %f ssim %f" % (mean_psnr, mean_ssim))


if __name__ == '__main__':
    main()
