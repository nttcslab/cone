import argparse
import glob
import os
import sys
import time

import numpy as np
import PIL.Image as Image
import skimage.metrics as metrics
import torch
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.nn as nn

import model
import utils


parser = argparse.ArgumentParser('cone')
parser.add_argument('--dataset', type=str, default='mit', help='dataset') # 'mit' or 'lsrw'
parser.add_argument('--cuda', default=True, type=bool, help='use cuda or not')
parser.add_argument('--gpu', type=str, default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=2, help='for reproducibility')
parser.add_argument('--save', type=str, default='exp', help='experiment dir')

parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--lr_iem', type=float, default=1e-4, help='learning rate of iem')
parser.add_argument('--lr_cem', type=float, default=1e-5, help='learning rate of cem (initial value)')
parser.add_argument('--step_size', type=int, default=100, help='period of learning rate decay (cem)')
parser.add_argument('--max_norm', type=int, default=5, help='max gradient norm')

parser.add_argument('--stages', type=int, default=3, help='no. of iem (sci) stages')
parser.add_argument('--cem', type=str, default='sigmoid', help='cem')
# The value of the argument 'cem' should be chosen from the following
# options:
#   'baseline': regarding reflectance 'r' as enhanced image 'y',
#       similar to convenitonal illumination estimation-centric
#       methods (w/o CEM)
#   'betagamma': BetaGamma Correction
#   'preferred': Preferred Correction
#   'sigmoid': Sigmoid Correction

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


save = os.path.join(args.save, args.dataset, 'train-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
utils.create_exp_dir(save, scripts_to_save=glob.glob('*.py'))
model_path = os.path.join(save, 'models')
os.makedirs(model_path, exist_ok=True)
with open(os.path.join(save, 'cem.txt'), mode='w') as f:
    f.write(args.cem)

logger = utils.setup_logger('train', os.path.join(save, 'train.log'), '%(asctime)s %(message)s', True)
logger.info("filename = %s", os.path.split(__file__))
logger.info("args = %s", args)
logger.info("gpu device = %s", args.gpu)
logger.info("experiment dir = %s", save)

# report psnr and ssim of test images every 10 epochs
logger_test = utils.setup_logger('test', os.path.join(save, 'test.log'), '%(message)s')


def main():
    if not torch.cuda.is_available():
        logger.info("no gpu device available")
        sys.exit(1)

    cudnn.benchmark = True
    cudnn.enabled = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    network = model.network(stages=args.stages, cem=args.cem)
    network.enhance.in_conv.apply(network.weights_init)
    network.enhance.conv.apply(network.weights_init)
    network.enhance.out_conv.apply(network.weights_init)
    network.calibrate.in_conv.apply(network.weights_init)
    network.calibrate.conv.apply(network.weights_init)
    network.calibrate.out_conv.apply(network.weights_init)
    network = network.cuda()
    # set different learning rates for iem and cem
    optimizer = torch.optim.Adam([{'params': network.enhance.parameters()},
                                  {'params': network.calibrate.parameters()},
                                  {'params': network.cem.parameters(), 'lr': args.lr_cem}],
                                 lr=args.lr_iem, betas=(0.9, 0.999), weight_decay=args.weight_decay)


    train_path = os.path.join('./data', args.dataset, 'train/input')
    train_data = utils.MemoryFriendlyLoader(img_dir=train_path, task='train')

    test_path = os.path.join('./data', args.dataset, 'test/input')
    test_data = utils.MemoryFriendlyLoader(img_dir=test_path, task='test')

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=True)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=True)


    for epoch in range(epochs):
        network.train()
        losses = []
        for batch_idx, (input, _) in enumerate(train_queue):
            input = autograd.Variable(input, requires_grad=False).cuda()

            optimizer.zero_grad()
            loss = network._loss(input)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), args.max_norm)
            optimizer.step()
            losses.append(loss.item())

            a = 0
            b = 0
            if args.cem is not 'baseline':
                a = network.cem.a
                b = network.cem.b
            logger.info("train-epoch %03d %03d loss %f a %f b %f", epoch + 1, batch_idx, loss, a, b)
            # break

        logger.info("train-epoch %03d loss %f", epoch + 1, np.average(losses))
        torch.save(network.state_dict(), os.path.join(model_path, 'model_%03d.pt' % (epoch + 1)))

        # decay learning rate of cem by 0.1 every 'step_size' epochs
        if (epoch + 1) % args.step_size == 0:
            optimizer.param_groups[2]['lr'] /= 10


        if (epoch + 1) % 10 == 0:
            network.eval()
            mean_psnr = 0
            mean_ssim = 0
            with torch.no_grad():
                for _, (input, img_name) in enumerate(test_queue):
                    input = autograd.Variable(input).cuda()
                    img_name = img_name[0]

                    _, _, y = network(input)
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

            n = len(test_queue.dataset)
            mean_psnr /= n
            mean_ssim /= n

            a = 0
            b = 0
            if args.cem is not 'baseline':
                a = network.cem.a
                b = network.cem.b
            # report psnr and ssim of test images every 10 epochs
            logger_test.info("test-epoch %03d psnr %f ssim %f a %f b %f",
                             epoch + 1, mean_psnr, mean_ssim, a, b)


if __name__ == '__main__':
    main()
