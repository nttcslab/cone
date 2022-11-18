import argparse
import mmcv.cnn.utils.flops_counter as fc
import model
import os
import warnings


parser = argparse.ArgumentParser('cone')
parser.add_argument('--dataset', type=str, default='mit', help='dataset') # 'mit' or 'lsrw'
parser.add_argument('--save', type=str, default='exp', help='experiment dir')
parser.add_argument('--model', type=str, default='train-20221108-052232', help='target model')

args = parser.parse_args()

epochs_dic = {'mit': 500, 'lsrw': 200}
epochs = epochs_dic[args.dataset] # no. of epochs: 500 for mit and 200 for lsrw

model_path = os.path.join(args.save, args.dataset, args.model, 'models')


def main():
    network = model.inference(os.path.join(model_path, 'model_%03d.pt' % epochs))

    # For ease of comparisons, we computed the FLOPs by assuming the size
    # of the test image to be 600 × 400, which is in accordance with the
    # common configuration of RUAS [8] and SCI [9].
    x = (3, 600, 400)
    
    warnings.simplefilter('ignore')
    flops, params = fc.get_model_complexity_info(network, x, print_per_layer_stat=False, as_strings=False)
    print("params (k) %f mflops %f" % (params / 1e+3, flops / 1e+6))


if __name__ == '__main__':
    main()
