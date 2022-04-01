import os
import moxing as mox
import argparse
import logging


parser = argparse.ArgumentParser()
parser.add_argument('--train_url', type=str, default=' ', help='the output path')
parser.add_argument('--s3_path', type=str, default='s3://bucket-9737/tianyunjie/checkpoints/MAE/', help='the path of the config file')
# configs/selfsup/byol/r50_bs1024_accum4_ep100.py
args, unparsed = parser.parse_known_args()

print('train_url:', args.train_url)
# ############# preparation stage ####################

print('Start copying dataset')
mox.file.copy_parallel('s3://bucket-9737/tianyunjie/datasets/imagenet-1000-tar/imagenet.tar', '/cache/imagenet.tar')
os.system('tar xf /cache/imagenet.tar -C /cache/')
print('Finish copying dataset')

# ############# preparation stage ####################

os.system('sh ./dist_train.sh 8 %s ' % (str(args.s3_path)))
