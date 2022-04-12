import ast
import os
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--train_url', type=str, default=' ', help='the output path')
parser.add_argument('--batch_size', type=int, default=128, help='the path of the config file')
parser.add_argument('--mask_ratio', type=float, default=0.75, help='the path of the config file')
parser.add_argument('--return_full', type=bool, default=True, help='the path of the config file')
parser.add_argument('--epochs', type=int, default=200, help='the path of the config file')
parser.add_argument('--warmup_epochs', type=int, default=20, help='the path of the config file')
parser.add_argument('--model', type=str, default='pretrain_mae_base_patch16_224', help='the path of the config file')

parser.add_argument('--num_gpus', type=int, default=8, help='the number of gpus')
parser.add_argument('--rank', type=int, default=0, help='node rank')
parser.add_argument('--world_size', type=int, default=2, help='world size')

args, unparsed = parser.parse_known_args()

print('train_url:', args.train_url)
# ############# preparation stage ####################
print('Current path: ' + os.getcwd())
print('Current dirs: ' + str(list(os.listdir())))
print()
os.chdir('./zoomed-in')
print('Current path changed to: ' + os.getcwd())

os.system('pip install -r requirements.txt')

os.system('pip install --ignore-installed PyYAML')
# os.system('pip install -v --disable-pip-version-check --no-cache-dir ./apex-master/')
os.system(
    'pip install -q --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex/')

###################################################################################################
master_host = os.environ['VC_WORKER_HOSTS'].split(',')[0]
master_addr = master_host.split(':')[0]
master_port = '8524'
# FLAGS.worldsize will be re-computed follow as FLAGS.ngpu*FLAGS.nodes_num
# FLAGS.rank will be re-computed in main_worker
modelarts_rank = args.rank  # ModelArts receive FLAGS.rank means node_rank
modelarts_world_size = args.world_size  # ModelArts receive FLAGS.worldsize means nodes_num
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port

print(f'IP: {master_addr},  Port: {master_port}')
print(f'modelarts rank {modelarts_rank}, world_size {modelarts_world_size}')
###########################################################################################################

cmd_str = f"python -m torch.distributed.launch --nproc_per_node {args.num_gpus} \
    --nnodes={modelarts_world_size} --node_rank={modelarts_rank} --master_addr={master_addr} \
    --master_port={master_port} run_mae_pretraining.py --batch_size {args.batch_size} --epochs {args.epochs} \
    --model {args.model} --warmup_epochs {args.warmup_epochs} --mask_ratio {args.mask_ratio} \
    --return_full {args.return_full}"

print('The running command is: ' + cmd_str)

os.system(cmd_str)
