import argparse
import datetime
import json
import random
import time
import math

import numpy as np
from pathlib import Path
import os

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import utils.misc as utils
from utils.misc import get_project_root
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, evaluate
os.environ['CURL_CA_BUNDLE'] = '/usr/local/share/ca-certificates/a.crt'

"""
This code is primarily based on the MedRPG implementation from:

Chen, Zhihao et al. "Medical Phrase Grounding with Region-Phrase Context Contrastive Alignment."
MICCAI, 2023. https://arxiv.org/abs/2307.11767

Original code: https://github.com/openmedlab/MedRPG

Please refer to the original authors for core algorithmic contributions.
"""

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=0., type=float)
    parser.add_argument('--lr_visu_cnn', default=0., type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='rmsprop', type=str)
    parser.add_argument('--lr_scheduler', default='poly', type=str)
    parser.add_argument('--lr_drop', default=80, type=int)
    
    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true',
                        help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="If true, use random translate augmentation")

    # Model parameters
    parser.add_argument('--model_name', type=str, default='TransVG_ca',
                        help="Name of model to be exploited.")

    # Memory parameters
    parser.add_argument('--memory', default=False, action='store_true',
                        help='is or not memory matrix')
    parser.add_argument('--m_threadHead', type=int, default=4,
                        help="head num of multi-head memory")
    parser.add_argument('--m_size', type=int, default=2048,
                        help="memory size")
    parser.add_argument('--m_dim', type=int, default=256,
                        help="memory dimension")
    parser.add_argument('--m_topK', type=int, default=32,
                        help="choose memory topk item")
    parser.add_argument('--m_linearMode', type=str, default='noLinear',
                        help="transform mode for memory before querying")
    parser.add_argument('--m_MMHA_outLinear', default=False, action='store_true',
                        help='do or not transform for MMHA output')
    parser.add_argument('--m_resLearn', default=False, action='store_true',
                        help='residual learning for visual and language model')
    parser.add_argument('--btloss', default=False, action='store_true',
                        help='do or not bbox-text loss')
    parser.add_argument('--lossbt_type', type=str, default='l1',
                        help="type of bt loss")
    parser.add_argument('--lossbt_weight', type=float, default=1.0,
                        help="loss weight of bt loss")

    # BBox-Text contrastive alignment parameters
    parser.add_argument('--CAsampleType', type=str, default='random',
                        help="[random, attention, crossImage, crossBatch]")
    parser.add_argument('--CAsampleNum', default=5, type=int,
                        help="Number of negative samples to be chosen")
    parser.add_argument('--CAlossWeightBase', type=float, default=1.0,
                        help="obvious")
    parser.add_argument('--CATextPoolType', type=str, default='mask',
                        help="[mask, all]")
    parser.add_argument('--CATemperature', default=0.07, type=float,
                        help="hyper parameters temperature")
    parser.add_argument('--CAMode', type=str, default='max_image',
                        help="[max_image, max_batch]")
    parser.add_argument('--ablation', type=str, default='none',
                        help="[none, onlyImage, onlyText]")

    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)

    # DETR parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=0, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--imsize', default=640, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')
    # Vision-Language Transformer
    parser.add_argument('--use_vl_type_embed', action='store_true',
                        help="If true, use vl_type embedding")
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='/mnt/DATASTORE/isaac/Code/Backup3/Backup/MedRPG-master/ln_data',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='MS_CXR', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--max_query_len', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    
    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs/MS_CXR',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_model', default='./saved_models/detr-r50.pth', type=str, help='detr model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # evalutaion options
    parser.add_argument('--eval_set', default='val', type=str)
    parser.add_argument('--eval_model', default='/mnt/DATASTORE/isaac/Code/Backup3/Backup/MedRPG-master/released_checkpoint/TransVG_R50_unc.pth', type=str)

    # visualization options
    parser.add_argument('--visualization', action='store_true',
                        help="If true, visual the bbox")
    parser.add_argument('--visual_MHA', action='store_true',
                        help="If true, visual the attention maps")

    return parser


def main(args):
    args.distributed = False 
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # # fix the seed for reproducibility
    seed = args.seed
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

    root = get_project_root()

    # build model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # build dataset
    dataset_test = build_dataset(args.eval_set, args)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    # dataset_test  = build_dataset('test', args)
    
    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    
    batch_sampler_test = torch.utils.data.BatchSampler(
        sampler_test, args.batch_size, drop_last=False)

    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(os.path.join(root, args.eval_model), weights_only=False)
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    # output log
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "eval_log.txt").open("a") as f:
            f.write(str(args) + "\n")
    
    start_time = time.time()
    
    # perform evaluation
    results = evaluate(args, model, data_loader_test, device)
    
    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        if args.dataset in ['MS_CXR', 'ChestXray8']:
            log_stats = {
                'test_model:': args.eval_model,
                '%s_set_total_accuracy' % args.eval_set: float(round(results['total_accuracy'], 4)),
                '%s_set_total_miou' % args.eval_set: float(round(results['total_iou'], 4)),
                '%s_set_sub_accu' % args.eval_set: [float(round(j, 4)) for j in results['category_accu']],
                '%s_set_sub_iou' % args.eval_set: [float(round(j, 4)) for j in results['category_iou']],
            }
        else:
            log_stats = {
                'test_model:': args.eval_model,
                '%s_set_total_accuracy' % args.eval_set: float(round(results['total_accuracy'], 4)),
                '%s_set_total_miou' % args.eval_set: float(round(results['total_iou'], 4)),
            }
        print(log_stats)
        if args.output_dir and utils.is_main_process():
                with (output_dir / "eval_log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransVG evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
