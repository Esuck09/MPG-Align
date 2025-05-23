import argparse
import datetime
import json
import random
import time
import math

import numpy as np
from pathlib import Path
from utils.misc import get_project_root
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch, validate
import os
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
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-5, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--lr_drop', default=60, type=int)
    
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
    parser.add_argument('--model_name', type=str, default='TransVG',
                        help="[TransVG, TransVG_m, TransVG_ca]")

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
                        help="[noLinear, kv, q_k_v, k_v]")
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

    # Genome supervision parameters
    parser.add_argument('--GNpath', type=str, default='data/MS_CXR/Genome',
                        help="path of imgGenome")
    parser.add_argument('--GNClsType', type=str, default='mcls',
                        help="[mcls, ml]")  # multi-class, multi-label
    parser.add_argument('--gnlossWeightBase', type=float, default=0.1,
                        help="loss weight")

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
    parser.add_argument('--ConsLossWeightBase', type=float, default=1.0,
                        help="obvious")
    parser.add_argument('--ablation', type=str, default='none',
                        help="[none, onlyImage, onlyText]")

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

    # Transformers in two branches
    parser.add_argument('--bert_enc_num', default=12, type=int)
    parser.add_argument('--detr_enc_num', default=6, type=int)

    # Vision-Language Transformer
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
    parser.add_argument('--vl_dec_layers', default=6, type=int,
                        help='Number of decoders in the vision-language transformer')

   # Dataset parameters
    parser.add_argument('--data_root', type=str, default='/mnt/DATASTORE/isaac/Code/Backup3/Backup/MedRPG-master/ln_data',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='MS_CXR', type=str,
                        help='referit/unc/unc+/gref/gref_umd/MS_CXR')
    parser.add_argument('--max_query_len', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    
   # dataset parameters
    parser.add_argument('--output_dir', default='outputs/MS_CXR',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_model_only', action='store_true')
    parser.add_argument('--detr_model', default='./pretrained/detr-r50-unc.pth', type=str, help='./checkpoints/detr-r50-unc.pth, detr model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # seed = args.seed + utils.get_rank()
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
    print("Model:", model)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    visu_cnn_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" in n) and ("backbone" in n) and p.requires_grad)]
    visu_tra_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" in n) and ("backbone" not in n) and p.requires_grad)]
    text_tra_param = [p for n, p in model_without_ddp.named_parameters() if (("textmodel" in n) and p.requires_grad)]
    rest_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]

    param_list = [{"params": rest_param},
                   {"params": visu_cnn_param, "lr": args.lr_visu_cnn},
                   {"params": visu_tra_param, "lr": args.lr_visu_tra},
                   {"params": text_tra_param, "lr": args.lr_bert},
                   ]

    # visu_param = [p for n, p in model_without_ddp.named_parameters() if "visumodel" in n and p.requires_grad]
    # text_param = [p for n, p in model_without_ddp.named_parameters() if "textmodel" in n and p.requires_grad]
    # rest_param = [p for n, p in model_without_ddp.named_parameters() if (("visumodel" not in n) and ("textmodel" not in n) and p.requires_grad)]
    
    
    # using RMSProp or AdamW
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    # using polynomial lr scheduler or half decay every 10 epochs or step
    if args.lr_scheduler == 'poly':
        lr_func = lambda epoch: (1 - epoch / args.epochs) ** args.lr_power
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'halfdecay':
        lr_func = lambda epoch: 0.5 ** (epoch // (args.epochs // 10))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'cosine':
        lr_func = lambda epoch: 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    else:
        raise ValueError('Lr scheduler type not supportted ')

    # build dataset
    dataset_train = build_dataset('train', args)
    dataset_val   = build_dataset('val', args)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    # dataset_test  =xi gai_dataset('test', args)
    
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val   = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        # model_without_ddp.load_state_dict(checkpoint['model'])
        # if args.model_name == 'TransVG_ca':
        #     del checkpoint['model']['vl_pos_embed.weight']

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        print(len(missing_keys), len(unexpected_keys))
        print("Resume Optimizer: ", not args.resume_model_only)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint and not args.resume_model_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    elif args.detr_model is not None:
        checkpoint = torch.load(os.path.join(root, args.detr_model), map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.visumodel.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys when loading detr model:')
        print(missing_keys)

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with open(os.path.join(root, output_dir, "log.txt"), "a") as f:
            f.write(str(args) + "\n")

    print("Start training")
    start_time = time.time()
    best_accu = 0
    best_miou = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            args, model, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()

        val_stats = validate(args, model, data_loader_val, device)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'validation_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 10 epochs
            # if (epoch + 1) == 45 or (epoch + 1) == 60 or (epoch + 1) == 75:
            #     checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            if val_stats['accu'] > best_accu:
                print(f"Epoch{epoch} have a best acc of {val_stats['accu']}. Saving!")
                checkpoint_paths.append(output_dir / 'best_accu_checkpoint.pth')
                best_accu = val_stats['accu']

            if val_stats['miou'] > best_miou:
                print(f"Epoch{epoch} have a best miou of {val_stats['miou']}. Saving!")
                checkpoint_paths.append(output_dir / 'best_miou_checkpoint.pth')
                best_miou = val_stats['miou']
            
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'val_accu': val_stats['accu'],
                    'val_miou': val_stats['miou']
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}, best_accu:{}'.format(total_time_str, best_accu))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransVG training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
