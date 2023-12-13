#!/usr/bin/env python
import argparse
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models


import loader
import builder
from sklearn.metrics.pairwise import cosine_similarity

from tensorboardX import SummaryWriter
from torch.nn import functional as F
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data-A', metavar='DIR Domain A', help='path to domain A dataset')
parser.add_argument('--data-B', metavar='DIR Domain B', help='path to domain B dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[50,100], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 2x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 0.0005)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--clean-model', default='', type=str, metavar='PATH',
                    help='path to clean model (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--low-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')
parser.add_argument('--epsilon', default=0.05, type=float,
                    help='sinkhorn regularization coefficient')
parser.add_argument('--sinkhorn-iterations', default=3, type=int,
                    help='sinkhorn algorithm max iterations')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco-v2/SimCLR data augmentation')
parser.add_argument('--exp-dir', default='experiment_pcl', type=str,
                    help='the directory of the experiment')
parser.add_argument('--num-cluster', default=100, type=int,
                    help='number of clusters for self entropy loss')
parser.add_argument('--prec-nums', default='1,5,15', type=str,
                    help='the evaluation metric')


def main():
    args = parser.parse_args()

    if args.seed is not None:

        setup_seed(args.seed)

        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    main_worker(args.gpu, args)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    cudnn.deterministic = True


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    print("=> creating model '{}'".format(args.arch))


    traindirA = args.data_A     
    traindirB = args.data_B   

    train_dataset = loader.TrainDataset(traindirA, traindirB, args.aug_plus)
    eval_dataset = loader.EvalDataset(traindirA, traindirB)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)

    model = builder.UCDIR(
        models.__dict__[args.arch],
        dim=args.low_dim, K_A=eval_dataset.domainA_size, K_B=eval_dataset.domainB_size,
        m=args.moco_m, T=args.temperature,epsilon=args.epsilon, sink_iters=args.sinkhorn_iterations)

    

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                betas=(0.9,0.999),eps=1e-8,weight_decay=args.weight_decay,amsgrad=True)



    if args.clean_model:
        if os.path.isfile(args.clean_model):
            print("=> loading pretrained clean model '{}'".format(args.clean_model))

            loc = 'cuda:{}'.format(args.gpu)
            clean_checkpoint = torch.load(args.clean_model, map_location=loc)

            current_state = model.state_dict()
            used_pretrained_state = {}

            for k in current_state:
                if 'encoder' in k:
                    k_parts = '.'.join(k.split('.')[1:])
                    used_pretrained_state[k] = clean_checkpoint['state_dict']['module.encoder_q.'+k_parts]
            current_state.update(used_pretrained_state)
            model.load_state_dict(current_state)
        else:
            print("=> no clean model found at '{}'".format(args.clean_model))

    # loc = 'cuda:{}'.format(args.gpu)
    # clean_checkpoint = torch.load(args.clean_model, map_location=loc)
    
    # model.load_state_dict(clean_checkpoint['state_dict'])
    
        

    info_save = open(os.path.join(args.exp_dir, 'results.txt'), 'w')
    best_res_A = [0., 0., 0.]
    best_res_B = [0., 0., 0.]


    writer = SummaryWriter(os.path.join(args.exp_dir,"results"))

    cluster_result = dict()

    for epoch in range(args.epochs):

        if epoch == 0:
            features_A, features_B,targets_A, targets_B = compute_features(eval_loader, model, args)

            

            features_A = features_A.numpy()
            features_B = features_B.numpy()
            
            model.queue_A.data = torch.tensor(features_A).T.cuda()
            model.queue_B.data = torch.tensor(features_B).T.cuda()


            targets_A = targets_A.numpy()
            targets_B = targets_B.numpy()

            prec_nums = args.prec_nums.split(',')
            res_A, res_B = retrieval_precision_cal(features_A, targets_A, features_B, targets_B,
                                                preck=(int(prec_nums[0]), int(prec_nums[1]), int(prec_nums[2])))
        

            info_save.write("Domain A->B: P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_A[0],
                                                                                int(prec_nums[1]), res_A[1],
                                                                                int(prec_nums[2]), res_A[2]))
            info_save.write("Domain B->A: P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_B[0],
                                                                            int(prec_nums[1]), res_B[1],
                                                                            int(prec_nums[2]), res_B[2]))

            info_save.flush()
            print("Domain A->B: P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_A[0],
                                                                                int(prec_nums[1]), res_A[1],
                                                                                int(prec_nums[2]), res_A[2]))
            print("Domain B->A: P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_B[0],
                                                                          int(prec_nums[1]), res_B[1],
                                                                          int(prec_nums[2]), res_B[2]))
        

        cluster_result =  run_kmeans_all(model.queue_A.data.T.cpu().numpy(), model.queue_B.data.T.cpu().numpy(), args,cluster_result,epoch)

        train(train_loader, model, optimizer, epoch, args, info_save,criterion,writer,cluster_result)

        features_A, features_B, targets_A, targets_B = compute_features(eval_loader, model, args)
        
       

        features_A = features_A.numpy()
        targets_A = targets_A.numpy()

        features_B = features_B.numpy()
        targets_B = targets_B.numpy()
        
            
        prec_nums = args.prec_nums.split(',')
        res_A, res_B = retrieval_precision_cal(features_A, targets_A, features_B, targets_B,
                                               preck=(int(prec_nums[0]), int(prec_nums[1]), int(prec_nums[2])))
        
        
        
        if (best_res_A[0] + best_res_B[0]) / 2 < (res_A[0] + res_B[0]) / 2:
            best_res_A = res_A
            best_res_B = res_B

        info_save.write("Domain A->B: P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_A[0],
                                                                            int(prec_nums[1]), res_A[1],
                                                                            int(prec_nums[2]), res_A[2]))
        info_save.write("Domain B->A: P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_B[0],
                                                                          int(prec_nums[1]), res_B[1],
                                                                          int(prec_nums[2]), res_B[2]))

        info_save.flush()
        print("Domain A->B: P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_A[0],
                                                                            int(prec_nums[1]), res_A[1],
                                                                            int(prec_nums[2]), res_A[2]))
        print("Domain B->A: P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), res_B[0],
                                                                          int(prec_nums[1]), res_B[1],
                                                                          int(prec_nums[2]), res_B[2]))
        

        #保存最好的值
        if epoch == 199:
            info_save.write("Best_result A->B: P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), best_res_A[0],
                                                                            int(prec_nums[1]), best_res_A[1],
                                                                            int(prec_nums[2]), best_res_A[2]))
            info_save.write("Best_result B->A: P@{}: {}; P@{}: {}; P@{}: {} \n".format(int(prec_nums[0]), best_res_B[0],
                                                                          int(prec_nums[1]), best_res_B[1],
                                                                          int(prec_nums[2]), best_res_B[2]))

        writer.add_scalar('Domain A_to_B/P@50',res_A[0],epoch)
        writer.add_scalar('Domain A_to_B/P@100',res_A[1],epoch)
        writer.add_scalar('Domain A_to_B/P@200',res_A[2],epoch)
        writer.add_scalar('Domain B_to_A/P@50',res_B[0],epoch)
        writer.add_scalar('Domain B_to_A/P@100',res_B[1],epoch)
        writer.add_scalar('Domain B_to_A/P@200',res_B[2],epoch)

        if epoch == 199:
            checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(args.exp_dir,str(epoch)+'_epoch_checkpoint.pth.tar'))
        
       



        
def train(train_loader, model,  optimizer, epoch, args, info_save,criterion,writer,cluster_result):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses = { 'in_domain_contrastive': AverageMeter('in_domain_contrastive', ':.4e'),
                'cross_domain_contrastive': AverageMeter('cross_domain_contrastive', ':.4e'),
                'Total_loss': AverageMeter('loss_Total', ':.4e')}

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time,
         losses['Total_loss'],losses['in_domain_contrastive'],
         losses['cross_domain_contrastive']],
        prefix="Epoch: [{}]".format(epoch))
    


    # switch to train mode
    model.train()

    end = time.time()
    for i, (images_A, image_ids_A, images_B, image_ids_B, cates_A, cates_B) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.gpu is not None:
            images_A[0] = images_A[0].cuda(args.gpu, non_blocking=True)
            images_A[1] = images_A[1].cuda(args.gpu, non_blocking=True)

            image_ids_A = image_ids_A.cuda(args.gpu, non_blocking=True)

            images_B[0] = images_B[0].cuda(args.gpu, non_blocking=True)
            images_B[1] = images_B[1].cuda(args.gpu, non_blocking=True)

            image_ids_B = image_ids_B.cuda(args.gpu, non_blocking=True)
        
        loss_in_domain,loss_cross_domain= model(im_q_A=images_A[0], im_k_A=images_A[1],
                             im_id_A=image_ids_A, im_q_B=images_B[0],
                             im_k_B=images_B[1], im_id_B=image_ids_B,criterion=criterion,cluster_result=cluster_result)


        losses['in_domain_contrastive'].update(loss_in_domain.item(), images_B[0].size(0)) 
        losses['cross_domain_contrastive'].update(loss_cross_domain.item(), images_B[0].size(0))

        all_loss =  loss_in_domain

        
        if epoch >= 100:
            all_loss +=  0.01*loss_cross_domain

            
        writer.add_scalar('loss/loss_all',all_loss,epoch *len(train_loader) +i)

        
        

        losses['Total_loss'].update(all_loss.item(), images_A[0].size(0))

        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            info = progress.display(i)
            info_save.write(info + '\n')
            info_save.flush()
        



def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()

    features_A = torch.zeros(eval_loader.dataset.domainA_size, args.low_dim).cuda()
    features_B = torch.zeros(eval_loader.dataset.domainB_size, args.low_dim).cuda()

    targets_all_A = torch.zeros(eval_loader.dataset.domainA_size, dtype=torch.int64).cuda()
    targets_all_B = torch.zeros(eval_loader.dataset.domainB_size, dtype=torch.int64).cuda()


    for i, (images_A, indices_A, targets_A, images_B, indices_B, targets_B) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images_A = images_A.cuda(non_blocking=True)
            images_B = images_B.cuda(non_blocking=True)

            targets_A = targets_A.cuda(non_blocking=True)
            targets_B = targets_B.cuda(non_blocking=True)

            feats_A, feats_B = model(im_q_A=images_A, im_q_B=images_B, is_eval=True)

            features_A[indices_A] = feats_A
            features_B[indices_B] = feats_B

            targets_all_A[indices_A] = targets_A
            targets_all_B[indices_B] = targets_B


    return features_A.cpu(), features_B.cpu(), targets_all_A.cpu(), targets_all_B.cpu()



def retrieval_precision_cal(features_A, targets_A, features_B, targets_B, preck=(1, 5, 15)):

    dists = cosine_similarity(features_A, features_B)

    res_A = []
    res_B = []
    for domain_id in ['A', 'B']:
        if domain_id == 'A':
            query_targets = targets_A
            gallery_targets = targets_B

            all_dists = dists

            res = res_A
        else:
            query_targets = targets_B
            gallery_targets = targets_A

            all_dists = dists.transpose()
            res = res_B

        sorted_indices = np.argsort(-all_dists, axis=1)

        sorted_cates = gallery_targets[sorted_indices.flatten()].reshape(sorted_indices.shape)
        correct = (sorted_cates == np.tile(query_targets[:, np.newaxis], sorted_cates.shape[1]))

        for k in preck:
            total_num = 0
            positive_num = 0
            for index in range(all_dists.shape[0]):

                temp_total = min(k, (gallery_targets == query_targets[index]).sum())
                pred = correct[index, :temp_total]

                total_num += temp_total
                positive_num += pred.sum()
            res.append(positive_num / total_num * 100.0)

    return res_A, res_B





def run_kmeans_all(x_A, x_B, args,cluster_result,epoch):
    print('performing kmeans clustering all')
    for domain_id in ['A', 'B']:
        if domain_id == 'A':
            x = x_A
        elif domain_id == 'B':
            x = x_B
        else:
            x = np.concatenate([x_A, x_B], axis=0)

        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(args.num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = False
        clus.niter = 50
        clus.nredo = 20
        clus.seed = 0
        clus.max_points_per_centroid = 2000
        clus.min_points_per_centroid = 2
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.gpu
        index = faiss.IndexFlatL2(d)

        clus.train(x, index)
        D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids_normed = nn.functional.normalize(centroids, p=2, dim=1)
        im2cluster = torch.LongTensor(im2cluster).cuda()

        if epoch == 0 :
       
            cluster_result['centroids_'+domain_id] = centroids_normed
            cluster_result['im2cluster_'+domain_id] = im2cluster

        counts = torch.unique(im2cluster,return_counts = True)
        probability_dist = counts[1]/counts[1].sum()


        cluster_result['dist_'+domain_id] = probability_dist

    return cluster_result



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        return ' '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



if __name__ == '__main__':
    main()
