import torch
import torch.nn as nn
from functools import partial
from torch.nn import functional as F


class UCDIR(nn.Module):

    def __init__(self, base_encoder, dim=128, K_A=65536, K_B=65536,
                 m=0.999, T=0.2,epsilon=0.05,sink_iters=3):

        super(UCDIR, self).__init__()

        self.K_A = K_A
        self.K_B = K_B
        self.m = m
        self.T = T
        self.epsilon = epsilon
        self.sink_iters = sink_iters

        norm_layer = partial(SplitBatchNorm, num_splits=2)

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim,norm_layer=norm_layer)

      
        dim_mlp = self.encoder_q.fc.weight.shape[1]

        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queues
        self.register_buffer("queue_A", torch.randn(dim, K_A))
        self.queue_A = F.normalize(self.queue_A, dim=0)
        self.register_buffer("queue_A_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_B", torch.randn(dim, K_B))
        self.queue_B = F.normalize(self.queue_B, dim=0)
        self.register_buffer("queue_B_ptr", torch.zeros(1, dtype=torch.long))




    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue_singlegpu(self, keys, key_ids, domain_id):

        if domain_id == 'A':
            self.queue_A.index_copy_(1, key_ids, keys.T)
        elif domain_id == 'B':
            self.queue_B.index_copy_(1, key_ids, keys.T)



    @torch.no_grad()
    def _batch_shuffle_singlegpu(self, x):

        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_singlegpu(self, x, idx_unshuffle):

        return x[idx_unshuffle]

    def forward(self, im_q_A=None, im_k_A=None, im_id_A=None,im_q_B=None,
                im_k_B=None, im_id_B=None, is_eval=False,criterion=None,cluster_result=None,flag=None):

        im_q = torch.cat([im_q_A, im_q_B], dim=0)

        if is_eval:
            k = self.encoder_k(im_q)
            k = F.normalize(k, dim=1)

            k_A, k_B = torch.split(k, im_q_A.shape[0])
            return k_A, k_B


        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        q_A, q_B = torch.split(q, im_q_A.shape[0])

        im_k = torch.cat([im_k_A, im_k_B], dim=0)
        


        with torch.no_grad():
            self._momentum_update_key_encoder()

            im_k, idx_unshuffle = self._batch_shuffle_singlegpu(im_k)
            
            k = self.encoder_k(im_k) 

            k = F.normalize(k, dim=1)

            k = self._batch_unshuffle_singlegpu(k, idx_unshuffle)

            k_A, k_B = torch.split(k, im_k_A.shape[0])
            
        

        self._dequeue_and_enqueue_singlegpu(k_A, im_id_A, 'A')
        self._dequeue_and_enqueue_singlegpu(k_B, im_id_B, 'B')

       
       

        loss_in_domain = self.in_domain(q_A, k_A, im_id_A, q_B, k_B, im_id_B, cluster_result,criterion)


        loss_cross_domain = self.cross_domain(q_A,im_id_A,q_B,im_id_B,cluster_result,criterion)


        return   loss_in_domain,loss_cross_domain


        
    def in_domain(self, q_A, k_A, im_id_A, q_B, k_B, im_id_B, cluster_result,criterion):

        all_losses = dict()

        beta_A = cluster_result["dist_A"]
        beta_B = cluster_result["dist_B"]
        
        for domain_id in ['A', 'B']:
            if domain_id == 'A':
                im_id = im_id_A
                q_feat = q_A
                k_feat = k_A
                queue = self.queue_A.clone().detach().T
                beta = beta_A
        
            else:
                im_id = im_id_B
                q_feat = q_B
                k_feat = k_B
                queue = self.queue_B.clone().detach().T
                beta = beta_B

            prototypes= cluster_result['centroids_' + domain_id]

            similarity_feat = F.normalize(torch.matmul(q_feat,queue.T),dim=1)
            similarity_feat[torch.arange(q_feat.shape[0]).cuda(), im_id] = -1000
            nearest_feat_index = torch.argmax(similarity_feat, dim=1)
            nearest_feat = queue[nearest_feat_index]
            
            similarity_feat_proto = F.normalize(torch.matmul(queue, prototypes.T),dim=1)
            sim_code = ProtoOT(similarity_feat_proto.detach(),self.epsilon,self.sink_iters, beta)

            mapped_proto_index = torch.argmax(sim_code[im_id], dim=1)
            mapped_proto = prototypes[mapped_proto_index]

            l_pos_aug = torch.einsum('nc,nc->n', [q_feat, k_feat]).unsqueeze(-1)
            l_pos_nearest = torch.einsum('nc,nc->n', [q_feat, nearest_feat]).unsqueeze(-1)
            l_pos_proto = torch.einsum('nc,nc->n', [q_feat, mapped_proto]).unsqueeze(-1)
            
            mask = torch.arange(prototypes.shape[0]).cuda() != mapped_proto_index[:,None]
            l_all = torch.einsum('nc,ck->nk', [q_feat, prototypes.T])
            l_neg = torch.masked_select(l_all, mask).reshape(q_feat.shape[0], -1)

          
            logits_aug = torch.cat([l_pos_aug, l_neg], dim=1)
            logits_nearest = torch.cat([l_pos_nearest, l_neg], dim=1)
            logits_proto = torch.cat([l_pos_proto, l_neg], dim=1)
            
            logits_aug /= self.T
            logits_nearest /= self.T
            logits_proto /= self.T

            labels_aug = torch.zeros(logits_aug.shape[0], dtype=torch.long).cuda()
            labels_nearest = torch.zeros(logits_nearest.shape[0], dtype=torch.long).cuda()
            labels_proto = torch.zeros(logits_proto.shape[0], dtype=torch.long).cuda()

            loss = criterion(logits_aug, labels_aug) + 0.5*criterion(logits_nearest, labels_nearest) + 0.005*criterion(logits_proto, labels_proto)


            cluster_result['centroids_' + domain_id] = F.normalize(torch.matmul(sim_code.T, queue), dim=1)
       
            all_losses['domain_' + domain_id] = loss


        return all_losses['domain_A'] + all_losses['domain_B']
    
      
   
    
    def cross_domain(self,q_A,im_id_A,q_B,im_id_B,cluster_result,criterion):

        all_losses = dict()

        prototypes_A =  cluster_result['centroids_A'] 
        prototypes_B =  cluster_result['centroids_B'] 

        beta_A = cluster_result["dist_A"]
        beta_B = cluster_result["dist_B"]
        

        for domain_id in ['A', 'B']:
            if domain_id == 'A':
                im_id = im_id_A
                q_feat = q_A
                queue = self.queue_A.clone().detach().T
                prototypes = prototypes_A
                prototypes_cross = prototypes_B
                beta = beta_A

                

            else:
                im_id = im_id_B
                q_feat = q_B
                queue = self.queue_B.clone().detach().T
                prototypes = prototypes_B
                prototypes_cross = prototypes_A
                beta = beta_B

            
            similarity_feat_cross_proto = F.normalize(torch.matmul(queue, prototypes_cross.T), dim=1)
            feat_cross_proto_code = ProtoOT(similarity_feat_cross_proto.detach(),self.epsilon,self.sink_iters,beta)

            
            cross_cor_proto_index = torch.argmax(feat_cross_proto_code[im_id], dim=1)

            cross_proto = prototypes_cross[cross_cor_proto_index]
            

            l_pos_proto = torch.einsum('nc,nc->n', [q_feat, cross_proto]).unsqueeze(-1)
        
            l_pos = l_pos_proto
     
            max_idx = cross_cor_proto_index
            
            mask = torch.arange(prototypes.shape[0]).cuda() != max_idx[:,None]
            l_all = F.normalize(torch.matmul(q_feat, prototypes_cross.T), dim=1)
            l_neg = torch.masked_select(l_all, mask).reshape(q_feat.shape[0], -1)

          
            logits = torch.cat([l_pos, l_neg], dim=1)
            
            logits /= self.T

            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            loss = criterion(logits, labels)

            
            all_losses['domain_' + domain_id] = loss

        return all_losses['domain_A'] + all_losses['domain_B']
    


def ProtoOT(out, epsilon, sinkhorn_iterations,beta):
    """
    from https://github.com/facebookresearch/swav
    """
    Q = torch.exp(out/epsilon).t() 
    B = Q.shape[1] 

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q *= beta.unsqueeze(1)

        
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B 
    return Q.t()


# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape

        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)

            outcome = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)
