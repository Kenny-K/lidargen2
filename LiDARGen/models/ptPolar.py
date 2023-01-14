#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter

class ptPolar(nn.Module):
    
    def __init__(self, grid_size, pt_pooling = 'max', fea_dim = 3,
                 max_pt_per_encode = 64, pt_selection = 'farthest'):
        super(ptPolar, self).__init__()
        assert pt_pooling in ['max']
        assert pt_selection in ['random','farthest']

        self.pt_pooling = pt_pooling
        self.max_pt = max_pt_per_encode
        self.pt_selection = pt_selection
        self.grid_size = grid_size
        self.pt_fea_dim = fea_dim

        
    def forward(self, pt_fea, xy_ind, normalize=True):
        """
            Polar voxelize module that should be used with torch.no_grad()
            pt_fea[i] := (x,y,z,r) feature of shape [N,4]
            xy_ind[i] := (x,y,z) voxel grid index of shape [N,3] in dtype "torch.int64"
        """

        cur_dev = pt_fea[0].get_device()
        
        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch],(1,0),'constant',value = i_batch))

        cat_pt_fea = torch.cat(pt_fea,dim = 0)      # (x,y,z,r)
        cat_pt_ind = torch.cat(cat_pt_ind,dim = 0)  # (b,x,y,z)
        pt_num = cat_pt_ind.shape[0]
        
        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind,return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)
        
        # subsample pts
        if self.pt_selection == 'random':
            grp_ind = grp_range_torch(unq_cnt,cur_dev)[torch.argsort(torch.argsort(unq_inv))]
            remain_ind = grp_ind < self.max_pt
        elif self.pt_selection == 'farthest':
            unq_ind = np.split(np.argsort(unq_inv.detach().cpu().numpy()), np.cumsum(unq_cnt.detach().cpu().numpy()[:-1]))
            remain_ind = np.zeros((pt_num,),dtype = np.bool)
            np_cat_fea = cat_pt_fea.detach().cpu().numpy()[:,:3]
            pool_in = []
            for i_inds in unq_ind:
                if len(i_inds) > self.max_pt:
                    pool_in.append((np_cat_fea[i_inds,:],self.max_pt))
            if len(pool_in) > 0:
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                FPS_results = pool.starmap(parallel_FPS, pool_in)
                pool.close()
                pool.join()
            count = 0
            for i_inds in unq_ind:
                if len(i_inds) <= self.max_pt:
                    remain_ind[i_inds] = True
                else:
                    remain_ind[i_inds[FPS_results[count]]] = True
                    count += 1
            
        cat_pt_fea = cat_pt_fea[remain_ind,:]
        cat_pt_ind = cat_pt_ind[remain_ind,:]
        unq_inv = unq_inv[remain_ind]
        unq_cnt = torch.clamp(unq_cnt,max=self.max_pt)
        
        # process feature(h:height, r:remission, i:intensity)
        if self.pt_pooling == 'max':
            h_data, h_argmax = torch_scatter.scatter_max(cat_pt_fea[:,2], unq_inv, dim=0)       # [N,]
            r_data = cat_pt_fea[:,3][h_argmax]      # [N,]
            # i_data is unq_cnt
        else: raise NotImplementedError
        h_data,r_data,i_data = h_data.view(-1,1), r_data.view(-1,1), unq_cnt.view(-1,1)
        processed_pooled_data = torch.cat([h_data, r_data, i_data],axis=1)
        
        # stuff pooled data into 4D tensor
        out_data_dim = [len(pt_fea),self.grid_size[0],self.grid_size[1],self.pt_fea_dim]
        out_data = torch.zeros(out_data_dim, dtype=torch.float32).to(cur_dev)
        out_data[unq[:,0],unq[:,1],unq[:,2],:] = processed_pooled_data
        out_data = out_data.permute(0,3,1,2)        # BHWC -> BCHW

        if normalize:
            max_h = out_data.amax(dim=(2,3),keepdims=True)
            min_h = out_data.amin(dim=(2,3),keepdims=True)
            assert (max_h > min_h).all()
            out_data = (out_data - min_h) / (max_h - min_h)  
        
        return out_data
    
def grp_range_torch(a,dev):
    idx = torch.cumsum(a,0)
    id_arr = torch.ones(idx[-1],dtype = torch.int64,device=dev)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return torch.cumsum(id_arr,0)

def parallel_FPS(np_cat_fea,K):
    return  nb_greedy_FPS(np_cat_fea,K)

@nb.jit('b1[:](f4[:,:],i4)',nopython=True,cache=True)
def nb_greedy_FPS(xyz,K):
    start_element = 0
    sample_num = xyz.shape[0]
    sum_vec = np.zeros((sample_num,1),dtype = np.float32)
    xyz_sq = xyz**2
    for j in range(sample_num):
        sum_vec[j,0] = np.sum(xyz_sq[j,:])
    pairwise_distance = sum_vec + np.transpose(sum_vec) - 2*np.dot(xyz, np.transpose(xyz))
    
    candidates_ind = np.zeros((sample_num,),dtype = np.bool_)
    candidates_ind[start_element] = True
    remain_ind = np.ones((sample_num,),dtype = np.bool_)
    remain_ind[start_element] = False
    all_ind = np.arange(sample_num)
    
    for i in range(1,K):
        if i == 1:
            min_remain_pt_dis = pairwise_distance[:,start_element]
            min_remain_pt_dis = min_remain_pt_dis[remain_ind]
        else:
            cur_dis = pairwise_distance[remain_ind,:]
            cur_dis = cur_dis[:,candidates_ind]
            min_remain_pt_dis = np.zeros((cur_dis.shape[0],),dtype = np.float32)
            for j in range(cur_dis.shape[0]):
                min_remain_pt_dis[j] = np.min(cur_dis[j,:])
        next_ind_in_remain = np.argmax(min_remain_pt_dis)
        next_ind = all_ind[remain_ind][next_ind_in_remain]
        candidates_ind[next_ind] = True
        remain_ind[next_ind] = False
        
    return candidates_ind