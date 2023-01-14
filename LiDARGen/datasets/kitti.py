from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from glob import glob
from .lidar_utils import point_cloud_to_range_image, point_cloud_to_bev_image, read_raw_point_cloud

class KITTI(Dataset):

    def __init__(self, path, config, split = 'train', resolution=None, transform=None):
        self.transform = transform
        self.return_remission = (config.data.channels == 2)
        self.random_roll = config.data.random_roll
        full_list = glob(os.path.join(os.environ.get('KITTI360_DATASET'), 'data_3d_raw/*/velodyne_points/data/*.bin'))
        if split == "train":
            self.full_list = list(filter(lambda file: '0000_sync' not in file and '0001_sync' not in file, full_list))
        else:
            self.full_list = list(filter(lambda file: '0000_sync' in file or '0001_sync' in file, full_list))
        self.length = len(self.full_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        filename = self.full_list[idx]
        if self.return_remission:
            real, intensity = point_cloud_to_range_image(filename, False, self.return_remission)
        else:
            real = point_cloud_to_range_image(filename, False, self.return_remission)
        #Make negatives 0
        real = np.where(real<0, 0, real) + 0.0001
        #Apply log
        real = ((np.log2(real+1)) / 6)
        #Make negatives 0
        real = np.clip(real, 0, 1)
        random_roll = np.random.randint(1024)

        if self.random_roll:
            real = np.roll(real, random_roll, axis = 1)
        real = np.expand_dims(real, axis = 0)

        if self.return_remission:
            intensity = np.clip(intensity, 0, 1.0)
            if self.random_roll:
                intensity = np.roll(intensity, random_roll, axis = 1)
            intensity = np.expand_dims(intensity, axis = 0)
            real = np.concatenate((real, intensity), axis = 0)

        return real, 0


class KITTI_BEV(Dataset):

    def __init__(self, preprocess_path, config, normalize=False, split = 'train', transform=None):
        self.transform = transform
        self.return_remission = (config.data.channels == 2)
        self.config = config
        self.normalize = normalize

        if os.path.exists(preprocess_path):
            self.preprocess = True
            if split == "train":
                self.full_list = glob(preprocess_path + '/train/*.npy')
            else:
                self.full_list = glob(preprocess_path + "/test/*.npy")
            self.full_list.sort(key=lambda x: int(x.split('/')[-1].rstrip('.npy').split('_')[-1]))
        else:
            self.preprocess = False
            full_list = glob(os.path.join(os.environ.get('KITTI360_DATASET'), 'data_3d_raw/*/velodyne_points/data/*.bin'))
            if split == "train":
                self.full_list = list(filter(lambda file: '0000_sync' not in file and '0001_sync' not in file, full_list))
            else:
                self.full_list = list(filter(lambda file: '0000_sync' in file or '0001_sync' in file, full_list))

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        filename = self.full_list[index]
        if not self.preprocess:
            height_map, remission_map, intensity_map = point_cloud_to_bev_image(filename, False, 
                                                self.config.data.voxel_size, 
                                                self.config.data.point_cloud_range)                                         
            data = np.stack([height_map, remission_map, intensity_map], axis=0)
        else:
            data = np.load(filename).astype(np.float)
        
        if self.normalize:
            max_h = data.max(axis=(1,2),keepdims=True)
            min_h = data.min(axis=(1,2),keepdims=True)
            assert (max_h > min_h).all()
            bev_map = (data - min_h) / (max_h - min_h)     # normalized to [0,1]
        else:
            bev_map = data

        return bev_map, 0
    

class KITTI_Polar(Dataset):
    def __init__(self, config, split = 'train'):
        self.config = config
        self.rotate_aug = config.data.rotate_aug
        self.flip_aug = config.data.flip_aug
        self.grid_size = np.asarray([self.config.data.image_size, self.config.data.image_width, self.config.data.channels])
        full_list = glob(os.path.join(os.environ.get('KITTI360_DATASET'), 'data_3d_raw/*/velodyne_points/data/*.bin'))
        if split == "train":
            self.full_list = list(filter(lambda file: '0000_sync' not in file and '0001_sync' not in file, full_list))
        else:
            self.full_list = list(filter(lambda file: '0000_sync' in file or '0001_sync' in file, full_list))
        self.length = len(self.full_list)

    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        filename = self.full_list[idx]
        xyz, sig = read_raw_point_cloud(filename, return_remission=True)

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)
        
        max_bound_r = np.percentile(xyz_pol[:,0],100,axis = 0)
        min_bound_r = np.percentile(xyz_pol[:,0],0,axis = 0)
        max_bound = np.max(xyz_pol[:,1:],axis = 0)
        min_bound = np.min(xyz_pol[:,1:],axis = 0)
        max_bound = np.concatenate(([max_bound_r],max_bound))
        min_bound = np.concatenate(([min_bound_r],min_bound))
        # if self.fixed_volume_space:
        #     max_bound = np.asarray(self.max_volume_space)
        #     min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1)

        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int)

        return_fea = np.concatenate([xyz, sig[:,np.newaxis]],axis=1)
     
        return grid_ind, return_fea
    

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)

def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
    return np.stack((x,y,input_xyz_polar[2]),axis=0)

def collate_fn_BEV(data):
    batch_grid_ind = [d[0] for d in data]
    batch_return_fea = [d[1] for d in data]
    return batch_grid_ind, batch_return_fea