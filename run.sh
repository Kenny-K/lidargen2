export CUDA_VISIBLE_DEVICES=1,2,3
export KITTI360_DATASET=/sharedata/home/shared/jiangq/KITTI-360


# training
python lidargen.py --train --exp train_bev --config kitti_bev.yml

# sampling 
# python lidargen.py --sample --exp train_lidargen --config kitti.yml

# evaluate by FID
# python lidargen.py --fid --exp train_lidargen --config kitti.yml

# visualization
# python LiDARGen/visualization.py --exp train_lidargen

# cmake openexr/ -DCMAKE_INSTALL_BINDIR=/sharedata/home/jiangq/anaconda3/bin -DCMAKE_INSTALL_DATAROOTDIR=/sharedata/home/jiangq/anaconda3/share -DCMAKE_INSTALL_INCLUDEDIR=include -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_LIBEXECDIR=libexec -DCMAKE_INSTALL_LOCALSTATEDIR=var -DCMAKE_INSTALL_OLDINCLUDEDIR=/usr/include CMAKE_INSTALL_PREFIX=/sharedata/home/jiangq/anaconda3/envs/lidar3/bin CMAKE_INSTALL_SBINDIR=sbin