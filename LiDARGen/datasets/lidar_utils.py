import numba as nb
import numpy as np



def load_matrices(kitti_path, data_name):
    cam_to_velo_path = kitti_path + '/calibration/calib_cam_to_velo.txt'
    cam_to_velo = np.identity(4)
    cam_to_velo[0:3, :] =  np.loadtxt(cam_to_velo_path, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)).reshape(3, 4)
    velo_to_cam = np.linalg.inv(cam_to_velo)

    calib_cam_to_pose_path = kitti_path + '/calibration/calib_cam_to_pose.txt'
    calib_cam_to_pose = np.identity(4)
    calib_cam_to_pose[0:3, :] = np.loadtxt(calib_cam_to_pose_path, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))[0].reshape(3, 4)

    poses_path = kitti_path + '/data_poses/' + data_name + '/poses.txt'
    poses_loaded = np.loadtxt(poses_path, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)).reshape(-1, 3, 4)

    poses = np.identity(4)
    poses = poses[np.newaxis, :, :]
    poses = np.repeat(poses, poses_loaded.shape[0], axis=0)
    poses[:,0:3,:] = poses_loaded

    velo_to_pose = calib_cam_to_pose @ velo_to_cam

    return velo_to_pose, poses




def point_cloud_to_range_image(point_cloud, isMatrix, return_remission = False, return_points=False):
    if (isMatrix):
        laser_scan = LaserScan()
        laser_scan.set_points(point_cloud)
        laser_scan.do_range_projection()
    else:
        laser_scan = LaserScan()
        laser_scan.open_scan(point_cloud)
        laser_scan.do_range_projection()

    if return_points:
        return laser_scan.proj_xyz, laser_scan.proj_range, laser_scan.proj_remission
    elif return_remission:
        return laser_scan.proj_range, laser_scan.proj_remission
    else:
        return laser_scan.proj_range

def point_cloud_to_bev_image(point_cloud, isMatrix, 
                             voxel_size, pcd_range, 
                             return_remission = False, return_points=False):
    if (isMatrix):
        laser_scan = LaserScan()
        laser_scan.set_points(point_cloud)
    else:
        laser_scan = LaserScan()
        laser_scan.open_scan(point_cloud)

    xyz = laser_scan.points
    xyz = xyz.squeeze()
    remission = laser_scan.remissions
    remission = remission.squeeze()
    
    voxel_size = np.array(voxel_size, dtype=np.float32)
    pcd_range = np.array(pcd_range, dtype=np.float32)
    pcd_range_mask = (xyz[:, 0] > pcd_range[0]) & (xyz[:,1] > pcd_range[1]) \
                    & (xyz[:, 2] > pcd_range[2]) & (xyz[:,0] < pcd_range[3]) \
                    & (xyz[:,1] < pcd_range[4]) & (xyz[:,2] < pcd_range[5])
    xyz = xyz[pcd_range_mask]
    remission = remission[pcd_range_mask]
    
    grid_size = (pcd_range[3:] - pcd_range[:3]) / voxel_size
    grid_size = np.around(grid_size).astype(np.int64)

    grid_ind = np.around( (xyz[:,:2] - pcd_range[:2]) / voxel_size[:2]).astype(np.int64)
    assert (grid_ind[:,0] <= grid_size[0]).any()
    assert (grid_ind[:,1] <= grid_size[1]).any()
    grid_size = grid_size[:2]
    height_map = np.zeros(grid_size)
    intensity_map = np.zeros(grid_size)
    remission_map = np.zeros(grid_size)
    height_data = xyz[:,2] - xyz[:,2].min(axis=-1)
    return bev_painting(grid_ind, grid_size, height_data, remission, height_map, remission_map, intensity_map)

@nb.jit(nopython=True)
def bev_painting(grid_ind, grid_size, height, remission, height_map, remission_map, intensity_map):
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            ind_mask = (grid_ind[:,0] == i) & (grid_ind[:,1] == j)
            num_pt = ind_mask.sum()
            if num_pt > 0:
                height_map[i][j] = np.amax(height[ind_mask])
                remission_map[i][j] = np.amax(remission[ind_mask])
                intensity_map[i][j] = num_pt
    return height_map, remission_map, intensity_map


'''
    Class taken fom semantic-kitti-api project.  https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/laserscan.py
'''
class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros(
            (0, 3), dtype=np.float32)        # [m, 3]: x, y, z
        self.remissions = np.zeros(
            (0, 1), dtype=np.float32)    # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)       # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """ Open raw scan and fill in attributes
        """
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]    # get xyz
        remissions = scan[:, 3]  # get remission
        self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points    # get xyz
        if remissions is not None:
            self.remissions = remissions  # get remission
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W                              # in [0.0, W]
        proj_y *= self.proj_H                              # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)
