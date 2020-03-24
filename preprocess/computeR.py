import numpy as np
from PIL import Image


class IMUComputeH(object):
    def __init__(self, path):
        f_param = open(path + '_param.txt', 'r')
        f_IMU_data = open(path + '_IMU_ori.txt', 'r')

        param_data = f_param.readlines()
        IMU_data = f_IMU_data.read().splitlines()
        param_data = [line.strip().strip('\n') for line in param_data]
        IMU_data = np.asarray([line.strip().strip('\n').split(' ') for line in IMU_data[1:]])

        f_param.close()
        f_IMU_data.close()

        # Load parameters
        self.image_H = int(float(param_data[3][14:]))
        self.image_W = int(float(param_data[7][13:]))
        self.rotation_o_x = float(param_data[10][21:])
        self.rotation_o_y = float(param_data[12][21:])
        self.focal_length = float(param_data[4][14:])
        self.pixel_size = float(param_data[8][12:])
        self.exposure = float(param_data[6][15:])
        self.pose = int(float(param_data[9][17:]))
        self.sample_freq = float(param_data[16][20:])
        self.read_out = float(param_data[1][14:])

        # Load inertial sensor data
        self.time_stamp = IMU_data[:, 0].astype(float)
        self.gyro_x = IMU_data[:, 1].astype(float)
        self.gyro_y = IMU_data[:, 2].astype(float)
        self.gyro_z = IMU_data[:, 3].astype(float)
        self.acc_x = IMU_data[:, 4].astype(float)
        self.acc_y = IMU_data[:, 5].astype(float)
        self.acc_z = IMU_data[:, 6].astype(float)

        self.h_num = 10
        self.record_num = self.pose // self.h_num
        self.interval = float(self.exposure / self.pose)
        self.intrinsicMat = np.array([[self.focal_length/self.pixel_size, 0, self.image_W/2 + self.rotation_o_x],
                                     [0, self.focal_length/self.pixel_size, self.image_H/2 + self.rotation_o_y],
                                     [0, 0, 1]])

    def compute_rotations(self):
        """
        :return: rotations List[3x3 ndarray]

        dt = exposure / #pose
        R0 =  Identity

        R_i = [[           1, -omega_zi*dt,  omega_yi*dt],
               [ omega_zi*dt,            1, -omega_xi*dt],
               [-omega_yi*dt,  omega_xi*dt,            1]] * R_i-1

        omega_xi = self.gyro_x[i]
        i = 0, 1, 2, ..., pose
        """

        rotations = []
        dt = self.interval
        R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        for i in range(self.pose+1):
            omega_xt, omega_yt, omega_zt = self.gyro_x[i], self.gyro_y[i], self.gyro_z[i]
            rotation_operator = np.asarray([[1.0, -omega_zt*dt, omega_yt*dt],
                                            [omega_zt*dt, 1.0, -omega_xt*dt],
                                            [-omega_yt*dt,  omega_xt*dt, 1.0]])

            new_R = np.matmul(rotation_operator, R)
            #new_R = new_R / np.linalg.norm(new_R)
            R = new_R

            rotations.append(R)
        return rotations

    def compute_translations(self, rotations):
        """
        :param rotations:
        :return translation: List[3x3 ndarray]

        R: rotations

        T_star_0 = [0, 0, 0]^T
        v0 = [0, 0, 0]^T

        a_ = a[i-1] = [self.acc_x[i-1], self.acc_y[i-1], self.acc_z[i-1]]^T
        a  = a[i]   = [self.acc_x[i], self.acc_y[i], self.acc_z[i]]^T
        invR_ = inv(R[i-1]) = np.linalg.inv(rotations[i-1])
        invR  = inv(R[i])   = np.linalg.inv(rotations[i])

        v_i  = v_i-1 + (invR_ * a_ + invR * a) * dt / 2
        T_star_i = T_star_i-1 + (v_i-1 + v_i) * dt / 2
        T_i = T_star_i - R_i * T_star_0
        """
        translations = []
        dt = self.interval
        T0_star = np.array([0, 0, 0]).reshape(3, 1)
        T = np.array([0, 0, 0]).reshape(3, 1)
        T_star = T0_star
        v = np.array([0, 0, 0]).reshape(3, 1)
        translations.append(T)

        for i in range(1, self.pose+1):
            a_ = np.array([self.acc_x[i-1], self.acc_y[i-1], self.acc_z[i-1]]).reshape(3, 1)
            a  = np.array([self.acc_x[i], self.acc_y[i], self.acc_z[i]]).reshape(3, 1)
            invR_ = np.linalg.inv(rotations[i-1])
            invR  = np.linalg.inv(rotations[i])
            R = rotations[i]
            v_ = v

            v = v_ + (np.matmul(invR_, a_) + np.matmul(invR, a)) * dt / 2
            T_star = T_star + (v_ + v) * dt / 2
            T = T_star - np.matmul(R, T0_star)


            translations.append(T)

        return translations

    def compute_homography(self):
        """
        # Generate N = self.pose Synthetic Homography
        :return: syn_H: Array[self.pose][3x3] perfect synthetic homography


        H_i = K * (R + T * normal_vector ^ T) * inv(K)

        i = 1, 2, 3, ..., pose

        """
        rotations = self.compute_rotations()
        translations = self.compute_translations(rotations)

        # Ri3_0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])  # set all rows of column 3 to 0
        norm_v = np.array([0, 0, 1]).reshape(1, 3)  # norm vector of current plane

        K = self.intrinsicMat
        syn_H = []
        H0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        syn_H.append(H0)

        for i in range(1, self.pose+1):
            R = rotations[i]  # np.matmul(rotations[i], Ri3_0)
            T = np.matmul(translations[i], norm_v)
            E = R + T
            E = E / E[2][2]
            H = np.matmul(np.matmul(K, R+T), np.linalg.inv(K))
            H = H / H[2][2]

            syn_H.append(H)

        return np.asarray(syn_H).astype(float)
