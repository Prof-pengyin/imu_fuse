# Main code for the Coursera SDC Course 2 final project
#
# Author: Trevor Ablett
# University of Toronto Institute for Aerospace Studies

import pickle
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from rotations import Quaternion, skew_symmetric
from data.student_utils import StampedData
sys.path.append('./data')

#### 1. Data ###################################################################################

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For part 3, you will use p3_data.pkl.
################################################################################################
with open('data/p1_data.pkl', 'rb') as file:
    data = pickle.load(file)

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################

# TODO Read IMU data
#! IMU data come with 10 Hz
imu_data = glob.glob('data/imu/*')
imu_data.sort()
imu_f = StampedData()
imu_w = StampedData()
for im_p in imu_data:
    f = open(im_p, "r")
    data = f.read()

    angx = (float)(data.split('\n')[0].split(' ')[-1])
    angy = (float)(data.split('\n')[1].split(' ')[-1])
    angz = (float)(data.split('\n')[2].split(' ')[-1])
    accx = (float)(data.split('\n')[3].split(' ')[-1])
    accy = (float)(data.split('\n')[4].split(' ')[-1])
    accz = (float)(data.split('\n')[5].split(' ')[-1])

    imu_f.data.append(np.array([accx, accy, accz]))
    imu_w.data.append(np.array([angx, angy, angz]))

    stamp_p = im_p.split('/')[-1].split('.')[-2].split('_')
    stamp = (float)(stamp_p[-2]) + (float)(stamp_p[-1])/1000000000.
    imu_f.t.append(stamp)
    imu_w.t.append(stamp)

imu_f.convert_lists_to_numpy()
imu_w.convert_lists_to_numpy()

# TODO Read Cam estimateion data [height, roll, pitch]
#! cam data come with 20 Hz, so we take half
#? what's the angle unit
cam_data = glob.glob('data/ground_est_raw/*')
cam_data.sort()
cam = StampedData()
for cam_p in cam_data:
    f = open(cam_p, "r")
    data = f.read()

    stamp_p = cam_p.split('/')[-1].split('.')[-2].split('_')
    if ((int)(stamp_p[-1])%100000000) != 0:
        continue
    stamp = (float)(stamp_p[-2]) + (float)(stamp_p[-1])/1000000000.
    cam.t.append(stamp)

    pitch  = (float)(data.split('\n')[0].split(' ')[-1])*np.pi/180
    roll   = (float)(data.split('\n')[1].split(' ')[-1])*np.pi/180
    height = (float)(data.split('\n')[2].split(' ')[-1])*np.pi/180

    cam.data.append(np.array([height, roll, pitch]))

cam.convert_lists_to_numpy()

### TODO: Your Cam convariance is defined here

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
var_imu_f = 0.01
var_imu_w = 0.01
var_cam = 0.001
gravity = 9.81

################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -gravity])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian
q_jac = np.zeros([3, 9])
# q_jac[:, 6:] = np.eye(3)  # measurement model jacobian
q_jac[0, 2] = 1  # 0 input is relative to 2 state (height)
q_jac[1, 6] = 1  # 1 input is relative to 6 state (roll)
q_jac[2, 7] = 1  # 2 input is relative to 7 state (pitch)

#### 3. Initial Values #########################################################################

################################################################################################
# Let's set up some initial values for our ES-EKF solver.
################################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

# Set initial values
p_est[0] = np.array([0,0,0])
v_est[0] = np.array([0,0,0])
#! Initialization q with roll, pitch from cam
q_est[0] = Quaternion(euler=np.array([cam.data[0][1], cam.data[0][2], 0.0])).to_numpy()
p_cov[0] = np.eye(9)  # covariance of estimate

#### 4. Measurement Update #####################################################################

################################################################################################
# Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
# a function for it.
################################################################################################
def measurement_update_position(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # 3.1 Compute Kalman Gain
    R_cov = sensor_var * np.eye(3)
    K = p_cov_check.dot(h_jac.T.dot(np.linalg.inv(h_jac.dot(p_cov_check.dot(h_jac.T)) + R_cov)))

    # 3.2 Compute error state
    delta_x = K.dot(y_k - p_check)

    # 3.3 Correct predicted state
    p_check = p_check + delta_x[:3]
    v_check = v_check + delta_x[3:6]
    q_check = Quaternion(axis_angle = delta_x[6:]).quat_mult(q_check)

    # 3.4 Compute corrected covariance
    p_cov_check = (np.eye(9) - K.dot(h_jac)).dot(p_cov_check)

    return p_check, v_check, q_check, p_cov_check

def measurement_update_rotation(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # 3.1 Compute Kalman Gain
    R_cov = sensor_var * np.eye(3)
    K = p_cov_check.dot(q_jac.T.dot(np.linalg.inv(q_jac.dot(p_cov_check.dot(q_jac.T)) + R_cov)))

    # 3.2 Compute error state
    q_rpy = Quaternion(w=q_check[0], x=q_check[1], y=q_check[2], z=q_check[3]).to_euler()

    ## TODO: Here we only update yaw angle, and y_k is your current observation
    #! est_k = [est_height, est_roll, est_pitch]
    #! y_k = [cam_height, cam_roll, cam_pitch]
    est_k = np.array([p_check[2], q_rpy[0], q_rpy[1]])
    delta_x = K.dot(y_k - est_k)

    # 3.3 Correct predicted state
    p_check = p_check + delta_x[:3]
    v_check = v_check + delta_x[3:6]
    q_check = Quaternion(axis_angle = delta_x[6:]).quat_mult(q_check)

    # 3.4 Compute corrected covariance
    p_cov_check = (np.eye(9) - K.dot(q_jac)).dot(p_cov_check)

    return p_check, v_check, q_check, p_cov_check

#### 5. Main Filter Loop #######################################################################

################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################
for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # 1. Update nominal state with IMU inputs
    Rotation_Mat = Quaternion(*q_est[k - 1]).to_mat()
    p_est[k] = p_est[k - 1] + delta_t * v_est[k - 1] + 0.5 * (delta_t ** 2) * (Rotation_Mat.dot(imu_f.data[k - 1]) + g)
    v_est[k] = v_est[k - 1] + delta_t * (Rotation_Mat.dot(imu_f.data[k - 1]) - g)
    q_est[k] = Quaternion(euler = delta_t * imu_w.data[k - 1]).quat_mult(q_est[k - 1])

    # 1.1 Linearize Motion Model and compute Jacobians
    F = np.eye(9)
    imu = imu_f.data[k - 1].reshape((3, 1))
    F[0:3, 3:6] = delta_t * np.eye(3)
    F[3:6, 6:9] = Rotation_Mat.dot(-skew_symmetric(imu)) * delta_t

    # 2. Propagate uncertainty
    Q = np.eye(6)
    Q[0:3, 0:3] = var_imu_f * Q[0:3, 0:3]
    Q[3:6, 3:6] = var_imu_w * Q[3:6, 3:6]
    Q = (delta_t ** 2) * Q #Integration acceleration to obstain Position
    p_cov[k] = F.dot(p_cov[k - 1]).dot(F.T) + l_jac.dot(Q).dot(l_jac.T)

    # 3. Check availability of GNSS and LIDAR measurements
    for i in range(len(cam.t)):
        if abs(cam.t[i] - imu_f.t[k]) < 0.01:
            # Assume 
            p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update_rotation(var_cam, p_cov[k],
                                                    cam.data[i], p_est[k], v_est[k], q_est[k])

#### 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
# ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Estimated Trajectory')
ax.legend()
ax.set_zlim(-1, 5)
plt.show()

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty.
################################################################################################
# error_fig, ax = plt.subplots(2, 3)
# error_fig.suptitle('Error plots')
# num_gt = gt.p.shape[0]
# p_est_euler = []

# # Convert estimated quaternions to euler angles
# for q in q_est:
#     p_est_euler.append(Quaternion(*q).to_euler())
# p_est_euler = np.array(p_est_euler)

# # Get uncertainty estimates from P matrix
# p_cov_diag_std = np.sqrt(np.diagonal(p_cov, axis1=1, axis2=2))

# titles = ['x', 'y', 'z', 'x rot', 'y rot', 'z rot']
# for i in range(3):
#     # ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
#     ax[0, i].plot(range(num_gt), 3 * p_cov_diag_std[:num_gt, i], 'r--')
#     ax[0, i].plot(range(num_gt), -3 * p_cov_diag_std[:num_gt, i], 'r--')
#     ax[0, i].set_title(titles[i])

# for i in range(3):
#     # ax[1, i].plot(range(num_gt), gt.r[:, i] - p_est_euler[:num_gt, i])
#     ax[1, i].plot(range(num_gt), 3 * p_cov_diag_std[:num_gt, i+6], 'r--')
#     ax[1, i].plot(range(num_gt), -3 * p_cov_diag_std[:num_gt, i+6], 'r--')
#     ax[1, i].set_title(titles[i+3])
# plt.show()