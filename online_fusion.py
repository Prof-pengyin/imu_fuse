import numpy as np
import pylab as pl
import glob
from data.student_utils import StampedData
from pykalman import KalmanFilter

########################################################
# NOTE Read Cam estimateion data [height, roll, pitch]##
########################################################
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
    # cam.data.append(np.array([height, roll, pitch]))
    cam.data.append(pitch)

cam.convert_lists_to_numpy()
observations = cam.data
x = np.arange(len(observations))

########################################################
# NOTE Set Kalman Filter (2 State version)
# NOTE We use two state in current filter 
# NOTE Position and Velocity (more state can better tracking the system)
# NOTE You can also try to just use one state, we will verfiy it below
# The transformation matrix is:
# [1,1]
# [0,1]
# And Coveriance matrix between two states are
# [0.01,    0]
# [0,    0.01]
# NOTE The above paremeter are the inital guess,
# NOTE current kalman filter can use EM algorithm automatically update them.
########################################################
# create a Kalman Filter by hinting at the size of the state and observation
# space.  If you already have good guesses for the initial parameters, put them
# in here.  The Kalman Filter will try to learn the values of all variables.
kf = KalmanFilter(transition_matrices=np.array([[1,1],[0,1]]),
                  transition_covariance=0.01 * np.eye(2))
# parameter updating with EM algorithm, iteration time =1
# Higher iteration time will cause overfit problem.
kf = kf.em(X=observations[:10], n_iter=1)



########################################################
# NOTE Online Updating                                ##
########################################################
states_pred = np.zeros([len(observations),2])
next_covariance = np.zeros([len(observations),2, 2])

for i in range(len(observations)):
    if i == 0:
        states_pred[i,0] = observations[i]
    else:
        states_pred[i], next_covariance[i] = kf.filter_update(states_pred[i-1], next_covariance[i-1], observations[i])


########################################################
# NOTE Plotting Results                               ##
########################################################
pl.figure(figsize=(16, 6))
obs_scatter = pl.scatter(x, observations, marker='x', color='b',
                         label='observations')
position_line = pl.plot(x, states_pred[:, 0],
                        linestyle='-', marker='o', color='r',
                        label='position est.')
velocity_line = pl.plot(x, states_pred[:, 1],
                        linestyle='-', marker='o', color='g',
                        label='velocity est.')
pl.legend(loc='lower right')
pl.xlim(left=0, right=x.max())
pl.xlabel('time')
pl.title("Two state version Kalman fitler for Pitch Smoothing")
pl.show()
print ("Sum of Residual error for Two state Kalman is:")
print (np.sum(states_pred[:,0]-observations))



########################################################
# NOTE Set Kalman Filter (1 State version)
########################################################
kf = KalmanFilter(transition_matrices=np.array([1]),
                  transition_covariance=0.01 * np.eye(1))
kf = kf.em(X=observations[:10], n_iter=1)



########################################################
# NOTE Online Updating                                ##
########################################################
states_pred = np.zeros([len(observations),1])
next_covariance = np.zeros([len(observations),1])

for i in range(len(observations)):
    if i == 0:
        states_pred[i] = observations[i]
    else:
        states_pred[i], next_covariance[i] = kf.filter_update(states_pred[i-1], next_covariance[i-1], observations[i])



########################################################
# NOTE Plotting Results                               ##
########################################################
pl.figure(figsize=(16, 6))
obs_scatter = pl.scatter(x, observations, marker='x', color='b',
                         label='observations')
position_line = pl.plot(x, states_pred[:],
                        linestyle='-', marker='o', color='r',
                        label='position est.')
pl.legend(loc='lower right')
pl.xlim(left=0, right=x.max())
pl.xlabel('time')
pl.title("One state version Kalman fitler for Pitch Smoothing")
pl.show()
print ("Sum of Residual error for One state Kalman is:")
print (np.sum(states_pred-observations))

print ("Haha! Two State is more accurate!!")