# EKF Fusion for IMU and Cam pitch

## EKF Fusion for IMU and Camera
```python
python imu_fusion.py
```

## Parameters

Your data will be load in the following way:

```python
cam_data = glob.glob('data/ground_est_raw/*')
imu_data = glob.glob('data/imu/*')
```

Your belief on the camera will be:
```python
var_cam = 0.1
```
Right now the data is updated based on all your cam inputs: **[height, roll, pitch]**.

![alt text](https://github.com/maxtomCMU/imu_fuse/blob/master/data/traj.png "Logo Title Text 1")

## EKF online filter only for Camera PITCH
```python
python online_fusion.py
```
You will get:
```bash
Sum of Residual error for Two state Kalman is:
-0.00046486205076725143
Sum of Residual error for One state Kalman is:
-2.046092015768805
```


![alt text](https://github.com/maxtomCMU/imu_fuse/blob/master/data/2s_ekf.png "Logo Title Text 1")

![alt text](https://github.com/maxtomCMU/imu_fuse/blob/master/data/1s_ekf.png "Logo Title Text 1")