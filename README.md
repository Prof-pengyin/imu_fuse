# EKF Fusion for IMU and Cam pitch

## To use the code
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

