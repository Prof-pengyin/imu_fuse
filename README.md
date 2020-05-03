# EKF Fusion for IMU and Cam pitch

## To use the code
```python
python es_ekf.py
```

## Parameters

```
### TODO Cam is your data
cam = data['gnss']

### TODO: Assume you only have pitch data
cam.data = cam.data[:,-1]

### TODO: Your Cam convariance is defined here
var_cam = 0.1
```
