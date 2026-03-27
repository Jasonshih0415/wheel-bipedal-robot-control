import numpy as np
import mujoco 
import mujoco.viewer
import time
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy.linalg
from pynput import keyboard
import pinocchio as pin
import math
import sys

""" by mujoco_viewer, check the xml 
<<------------- Actuator ------------->>
actuator_index: 0 , name: L_thigh
actuator_index: 1 , name: L_calf
actuator_index: 2 , name: L_wheel
actuator_index: 3 , name: R_thigh 大腿
actuator_index: 4 , name: R_calf 小腿
actuator_index: 5 , name: R_wheel 

<<------------- Sensor ------------->>
sensor_index: 0 , name: L_thigh_pos , dim: 1
sensor_index: 1 , name: L_calf_pos , dim: 1
sensor_index: 2 , name: L_wheel_pos , dim: 1
sensor_index: 3 , name: R_thigh_pos , dim: 1
sensor_index: 4 , name: R_calf_pos , dim: 1
sensor_index: 5 , name: R_wheel_pos , dim: 1
sensor_index: 6 , name: L_thigh_vel , dim: 1
sensor_index: 7 , name: L_calf_vel , dim: 1
sensor_index: 8 , name: L_wheel_vel , dim: 1
sensor_index: 9 , name: R_thigh_vel , dim: 1
sensor_index: 10 , name: R_calf_vel , dim: 1
sensor_index: 11 , name: R_wheel_vel , dim: 1
sensor_index: 12 , name: L_thigh_torque , dim: 1
sensor_index: 13 , name: L_calf_torque , dim: 1
sensor_index: 14 , name: L_wheel_torque , dim: 1
sensor_index: 15 , name: R_thigh_torque , dim: 1
sensor_index: 16 , name: R_calf_torque , dim: 1
sensor_index: 17 , name: R_wheel_torque , dim: 1
sensor_index: 18 , name: imu_quat , dim: 4
sensor_index: 22 , name: imu_gyro , dim: 3
sensor_index: 25 , name: imu_acc , dim: 3
sensor_index: 28 , name: frame_pos , dim: 3
sensor_index: 31 , name: frame_lin_vel , dim: 3
sensor_index: 34 , name: frame_ang_vel , dim: 3

wheel pos :rad
wheel vel :rad/s
"""

# Robotic Parameters from urdf
M = 6.499               # Total mass(From pinocchio) [kg]
m = 0.2805              # Single wheel mass [kg]    
d = 0.3291              # Wheel track [m] #兩輪之間距離
r = 0.07                # Wheel radius [m]
g = 9.81                # Gravity [m/s^2]

#control the chassis
def LQR(current_l): 
    #current_l: 擺桿質心到轉軸距離(機器人質心到底盤馬達轉軸的距離)
    I_wheel  = (1/2)*m*(r**2)
    J_p = (1/3)*M*(current_l**2) #對z軸轉動慣量,俯仰方向
    J_delta  = (1/12)*M*(d**2) #對y軸轉動慣量,左右方向

    term1 = ( J_p + M * (current_l**2))
    term2 = ((2 * m) + (2 * I_wheel / (r**2)))
    term3 =  J_p * M
    Qeq = (term1 * term2) + term3

    A23 = - ( (M**2) * (current_l**2) * g ) / Qeq   
    A43 = ( M * current_l * g * (M+term2) ) / Qeq
            
    A = np.array([
        [0, 1, 0,   0, 0, 0],
        [0, 0, A23, 0, 0, 0],
        [0, 0, 0,   1, 0, 0],
        [0, 0, A43, 0, 0, 0],
        [0, 0, 0,   0, 0, 1],
        [0, 0, 0,   0, 0, 0]
    ])

    #穩定性分析 pos value, not stable system, need control
    #eigenvalue = np.linalg.eig(A)
    #print(f"eigenvalue:{eigenvalue}") 

    term4 = r * ( (m * d) + ((I_wheel * d)/(r**2)) + (2 * J_delta / d)  ) 
    B21 = ( J_p + (M * (current_l**2)) + (M * current_l * r) ) / (Qeq * r)
    B22 = B21
    B41 = -( ( (M * current_l) / r ) + (M+term2) ) / Qeq
    B42 = B41
    B61 = -1 / term4 #delta = (accel_r - accel_l)/d, set turn left be positive
    B62 = 1 / term4 #in mujoco world z axis is going up(turm left be positive)

    B = np.array([ 
        [0,     0],
        [B21, B22],
        [0,     0],
        [B41, B42],
        [0,     0],
        [B61, B62]
    ])

    Q = np.diag([1.0, 5.0, 300.0, 1.0, 1.0, 1.0]) #x, dx, p, dp, delta, d(delta)
    R = np.array([[1.0, 0.0], [0.0, 1.0]])  #TL, TR 

    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P #No.24 (5)
    return K

#control posture 
def PD_control(target_q,q,kp,target_dq,dq,kd):
    tau = (target_q - q)*kp + (target_dq - dq)*kd
    return tau

def get_l(model,data,q):
    pin.centerOfMass(model, data, q)
    pin.forwardKinematics(model, data, q) #joint
    pin.updateFramePlacements(model, data) #frame base on joint
    com_pos = data.com[0]
    left_wheel_idx = model.getFrameId("L_calf2wheel")
    wheel_pos = data.oMf[left_wheel_idx].translation
    #print(f"機器人當前的質心位置 (x, y, z): {com_pos}")
    #print(f"機器人輪子位置 (x, y, z): {wheel_pos}")
    #print(com_pos[0] - wheel_pos[0])
    l = ((com_pos[0]-wheel_pos[0])**2 + (com_pos[2]-wheel_pos[2])**2)**0.5
    return l

def quat2Euler(quat):
    w,x,y,z = quat
    Pitch = math.asin(2*(w*y -x*z))
    Yaw = math.atan2(2*(w*z+x*y),1-2*(y**2+z**2))
    return Pitch, Yaw

def get_robot_vel(data,r):
    r_wheel_w = data.sensordata[8]
    l_wheel_w = data.sensordata[11]
    avg_wheel_w = (r_wheel_w+l_wheel_w)/2.0
    avg_wheel_vel = avg_wheel_w *r
    return avg_wheel_vel

def imu2com_error(imu,com,wheel): #IMU與質心的角度誤差
    ax, ay, az = imu
    bx, by, bz = com
    cx, cy, cz = wheel
    com_angle = math.atan2(bx-cx,bz-cz)
    imu_angle = math.atan2(ax-cx,az-cz)
    angle_error = com_angle-imu_angle
    return angle_error

#PD setting
target_q = np.array([0.86,-1.5,0,0.86,-1.5,0]) #check in robot viewer
target_dq = np.array([0,0,0,0,0,0], dtype=np.float64) #stable in one place
kp = np.array([50, 50 ,0, 50, 50, 0])
kd = np.array([10, 10, 0, 10, 10, 0])

#posture setting for pin
q_pin = np.array([
    0.86,      # L_hip2thigh
    -1.5,      # L_thigh2calf
    0.0, 1.0, # L_calf2wheel 
    0.86,      # R_hip2thigh
    -1.5,     # R_thigh2calf
    0.0, 1.0  # R_calf2wheel 
])

#load file
modelPath_xml = './crazydog_urdf/urdf/scene.xml'
model = mujoco.MjModel.from_xml_path(modelPath_xml)
data = mujoco.MjData(model)

modelPath_urdf = "./crazydog_urdf/urdf/crazydog_urdf.urdf"
model_pin = pin.buildModelFromUrdf(modelPath_urdf)
data_pin = model_pin.createData()
l = get_l(model_pin,data_pin,q_pin)
distance =0

#read arg
static_balance = False
if "static_balance=True" in sys.argv:
    static_balance = True

#logging
log_time = []
log_thigh = []
log_calf = []
log_pitch = []
log_tau_r =[]
log_tau_l =[]
log_yaw = []
log_distance = []
log_vel = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    target_body_name = "base_link" 
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = body_id
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90

    while viewer.is_running():

        step_start = time.time()
        #state 
        quat = data.sensordata[18:22]
        Pitch, Yaw = quat2Euler(quat)
        gyro_y = data.sensordata[23]
        gyro_z = data.sensordata[24]
        robot_vel = get_robot_vel(data,r)
        dt = model.opt.timestep
        distance += robot_vel * dt    

        target_state = np.array([
            0.0,       # target_distance
            0.0,       # target_vel
            0.0,       # target_pitch
            0.0,       # target_gyro_y
            0.0,       # target_yaw
            0.0        # target_gyro_z
        ])

        if static_balance == False:
            target_state[0] = distance
            target_state[1] = 0.2
            
        state = np.array([
            distance, 
            robot_vel ,
            Pitch ,
            gyro_y,
            Yaw,
            gyro_z
        ])

        #control
        k = LQR(l)
        u = -k @ (state-target_state) 
        tau = PD_control(target_q, data.sensordata[:6],kp, target_dq, data.sensordata[6:12],kd)
        tau[2] = u[0] 
        tau[5] = u[1] 
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        viewer.sync()

        #log 
        log_time.append(data.time)
        log_thigh.append(data.sensordata[0]) 
        log_calf.append(data.sensordata[1]) 
        log_pitch.append(Pitch)
        log_yaw.append(Yaw)
        log_tau_l.append(tau[2])
        log_tau_r.append(tau[5])
        log_distance.append(distance)
        log_vel.append(robot_vel)
        com_pos = data.subtree_com[0]

        #imu and center of mass error
        wheel_pos = data.xpos[8]
        imu_pos = data.sensordata[28:31]
        angle_error = imu2com_error(imu_pos,com_pos,wheel_pos)

        # Loop timing
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

plt.figure(figsize=(12,10))

plt.subplot(2,3,1)
plt.plot(log_time,log_pitch,label='rad')
plt.title('pitch')
plt.axhline(target_state[2],color ='red', linestyle = '--')
plt.grid(True)
plt.legend()

plt.subplot(2,3,2)
plt.plot(log_time,log_vel,label='m')
plt.title('vel_robot')
plt.grid(True)
plt.legend()

plt.subplot(2,3,3)
plt.plot(log_time,log_tau_l,label='N-m')
plt.title('tau_l_wheel')
plt.grid(True)
plt.legend()

plt.subplot(2,3,4)
plt.plot(log_time,log_tau_r,label='N-m')
plt.title('tau_r_wheel')
plt.grid(True)
plt.legend()

plt.subplot(2,3,5)
plt.plot(log_time,log_thigh,label='rad')
plt.title('thigh_angle')
plt.axhline(q_pin[0],color ='red', linestyle = '--')
plt.grid(True)
plt.legend()

plt.subplot(2,3,6)
plt.plot(log_time,log_calf,label='N-m')
plt.title('calf_angle')
plt.axhline(q_pin[1],color ='red', linestyle = '--')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
