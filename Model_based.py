import simcontrol
from simple_pid import PID
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

port = 25556
control_interval = 0.002 # 控制频率，500Hz
duration = 10      # 运行时间
iterations = 1      # 后续可能的ILC会用到，迭代次数
control_min = 0.0   # min和max分别代表这PWM的占空比
control_max = 1.0


# 定义无人机的物理参数
pa.m = 1


controller = simcontrol.Controller("localhost", port) # 从仿真器那里获取控制器接口，输入和输出

# setup controls
control_names = ['fl', 'fr', "bl", "br"] # 分别代表着前左、前右、后左、后右
control_indices = []
controls = {}
# 从仿真器那里获取上面四个名字所对应的index，然后把index放在control_indices，每个index需要返回的PWM值放在controls中。
for name in control_names:
    idx = controller.get_control_index(name)
    if idx is None:
        print(f'Control not found: {name}')
        continue

    control_indices.append(idx)
    controls[idx] = 0.0 # scale

# define mapping from observation to targets (pid input)
def get_targets(state : tuple) -> tuple:
    force_up = state[2]
    torque_roll = state[7]
    torque_pitch = state[6]
    torque_yaw = state[8]
    return (force_up, torque_roll, torque_pitch, torque_yaw)


# setup mapping between targets and controls
front_len = 1
behind_len = 1
controls_to_targets = np.array(((front_len, front_len, behind_len, behind_len), (front_len, -1 * front_len, behind_len, -1 * behind_len), (front_len, front_len, -1 * behind_len, -1 * behind_len), (front_len, -1 * front_len, -1 * behind_len, behind_len))) #pitch is not balanced due to propeller position difference between front and back
targets_to_control = np.linalg.inv(controls_to_targets)
# print(f"target_2_control: {targets_to_control}")

# setup pid controller for targets
Kp = (1, 1, 5, 5, 5, 5)     # x, y, z, roll, pitch, yaw
Kd = np.dot(Kp, 0.1)

def get_desired(trajectory_flag, t):
    if trajectory_flag == 1:
        desired = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    elif trajectory_flag == 2:
        xd = 1 * np.exp(t / 20) * np.sin(t) + 0.2
        yd = np.exp(t / 20) * np.cos(t) - 0.8
        zd = 0.5 * t
        d_xd = 1 / 20 * np.exp(t / 20) * np.sin(t) + 1 * np.exp(t / 20) * np.cos(t)
        d_yd = 0.05 * np.exp(t / 20) * np.cos(t) - np.exp(t / 20) * np.sin(t)
        d_zd = 0.5
        desired = (xd, yd, zd, d_xd, d_yd, d_zd, 0, 0, 0, 0, 0, 0)
    return desired
# pid_tunes = 1 * ((1, 0, 0.0), (1, 0, 0.0), (1, 0, 0.0), (1, 0, 0.0))  # Kp, Ki, Kd
# pids = []
# for tune in pid_tunes:
#     pid = PID(tune[0], tune[1], tune[2], setpoint=0.0, sample_time=None)
#     pids.append(pid)

# pids[0].setpoint = 1  # 将第一个PID闭环系统的desired值设为0.5
# pids[2].setpoint = 4.5992468 / 180 * np.pi

# get time step
time_step = controller.get_time_step()
steps_per_control = max(round(control_interval / time_step), 1)
print(f'Time step: {time_step}, steps per control: {steps_per_control}') # 一次控制操作对应的仿真步数
dt = time_step * steps_per_control  # 这里为什么不直接用 control_interval 呢？

# simulation loop
for i in range(iterations):
    x = [] # 存储时间序列
    thrust = []
    state = []
    desired_states = []
    for i in range(len(control_names)):
        thrust.append([]) # 存储发送到每个旋翼的PWM值，以便画图
    for i in range(0, 12):
        state.append([])  # 用来存储状态信息以便画图
    for i in range(0, 12):
        desired_states.append([])  # 用来存储状态信息以便画图

    #controller.reset()
    t = 0.0
    while t < duration:
        reply = controller.simulate(steps_per_control, controls) # 这里的 controls 和 thrust 的区别是什么？
        if reply is None:
            print('Simulation failed, finish iteration early.')
            break

        t += dt
        x.append(t) # 存储当前的时间序列
        print(f'position: ({reply[0]:.5f}, {reply[1]:.5f}, {reply[2]:.5f}), velocity: ({reply[3]:.5f}, {reply[4]:.5f}, {reply[5]:.5f}), rotation: ({reply[6]:.5f}, {reply[7]:.5f}, {reply[8]:.5f}), angular velocity: ({reply[9]:.5f}, {reply[10]:.5f}, {reply[11]:.5f})')  # position: x, y, z;  rotation: roll, pitch, yaw

        for i in range(0, 12):
            state[i].append(reply[i])

        target_input = get_targets(reply)
        target_output = np.zeros(len(target_input)) # 返回一个和target_input同维度的全零数组

        trajectory_flag = 2
        desired_state = get_desired(trajectory_flag, t)
        for i in range(0, 12):
            desired_states[i].append(desired_state[i])

        temp_x = Kp[0] * (desired_state[0] - reply[0]) + Kd[0] * (desired_state[3] - reply[3])
        temp_y = Kp[1] * (desired_state[1] - reply[1]) + Kd[1] * (desired_state[4] - reply[4])
        target_output[0] = Kp[2] * (desired_state[2] - reply[2]) + Kd[2] * (desired_state[5] - reply[5])
        target_output[1] = Kp[3] * (desired_state[7] - reply[7] + temp_x) + Kd[3] * (desired_state[10] - reply[10])
        target_output[2] = Kp[4] * (desired_state[6] - reply[6] - temp_y) + Kd[4] * (desired_state[9] - reply[9])
        target_output[3] = Kp[5] * (desired_state[8] - reply[8]) + Kd[5] * (desired_state[11] - reply[11])
        # print(f"At time {x}, y is {y}")

        # new_controls = np.matmul(targets_to_control, target_output) / 1.0 # 通过mapping matrix的逆，计算每个电机的PWM输入
        new_controls = np.dot(targets_to_control, target_output) / 1.0
        #new_controls = np.ones(len(control_hashes)) * control_target
        print(f"Control_input is: {target_output} -> {new_controls}.")
        for i in range(len(control_indices)):  # 将计算得到的PWM通过饱和模块
            hash = control_indices[i]
            next = max(control_min, min(new_controls[i], control_max))
            if t < 0:
                if i == 0:
                    next = 1.0
                elif i == 1:
                    next = 1.0
                elif i == 2:
                    next = 0.6
                elif i == 3:
                    next = 0.6

            controls[control_indices[i]] = next  # 更新 controls 的值，是返回Simulator的值
            thrust[i].append(next)   # 记录，新增当前的PWM输入，以便画图

        print(f"Now is {t}, Control_PWM is: {controls}.")

    fig, axes = plt.subplots(2, 2)

    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    color1 = 'tab:red'
    ax1.set_xlabel('time(s)')
    ax1.set_ylabel('propeller RPM', color=color1)
    ax1.plot(x, thrust[0], label='front left')
    ax1.plot(x, thrust[1], label='front right')
    ax1.plot(x, thrust[2], label='back left')
    ax1.plot(x, thrust[3], label='back right')
    ax1.tick_params(axis='y', labelcolor=color1) # 将 y轴刻度 的颜色设置为color1.
    ax1.set_ylim(-0.25, 1.25)
    ax1.legend()
    ax1.set_title('Propeller PWM vs. Time')

    # ax2 = ax1.twinx()

    color2 = 'tab:brown'
    ax2.set_xlabel('time(s)')
    ax2.set_ylabel('velocity(m/s)', color=color2)
    ax2.plot(x, state[2], label='vertical position', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.25, 8)
    ax2.legend()
    ax2.set_title('vertical velocity vs. Time')


    # color1 = 'tab:red'
    # ax3.set_xlabel('time(s)')
    # ax3.set_ylabel('propeller RPM', color=color1)
    # ax3.plot(x, thrust[0], label='front left')
    # ax3.plot(x, thrust[1], label='front right')
    # ax3.plot(x, thrust[2], label='back left')
    # ax3.plot(x, thrust[3], label='back right')
    # ax3.tick_params(axis='y', labelcolor=color1)
    # ax3.set_ylim(-3000, 3000)
    # ax3.legend()

    # ax4 = ax3.twinx()

    color2 = 'tab:olive'
    color3 = 'tab:cyan'
    color4 = 'tab:gray'
    ax3.set_xlabel('time(s)')
    ax3.set_ylabel('attitude(degrees)', color=color2)
    ax3.plot(x, state[6], label='pitch', color=color2)
    ax3.plot(x, state[7], label='roll', color=color3)
    ax3.plot(x, state[8], label='yaw', color=color4)
    ax3.tick_params(axis='y', labelcolor=color2)
    ax3.set_ylim(-3, 3)
    ax3.legend()
    ax3.set_title('Attitude vs. Time')

    color2 = 'tab:olive'
    color3 = 'tab:cyan'
    color4 = 'tab:gray'
    ax4.set_xlabel('time(s)')
    ax4.set_ylabel('position(m)', color=color2)
    ax4.plot(x, state[0], label='position x', color=color2)
    ax4.plot(x, state[1], label='position y', color=color3)
    ax4.plot(x, state[2], label='position z', color=color4)
    ax4.tick_params(axis='y', labelcolor=color2)
    ax4.set_ylim(-2, 5)
    ax4.legend()
    ax4.set_title('Position vs. Time')

    fig.tight_layout()
    fig.set_size_inches(15.0, 5.0)
    # plt.show()


    fig2 = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(desired_states[0], desired_states[1], desired_states[2], s=5, c=desired_states[2])
    ax1.plot3D(state[0], state[1], state[2], 'gray')
    plt.show()


    def update(i):
        label = 'timestep {0}'.format(i)

        # 更新直线和x轴（用一个新的x轴的标签）。
        # 用元组（Tuple）的形式返回在这一帧要被重新绘图的物体
        horiAngle = (45 + 3 * i) % 360  # transition(i, 40)
        vertAngle = (45 + 3 * i) % 360  # transition(i, 40)
        print(label, horiAngle, vertAngle)
        ax1.view_init(vertAngle, horiAngle)
        filename = 'animation/' + str('%03d' % i) + '.png'
        plt.savefig(filename, dpi=96)
        return ax1


    anim = FuncAnimation(fig2, update, frames=np.arange(0, 120), interval=200)
    anim.save('animation/line.gif', dpi=120, writer='imagemagick')

controller.close()