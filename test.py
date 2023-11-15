import simcontrol
from simple_pid import PID
import numpy as np
import matplotlib.pyplot as plt

port = 25556
control_interval = 0.002
<<<<<<< HEAD
duration = 2
=======
duration = 10
>>>>>>> 43ed930 (Flare_UAV_Control)
iterations = 1
control_min = 0.0
control_max = 1.0

controller = simcontrol.Controller("localhost", port)

# setup controls
control_names = ['fl', 'fr', "bl", "br"]
control_indices = []
controls = {}
for name in control_names:
    idx = controller.get_control_index(name)
    if idx is None:
        print(f'Control not found: {name}')
        continue

    control_indices.append(idx)
    controls[idx] = 0.0

# define mapping from observation to targets (pid input)
def get_targets(state : tuple) -> tuple:
    force_up = state[1]
    torque_roll = state[3]
    torque_pitch = state[4]
    torque_yaw = state[5]
    return (force_up, torque_roll, torque_pitch, torque_yaw)

# setup mapping between targets and controls
controls_to_targets = np.array(((1, 1, 1, 1), (1, -1, 1, -1), (0.9, 0.9, -1.1, -1.1), (1, -1, -1, 1))) #pitch is not balanced due to propeller position difference between front and back
targets_to_control = np.linalg.inv(controls_to_targets)

# setup pid controller for targets
pid_tunes = ((0.5, 0.5, 0.0), (0.01, 0.01, 0.0), (0.01, 0.01, 0.0), (0.01, 0.01, 0.0))  # Kp, Ki, Kd
pids = []
for tune in pid_tunes:
    pid = PID(tune[0], tune[1], tune[2], setpoint=0.0, sample_time=None)
    pids.append(pid)

pids[0].setpoint = 0.5
#pids[2].setpoint = 4.5992468

# get time step
time_step = controller.get_time_step()
steps_per_control = max(round(control_interval / time_step), 1)
print(f'Time step: {time_step}, steps per control: {steps_per_control}')
dt = time_step * steps_per_control

# simulation loop
for i in range(iterations):
    x = []
    y = []
    thrust = []
    for i in range(len(pid_tunes)):
        y.append([])
    for i in range(len(control_names)):
        thrust.append([])

    #controller.reset()
    t = 0.0
    while t < duration:
        reply = controller.simulate(steps_per_control, controls)
        if reply is None:
            print('Simulation failed, finish iteration early.')
            break

        t += dt
        x.append(t)
        print(f'position: ({reply[0]:.5f}, {reply[1]:.5f}, {reply[2]:.5f}), velocity: ({reply[3]:.5f}, {reply[4]:.5f}, {reply[5]:.5f}), rotation: ({reply[6]:.5f}, {reply[7]:.5f}, {reply[8]:.5f}), angular velocity: ({reply[9]:.5f}, {reply[10]:.5f}, {reply[11]:.5f})')

        target_input = get_targets(reply)
        target_output = np.zeros(len(target_input))
        for i in range(len(target_input)):
            y[i].append(target_input[i])
            #if i == 2: continue # currently ignore yaw
            target_output[i] = pids[i](target_input[i], dt)
        
        new_controls = np.matmul(targets_to_control, target_output)
        #new_controls = np.ones(len(control_hashes)) * control_target
        print(new_controls)
        for i in range(len(control_indices)):
            hash = control_indices[i]
            next = max(control_min, min(new_controls[i], control_max))
            # if t < 0.5:
            #     next = 0.0
            controls[control_indices[i]] = next
            thrust[i].append(next)

    fig, axes = plt.subplots(1, 2)

    ax1 = axes[0]
    ax3 = axes[1]

    color1 = 'tab:red'
    ax1.set_xlabel('time(s)')
    ax1.set_ylabel('propeller RPM', color=color1)
    ax1.plot(x, thrust[0], label='front left')
    ax1.plot(x, thrust[1], label='front right')
    ax1.plot(x, thrust[2], label='back left')
    ax1.plot(x, thrust[3], label='back right')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(-3000, 3000)
    ax1.legend()

    ax2 = ax1.twinx()

    color2 = 'tab:brown'
    ax2.set_ylabel('velocity(m/s)', color=color2)
    ax2.plot(x, y[0], label='vertical velocity', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(-0.75, 0.75)
    ax2.legend()

    ax1.set_title('Propeller RPM and vertical velocity vs. Time')

    color1 = 'tab:red'
    ax3.set_xlabel('time(s)')
    ax3.set_ylabel('propeller RPM', color=color1)
    ax3.plot(x, thrust[0], label='front left')
    ax3.plot(x, thrust[1], label='front right')
    ax3.plot(x, thrust[2], label='back left')
    ax3.plot(x, thrust[3], label='back right')
    ax3.tick_params(axis='y', labelcolor=color1)
    ax3.set_ylim(-3000, 3000)
    ax3.legend()

    ax4 = ax3.twinx()

    color2 = 'tab:olive'
    color3 = 'tab:cyan'
    color4 = 'tab:gray'
    ax4.set_ylabel('attitude(degrees)', color=color2)
    ax4.plot(x, y[1], label='pitch', color=color2)
    ax4.plot(x, y[2], label='yaw', color=color3)
    ax4.plot(x, y[3], label='roll', color=color4)
    ax4.tick_params(axis='y', labelcolor=color2)
    ax4.set_ylim(-30, 30)
    ax4.legend()

    ax3.set_title('Propeller RPM and attitude vs. Time')

    fig.tight_layout()
    fig.set_size_inches(15.0, 5.0)
    plt.show()


controller.close()