import numpy as np
import copy as cp
from .Connect_file import Robot_control
from gym import spaces
from .Fuzzy_rules import fuzzy_control


class env_assembly_search(object):
    def __init__(self,
                 step_max=100,
                 fuzzy=False,
                 add_noise=False
                 ):

        self.observation_dim = 12
        self.action_dim = 6
        self.fuzzy_gvfs = fuzzy

        """ Build the controller and connect with robot """
        self.robot_control = Robot_control()

        """ state """
        self.state = np.zeros(self.observation_dim)
        self.next_state = np.zeros(self.observation_dim)
        self.init_state = np.zeros(self.observation_dim)

        """ action """
        self.action = np.zeros(self.action_dim)

        """ reward """
        self.step_max = step_max
        self.reward = 1.

        """ initial noise """
        self.add_noise = add_noise

        """ force reference """
        self.desired_forces_moments = np.array([[0, 0, -50, 0, 0, 0],
                                                [0, 0, -80, 0, 0, 0],
                                                [0, 0, 60, 0, 0, 0],
                                                [0, 0, -80, 0, 0, 0],
                                                [0, 0, -90, 0, 0, 0],
                                                [0, 0, -100, 0, 0, 0]])

        """ force and moment """
        self.max_force_moment = [70, 5]
        self.safe_force_search = [5, 1]
        self.last_force_error = np.zeros(3)
        self.former_force_error = np.zeros(3)
        self.last_pos_error = np.zeros(3)
        self.former_pos_error = np.zeros(3)
        self.last_setPosition = np.zeros(3)

        """ parameters for search phase """
        self.kp_search = np.array([0.025, 0.025, 0.002])
        self.kd_search = np.array([0.005, 0.005, 0.005])
        self.kp_insert = np.array([0.002, 0.002, 0.015])
        self.kd_insert = np.array([0.0002, 0.0002, 0.0002])
        self.kp = self.kp_search
        self.kd = self.kd_search
        self.kr = np.array([0.02, 0.02, 0.02])
        self.kv = 0.5
        self.Kp_z_0 = 0.93
        self.Kp_z_1 = 0.6
        self.vel_pull_up = 20

        """ action and state space """
        self.state_high = np.array([50, 50, 0, 5, 5, 6, 1453, 70, 995, 5, 5, 6])
        self.state_low = np.array([-50, -50, -50, -5, -5, -6, 1456, 76, 985, -5, -5, -6])
        self.terminated_state = np.array([30, 30, 30, 2, 2, 2])
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(self.observation_dim,), dtype=np.float32)

        self.action_low_bound = np.array([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2])
        self.action_high_bound = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.action_space = spaces.Box(low=-0.2, high=0.2,
                                       shape=(self.action_dim,), dtype=np.float32)

        """ fuzzy parameters range """
        # self.fc = fuzzy_control(low_output=np.array([0., 0., 0., 0., 0., 0.]),
                                # high_output=np.array([0.03, 0.03, 0.004, 0.03, 0.03, 0.03]))

        self.fc = fuzzy_control(low_output=np.array([0., 0., 0., 0., 0., 0.]),
                                high_output=np.array([0.03, 0.03, 0.005, 0.03, 0.03, 0.03]))
        """ initial position and orientation """
        # self.set_initial_pos = np.array([1453.2509, 73.2577, 1000])
        # self.set_initial_euler = np.array([179.8938, 0.9185, 1.0311])
        self.set_initial_pos = np.array([539.8574, - 39.699, 196.9704])
        self.set_initial_euler = np.array([179.4024, -0.3589, -43.0688])

        """ search phase """
        # self.set_search_start_pos = np.array([1453.2509, 73.2577, 995.8843])
        # self.set_search_start_euler = np.array([179.8938, 0.9185, 1.0311])
        self.set_search_start_pos = np.array([539.8548, -39.6966, 194.4638])
        self.set_search_start_euler = np.array([-179.9162, -0.2131, -42.9221])

        # self.set_search_goal_pos = np.array([1453.2509, 73.2577, 990])
        # self.set_search_goal_euler = np.array([179.8938, 0.9185, 1.0311])

        self.set_search_goal_pos = np.array([539.8574, -39.699, 188.0004])
        self.set_search_goal_euler = np.array([179.4024, -0.3589, -43.0688])

        """ insertion phase """
        # self.set_insert_start_pos = np.array([1453.2509, 73.2577, 990])
        # self.set_insert_start_euler = np.array([179.8938, 0.9185, 1.0311])
        self.set_insert_start_pos = np.array([539.8574, -39.699, 188.0004])
        self.set_insert_start_euler = np.array([179.4024, -0.3589, -43.0688])

        self.set_insert_goal_pos = np.array([539.8574, -39.699, 185.0004])
        self.set_insert_goal_euler = np.array([179.4024, -0.3589, -43.0688])

        # self.set_insert_goal_pos = np.array([1453.2509, 73.2577, 980])
        # self.set_insert_goal_euler = np.array([179.8938, 0.9185, 1.0311])
        # self.set_insert_goal = np.array([1453.2509, 73.2577, 980, 179.8938, 0.9185, 1.0311])

        """ random number generator """
        self.rng = np.random.RandomState(5)

    def reset(self):
        """reset the initial parameters"""

        self.timer = 0
        self.pull_terminal = False
        self.kp = self.kp_search
        self.kd = self.kd_search
        self.kr = np.array([0.02, 0.02, 0.02])

        self.desired_force_moment = self.desired_forces_moments[0, :]

        """ pull peg up """
        self.__pull_peg_up()

        if self.pull_terminal:
            self.robot_control.MovelineTo(self.set_initial_pos, self.set_initial_euler, 20)
        else:
            exit("+++++++++++++++++++The pegs didn't move the init position!!!+++++++++++++++++++++")

        """ add randomness for the initial position and orietation"""
        if self.add_noise:
            state_noise = np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3), 0., 0., 0., 0.])

            initial_pos = self.set_search_start_pos + state_noise[0:3]
            inital_euler = self.set_search_start_euler + state_noise[3:6]
        else:
            initial_pos = self.set_search_start_pos
            inital_euler = self.set_search_start_euler

        self.robot_control.MovelineTo(initial_pos, inital_euler, 5)

        self.state = self.__get_state()

        """get max force and moment"""
        self.safe_or_not = self.__safe_or_not()

        if self.safe_or_not is False:
            print('initial force :::::', self.state[:6])
            self.__pull_peg_up()
            exit("++++++++++++++++++The pegs can't move for the exceed force!!!+++++++++++++++++++")
        else:
            print("+++++++++++++++++++++++++++++ Reset Finished !!! ++++++++++++++++++++++++++++++")
            print('initial force :::::', self.state[:6])
            print('initial state :::::', self.state[6:])

        return self.code_state(self.state), self.state, self.pull_terminal

    def step(self, action):
        """execute the action"""

        movePosition, setVel = self.__expert_action()
        print('action', action)
        executeAction = movePosition + movePosition * action
        self.safe_or_not = self.__safe_or_not()

        if self.safe_or_not is False:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Max_Force_Moment:::', self.state[:6])
            print("++++++++++++++++++++++++ The force is too large and pull up !!!!!+++++++++++++++++++++++")
            self.__pull_peg_up()
        else:
            self.robot_control.MovelineTo(self.state[6:9] + executeAction[:3], self.state[9:12] + executeAction[3:], setVel)
            print('setPosition:::', executeAction[:3])
            print('setEuLer::: ', executeAction[3:])
            print('force:::', self.state[:6])
            print('state:::', self.state[6:])

        self.next_state = self.__get_state()
        reward, done = self.__get_reward(self.next_state, self.timer)
        print('Number of steps::', self.timer)
        self.timer += 1

        return self.code_state(self.next_state), self.next_state, reward, done, self.safe_or_not, executeAction, action

    def step_discrete_action(self, action):
        """execute the action"""
        print('action', action)
        movePosition, setVel = self.__expert_action()
        executeAction = np.zeros(6)
        executeAction[action] = movePosition[action]
        executeAction[2] = min(movePosition[2], 0.1)

        self.safe_or_not = self.__safe_or_not()

        if self.safe_or_not is False:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Max_Force_Moment:::', self.state[:6])
            print("++++++++++++++++++++++++ The force is too large and pull up !!!!!+++++++++++++++++++++++")
            self.__pull_peg_up()
        else:
            self.robot_control.MovelineTo(self.state[6:9] + executeAction[:3], self.state[9:12] + executeAction[3:],
                                          setVel)
            print('setPosition:::', executeAction[:3])
            print('setEuLer::: ', executeAction[3:])
            print('force:::', self.state[:6])
            print('state:::', self.state[6:])

        self.next_state = self.__get_state()
        reward, done = self.__get_reward(self.next_state, self.timer)
        print('Number of steps::', self.timer)
        self.timer += 1

        return self.code_state(self.next_state), self.next_state, reward, done, self.safe_or_not, executeAction

    def code_state(self, current_state):
        """ normalize state """

        state = cp.deepcopy(current_state)
        state[6:9] = state[6:9] - self.set_initial_pos

        if state[9] > 0 and state[9] < 180:
            state[9] -= 180
        elif state[9] < 0 and state[9] > -180:
            state[9] += 180
        else:
            pass

        state[10:] = state[10:] - self.set_initial_euler[1:]
        final_state = state
        scale = self.state_high - self.state_low
        final_state[:6] = (state[:6] - self.state_low[:6])/scale[:6]

        return final_state

    def get_running_cost(self, x, u):
        """ ilqg cost """

        target_x = self.code_state(self.set_insert_goal_pos)
        return (x[8] - target_x[2]) + 1/2 * np.linalg.norm(u)

    def seed(self, seed=None):
        """Seed the environment"""

        if seed is not None:
            self.rng = np.random.RandomState(seed)

    """ Get the expert action by PD controller"""
    def __expert_action(self):

        force = self.state[:6]

        self.safe_or_not = self.__safe_or_not()

        force_error = self.desired_force_moment - force
        force_error *= np.array([-1, 1, 1, -1, 1, 1])

        if self.state[8] < self.set_insert_start_pos[2]:
            print("+++++++++++++++++++++++++++++++++ The insert Phase !!!! +++++++++++++++++++++++++++++++++")
            self.kp = self.kp_insert
            self.kd = self.kd_insert
            self.desired_force_moment = self.desired_forces_moments[1, :]
        else:
            print("+++++++++++++++++++++++++++++++++ The search phase !!!! +++++++++++++++++++++++++++++++++")
            if self.fuzzy_gvfs:
                self.kp = self.fc.get_output(force)[:3]
                self.kr = self.fc.get_output(force)[3:]

        if self.timer == 0:
            setPosition = self.kp * force_error[:3]
            self.former_force_error = force_error
        elif self.timer == 1:
            setPosition = self.kp * force_error[:3]
            self.last_setPosition = setPosition
            self.last_force_error = force_error
        else:
            setPosition = self.last_setPosition + self.kp * (force_error[:3] - self.last_force_error[:3]) + \
                          self.kd * (force_error[:3] - 2 * self.last_force_error[:3] + self.former_force_error[:3])
            self.last_setPosition = setPosition
            self.former_force_error = self.last_force_error
            self.last_force_error = force_error

        setEuler = self.kr * force_error[3:6]

        """Set the velocity of robot"""
        setVel = max(self.kv * abs(sum(force_error[:3])), 0.5)
        movePosition = np.zeros(6)
        movePosition[:3] = setPosition
        movePosition[3:6] = setEuler

        return movePosition, setVel

    """get reward"""
    def __get_reward(self, state, step_num):
        done = False
        if self.safe_or_not is False:
            self.reward = -1 + (self.set_search_start_pos[2] - state[8])/(self.set_search_start_pos[2] - self.set_insert_goal_pos[2])
        else:
            """consider force and moment"""
            self.reward = -0.1

        if state[8] < self.set_insert_goal_pos[2]:
            print("+++++++++++++++++++++++++++++ The Assembly Phase Finished!!! ++++++++++++++++++++++++++++")
            self.reward = 1 - step_num / self.step_max
            done = True

        return self.reward, done

    """get the current state"""
    def __get_state(self):

        force = self.robot_control.GetFCForce()
        position, euler, T = self.robot_control.GetCalibTool()

        self.state[:6] = force
        self.state[6:9] = position
        self.state[9:12] = euler

        return self.state

    """pull peg up"""
    def __pull_peg_up(self):

        Position, Euler, T = self.robot_control.GetCalibTool()
        self.robot_control.MovelineTo(Position + np.array([0., 0., 0.2]), Euler, self.vel_pull_up)
        self.robot_control.MovelineTo(Position + np.array([0., 0., 0.5]), Euler, self.vel_pull_up)

        while True:
            """Get the current position"""
            Position, Euler, T = self.robot_control.GetCalibTool()

            """move and rotate"""
            self.robot_control.MovelineTo(Position + np.array([0., 0., 1.]), Euler, self.vel_pull_up)

            """finish or not"""
            if Position[2] > self.set_initial_pos[2]:
                self.pull_terminal = True
                print("++++++++++++++++++++++++++ Pull up the pegs finished!!! ++++++++++++++++++++++++")
                break

    """execute position control"""
    def __positon_control(self, target_position):

        E_z = np.zeros(30)
        action = np.zeros((30, 3))
        """Move by a little step"""
        for i in range(30):

            myForceVector = self.robot_control.GetFCForce()

            if max(abs(myForceVector[0:3])) > 5:
                exit("The pegs can't move for the exceed force!!!")

            """"""
            Position, Euler, Tw_t = self.robot_control.GetCalibTool()

            E_z[i] = target_position[2] - Position[2]

            if i < 3:
                action[i, :] = np.array([0., 0., self.Kp_z_0*E_z[i]])
                vel_low = self.kv * abs(E_z[i])
            else:
                action[i, :] = np.array([0., 0., self.Kp_z_1*E_z[i]])
                vel_low = min(self.kv * abs(E_z[i]), 0.5)

            self.robot_control.MovelineTo(Position + action[i, :], Euler, vel_low)
            if abs(E_z[i]) < 0.001:
                print("The pegs reset successfully!!!")
                self.init_state[0:6] = myForceVector
                self.init_state[6:9] = Position
                self.init_state[9:12] = Euler
                break

        return self.init_state

    """check safety or not"""
    def __safe_or_not(self):

        max_abs_F_M = np.array([max(abs(self.state[0:3])), max(abs(self.state[3:6]))])
        safe_or_not = all(max_abs_F_M < self.max_force_moment)
        return safe_or_not

    """Get the expert action by PD controller"""
    def __pd_controller(self):

        force_desired = self.pull_desired_force_moment
        force = self.state[:6]

        """Judge the force&moment is safe for object"""
        max_abs_F_M = np.array([max(abs(force[0:3])), max(abs(force[3:6]))])
        self.safe_or_not = all(max_abs_F_M < self.max_force_moment)

        force_error = force_desired - force
        force_error *= np.array([-1, 1, 1, -1, 1, 1])

        setPosition = self.kp * force_error[:3]
        self.former_force_error = force_error

        """Get the euler"""
        setEuler = self.kr * force_error[3:6]

        """Set the velocity of robot"""
        setVel = max(self.kv * abs(sum(force_error[:3])), 0.5)

        movePosition = np.zeros(6)
        movePosition[:3] = setPosition
        movePosition[3:6] = setEuler

        return movePosition, setVel
