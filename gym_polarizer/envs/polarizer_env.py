import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
# from os import path
from numpy import pi, sin, cos, exp

class PolarizerEnv(gym.Env):

    def __init__(self):
        # self.__version__ = "0.1.0"
        # logging.info("PolarizerEnv - Version {}".format(self.__version__))

        # General variables defining the environment
        self.tau = 1
        self.mu = 1e6
        self.randomize_input_polarization = True
        self.max_num_steps = 4096
        self.theta_max = 180
        self.theta_stepsize = 1
        self.num_steps = 0

        # Define what the agent can do
        self.action_space = spaces.Discrete(7)

        # Observation space is all the available observation data
            # Polarizer 1 position: -180 to 180
            # Polarizer 2 position: -180 to 180
            # Polarizer 3 position: -180 to 180
            # Input polarization H: 0 to 1
            # Input polarization V: 0 to 1
            # self.state = [thetaP1, thetaP1, thetaP1, inputPH, inputPV]
        low = np.array([-180,-180,-180,0,0])
        high = np.array([180, 180, 180, 1, 1])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Initialize state
        self.state = np.array([0.,0.,0.,0.,0.]) # Create array for state
        self._update_state(
            theta1 = 0,
            theta2 = 0,
            theta3 = 0,
            P_in_H = 0.5,
            P_in_V = 0.5,
            )
        self.seed()
        self.reset()



    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        
        # Actions:
            # 0-1: change theta1 (1 = change negative, 2 = change positive)
            # 2-3: change theta2
            # 4-5: change thera3
            # 7 : do nothing
#        theta_actions = np.array([-1, 0, 1])*pi/180

        assert self.action_space.contains(action), "%r (%s) action invalid"%(action, type(action))

        if action == 6:
            pol_idx = 0
            theta_change = 0
        elif action < 6:
            pol_idx = action // 2 # Select which polarizer to change
            pol_dir = (action % 2) # Which direction to change
            if pol_dir == 0:
                theta_change = -self.theta_stepsize
            elif pol_dir == 1:
                theta_change = self.theta_stepsize
        else:
            raise ValueError('Invalid action')

        self.state[pol_idx] += theta_change
        # self.state[:3] = np.clip(self.state[:3], -self.theta_max, self.theta_max)
        self.num_steps += 1

        reward = self._get_reward()
        done = self._is_over()

        return self.state, reward, done, {}


    def reset(self):
        self._randomize_thetas()
        if self.randomize_input_polarization == True:
            self._randomize_input_polarization()
        self.num_steps = 0

        return self.state

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _update_state(self, theta1 = None, theta2 = None, theta3 = None,
                            P_in_H = None, P_in_V = None): 
        for n, var in enumerate([theta1, theta2, theta3, P_in_H, P_in_V]):
            if var is None:
                self.state[n] = self.state[n]
            else:
                self.state[n] = var
        # if (P_in_H is not None) and (P_in_V is not None):
        # Normalize input polarization
        P_in_norm = (self.state[3]**2 + self.state[4]**2)**0.5
        self.state[3] /= P_in_norm # Input polarization state
        self.state[4] /= P_in_norm # Input polarization state
        
    def _randomize_input_polarization(self):
#        import pdb; pdb.set_trace()
#        print('hello')
        new_P_in = np.random.random(size = 2)
        self._update_state(
            P_in_H = new_P_in[0],
            P_in_V = new_P_in[1],
            )
        
        
    def _randomize_thetas(self):
        new_thetas = np.random.uniform(-self.theta_max, self.theta_max, size = 3)
        self._update_state(
                theta1 = new_thetas[0],
                theta2 = new_thetas[1],
                theta3 = new_thetas[2],
            )
        
    
    def _num_output_photons(self):
        P_in = self.state[-2:]
        H1 = self._hwp(self.state[0])
        Q2 = self._qwp(self.state[1])
        H3 = self._hwp(self.state[2])
        P_out =  H3 @ Q2 @ H1 @ P_in
        
#        input_photons = np.random.poisson(self.mu*self.tau) # Make random
        input_photons = self.mu*self.tau # Make deterministic
        output_photons =  input_photons * np.abs(P_out[1])**2 # Extract only the H component
        return output_photons

    def _get_reward(self):
        output_photons = self._num_output_photons()
        input_photons = self.mu*self.tau # Make deterministic
        # reward = - np.log2(1 + output_photons) # Don't make the reward negative! Policy will try to end episode if you do
        reward = (1-output_photons/input_photons)**4
        # reward -= any(np.abs(self.state[:3]) > (self.theta_max - 1)) # Punish extra for going to edge of bounds
#        print(reward)
        return reward
        
    
    def _qwp(self, theta):
        theta = theta*pi/180
        t11 = cos(theta)**2 + 1j*sin(theta)**2
        t12 = (1-1j)*sin(theta)*cos(theta)
        t21 = t12
        t22 = sin(theta)**2 + 1j*cos(theta)**2
        
        Q = exp(-1j*pi/4)*np.array([[t11, t12],[t21, t22]])
        return Q
    
    
    def _hwp(self, theta):
        theta = theta*pi/180
        t11 = cos(2*theta)
        t12 = sin(2*theta)
        t21 = t12
        t22 = -cos(2*theta)
        
        H = exp(-1j*pi/2)*np.array([[t11, t12],[t21, t22]])
        return H
    
    
    def _is_over(self):
        # input_photons = self.mu*self.tau
        # if self._num_output_photons() < input_photons/1000:
        #     print('Final score: %i' % self._get_reward())
        #     return True
        if self.num_steps > self.max_num_steps:
            # print('Final score: %i' % self._get_reward())
            return True
        if any(np.abs(self.state[:3]) > (self.theta_max - 1)): # Game over if polarizer goes all the way to 180deg
            return True
        else:
            return False

if __name__ == '__main__':
    print('hello')
    # import pdb; pdb.set_trace()


