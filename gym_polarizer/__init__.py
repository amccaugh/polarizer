from gym.envs.registration import register

register(
    id='polarizer-v0',
    entry_point='gym_polarizer.envs:PolarizerEnv',
)