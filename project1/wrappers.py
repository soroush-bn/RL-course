import gym


class NegMovement(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        return observation, reward-0.05, done, info



class NegHole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if observation in [5,7,11,12]:
            return observation, -2, done, info
        else:
            return observation, reward, done, info


