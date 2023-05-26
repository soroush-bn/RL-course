from gym import ActionWrapper
from gym import Wrapper
from gym.spaces import Discrete

UP = 1
DOWN = 0
RIGHT = 2
LEFT = 3


class AdditionalActions(ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        # self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(10)

    def action(self, action):
        return action


"""

    0: Move south (down)

    1: Move north (up)

    2: Move east (right)

    3: Move west (left)

    4: Pickup passenger

    5: Drop off passenger
    
    6: up right
    7 : yp left
    8: down right
    9: down left
"""


class DiagonalEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    # up -100 right +20 down +100 left -20
    def step(self, action,pre):
        state = list(self.env.decode(self.env.s))
        if action < 6:
            return self.env.step(action)
        else:
            if action == 6 and (state[0] > 0 and state[1] < 4):  # up + right
                self.env.step(UP)
                return self.env.step(RIGHT)
            elif action == 7 and (state[0] > 0 and state[1] > 0):  # up + left
                self.env.step(UP)
                return self.env.step(LEFT)
            elif action == 8 and (state[0] < 4 and state[1] < 4):  # Down + Right
                self.env.step(DOWN)
                return self.env.step(RIGHT)
            elif action == 9 and (state[0] < 4 and state[1] > 0):  # Down + left self.env.step(UP)
                self.env.step(DOWN)
                return self.env.step(LEFT)
            else:
                return pre
