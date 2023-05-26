RIGHT_UP = 0
RIGHT_DOWN = 1
LEFT_UP = 2
LEFT_DOWN = 3


class Cell():

    def __init__(self, color, index, i, j, reward_fun):
        self.color = color  # 0 for blue 1 for red 2 for green
        self.i = i
        self.index = index
        self.j = j
        self.reward = reward_fun(color)


class Agent():
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.cum_return = 0


class Board():
    def __init__(self, cells, agent: Agent):
        self.cells = cells
        self.agent = agent

    def recolor(self, red_inicies: list, green_index, ra=-4, rg=4, rr=0):
        self.cells[green_index].reward = rg
        for i in red_inicies:
            self.cells[i].reward = rr

    def move(self, agent, dest_cell):
        pass


def reward_func(color, ra=-4, rg=4, rr=0):
    if color == 0:
        return ra
    elif color == 1:
        return rr
    else:
        return rg


if __name__ == '__main__':
    cells = []

    for i in range(6):
        for j in range(6):
            temp_cell = Cell(0, i * j + j, i, j, reward_func)
