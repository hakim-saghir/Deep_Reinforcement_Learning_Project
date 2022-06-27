import math
from random import random

from drl_lib.do_not_touch.contracts import *


class LineWorldMDP(MDPEnv):

    def __init__(self):
        self._S = np.array([i for i in range(7)])
        self._A = np.array([0, 1])
        self._R = np.array([-1.0, 0.0, 1.0])
        self._psasr = np.zeros((len(self._S), len(self._A), len(self._S), len(self._R)))

        for s in self._S[1:-1]:
            if s == 1:
                self._psasr[s, 0, s - 1, 0] = 1.0
            else:
                self._psasr[s, 0, s - 1, 1] = 1.0

            if s == 7 - 2:
                self._psasr[s, 1, s + 1, 2] = 1.0
            else:
                self._psasr[s, 1, s + 1, 1] = 1.0

    def states(self) -> np.ndarray:
        return self._S

    def actions(self) -> np.ndarray:
        return self._A

    def rewards(self) -> np.ndarray:
        return self._R

    def is_state_terminal(self, s: int) -> bool:
        return s == 0 or s == 6

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self._psasr[s, a, s_p, r]

    def view_state(self, s: int):
        view = ""
        for s_1 in self._S:
            if s_1 == s:
                view += "X"
            else:
                view += "_"
        print(view)


class LineWorld(SingleAgentEnv):

    def __init__(self, nb_cells: int = 7):
        self.nb_cells = nb_cells
        self.current_cell = math.floor(nb_cells / 2)
        self.step_count = 0

    def state_id(self) -> int:
        return self.current_cell

    def is_game_over(self) -> bool:
        return self.current_cell == self.nb_cells - 1 or self.current_cell == 0

    def act_with_action_id(self, action_id: int):
        self.step_count += 1
        if action_id == 0:
            self.current_cell -= 1
        else:
            self.current_cell += 1

    def score(self) -> float:
        if self.current_cell == 0:
            return -1.0
        elif self.current_cell == self.nb_cells - 1:
            return 1.0
        else:
            return 0.0

    def available_actions_ids(self) -> np.ndarray:
        return np.array([0, 1])

    def reset(self):
        self.current_cell = math.floor(self.nb_cells / 2)
        self.step_count = 0

    def view(self):
        print(f"Game over : {self.is_game_over()}")
        print(f"Score : {self.score()}")
        print(f"Number of steps : {self.score()}")
        for i in range(self.nb_cells):
            if i == self.current_cell:
                print("X", end='')
            else:
                print("_", end='')
        print()

    def reset_random(self):
        self.current_cell = random.randint(0, self.nb_cells - 1)
        self.step_count = 0
