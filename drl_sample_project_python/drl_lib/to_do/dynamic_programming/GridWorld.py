import math
import numpy as np
from drl_sample_project_python.drl_lib.do_not_touch.contracts import MDPEnv


class GridWorldMDP(MDPEnv):

    def __init__(self):
        self._S = np.array([(j, i) for j in range(5) for i in range(5)])
        s_shape = int(math.sqrt(len(self._S)))
        self._A = np.array([0, 1, 2, 3])  # 0 droite (+1, +0) 1 gauche (-1, +0) 2 bas (+0, +1) 3 haut (+0 ,-1)
        self._R = np.array([-1.0, 0.0, 1.0])
        self._psasr = np.zeros((s_shape, s_shape, len(self._A), s_shape, s_shape, len(self._R)))
        for s in self._S[1:-1]:
            if s[0] == 1 and s[1] == 0:
                self._psasr[s[0], s[1], 1, s[0] - 1, s[1], 0] = 1.0
            elif s[0] == 0:
                self._psasr[s[0], s[1], 1, s[0], s[1], 1] = 0.0
            else:
                self._psasr[s[0], s[1], 1, s[0] - 1, s[1], 1] = 1.0

            if s[0] == 0 and s[1] == 1:
                self._psasr[s[0], s[1], 3, s[0], s[1] - 1, 0] = 1.0
            elif s[1] == 0:
                self._psasr[s[0], s[1], 3, s[0], s[1], 1] = 0.0
            else:
                self._psasr[s[0], s[1], 3, s[0], s[1] - 1, 1] = 1.0

            if s[0] == 3 and s[1] == 4:
                self._psasr[s[0], s[1], 0, s[0] + 1, s[1], 2] = 1.0
            elif s[0] == 4:
                self._psasr[s[0], s[1], 0, s[0], s[1], 1] = 0.0
            else:
                self._psasr[s[0], s[1], 0, s[0] + 1, s[1], 1] = 1.0

            if s[0] == 4 and s[1] == 3:
                self._psasr[s[0], s[1], 2, s[0], s[1] + 1, 2] = 1.0
            elif s[1] == 4:
                self._psasr[s[0], s[1], 2, s[0], s[1] , 1] = 0.0
            else:
                self._psasr[s[0], s[1], 2, s[0], s[1] + 1, 1] = 1.0

    def states(self) -> np.ndarray:
        return self._S

    def actions(self) -> np.ndarray:
        return self._A

    def rewards(self) -> np.ndarray:
        return self._R

    def is_state_terminal(self, s: int) -> bool:
        return s == 0 or s == len(self._S)-1

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self._psasr[s[0], s[1], a, s_p[0], s_p[1], r]

    def view_state(self, s: int):
        view = ""
        for s_1 in range(int(len(self._S)/5)):
            for s_2 in range(int(len(self._S) / 5)):
                if(s_1*5 + s_2) == s:
                    view += '|X'
                else:
                    view += "| "
            view += "|\n"
        print(view)