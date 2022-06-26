import numpy as np
import itertools
import random
from drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv


def find_possible_states():
    # 19683 combinaisons
    all_combinations = list(itertools.product([0, 1, 2], repeat=9))
    # Enlever les combinaisons impossibles
    possible_combinations = list()
    for state in all_combinations:
        if state.count(1) == state.count(2) or state.count(1) == (state.count(2) + 1):
            possible_combinations.append(list(state))
    # 6046 combinaisons
    return possible_combinations


class TicTacToeEnv(SingleAgentEnv):
    def __init__(self, is_random_player=True, S=None, watch_the_game=False):
        if S is None:
            self.S = np.array(find_possible_states())
        else:
            self.S = S
        self.A = np.array(range(0, 9))
        self.R = np.array([-1.0, 0.0, 1.0])
        self.reset()
        self.agent_number = 1
        self.second_player = 2
        self.is_random_player = is_random_player
        self.watch_the_game = watch_the_game

    def state_id(self) -> int:
        return np.where((self.S == self.current_state).all(axis=1))[0][0]

    def is_game_over(self) -> bool:
        if list(self.current_state).count(0) == 0:
            return True
        return self.score() != 0

    def act_with_action_id(self, action_id: int):
        if action_id not in self.available_actions_ids():
            return
        self.current_state[action_id] = self.agent_number   # Agent
        if not self.is_game_over():
            if self.is_random_player:
                if self.watch_the_game:
                    self.view()
                random_action_id = np.random.choice(self.available_actions_ids())
                self.current_state[random_action_id] = self.second_player
                if self.watch_the_game:
                    print("****** Random player played in position", random_action_id + 1, "******")
                    self.view()
            else:
                self.view()
                a = -1
                while((a - 1) not in list(self.available_actions_ids())):
                    available_actions = self.available_actions_ids()
                    available_actions += 1
                    print(">> It's up to you, choose one of these positions : ", available_actions)
                    try:
                        a = int(input(">> "))
                    except:
                        a = -1
                self.current_state[a - 1] = self.second_player
                self.view()

    def score(self) -> float:
        for player in [1, 2]:
            if (self.current_state[0] == self.current_state[1] == self.current_state[2] == player) or \
                    (self.current_state[3] == self.current_state[4] == self.current_state[5] == player) or \
                    (self.current_state[6] == self.current_state[7] == self.current_state[8] == player) or \
                    (self.current_state[0] == self.current_state[3] == self.current_state[6] == player) or \
                    (self.current_state[1] == self.current_state[4] == self.current_state[7] == player) or \
                    (self.current_state[2] == self.current_state[5] == self.current_state[8] == player) or \
                    (self.current_state[0] == self.current_state[4] == self.current_state[8] == player) or \
                    (self.current_state[2] == self.current_state[4] == self.current_state[6] == player):
                if player == self.agent_number:
                    return self.R[2]
                return self.R[0]
        return self.R[1]

    def available_actions_ids(self) -> np.ndarray:
        if self.is_game_over():
            return np.array([])
        return np.sort(np.array([index for index, value in enumerate(list(self.current_state)) if value == 0]))

    def reset(self):
        self.current_state = np.zeros(9, dtype=np.int)

    def view(self):
        print("# TIC TAC TOE #")
        for k, s in enumerate(self.current_state):
            if k % 3 == 0:
                print(end="   ")
            if s == 0:
                print("_", end="   ")
            elif s == self.agent_number:
                print("O", end="   ")
            else:
                print("X", end="   ")
            if (k + 1) % 3 == 0 and k != 0:
                print()
        print("###############")

    def reset_random(self):
        self.current_state = np.array(self.S[random.randint(0, len(self.S) - 1)])
        while(list(self.current_state).count(1) != list(self.current_state).count(2) or self.is_game_over()):
            self.current_state = np.array(self.S[random.randint(0, len(self.S) - 1)])
