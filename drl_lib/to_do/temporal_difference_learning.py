import numpy as np
import random as rand

from LineWorld import *
from GridWorld import *
from TicTacToeEnv import TicTacToeEnv
from drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction
from drl_lib.do_not_touch.single_agent_env_wrapper import Env3


def sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def q_learning_on_tic_tac_toe_solo(max_iter_count = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = TicTacToeEnv()
    max_iter_count = 10000
    gamma = 0.99
    alpha = 0.1
    epsilon = 0.2

    q = dict()

    for it in range(max_iter_count):

        if env.is_game_over():
            env.reset()

        s = env.state_id()
        aa = env.available_actions_ids()

        if s not in q:
            q[s] = {}
            #pi[s] = {}
            for a in aa:
                q[s][a] = 0.0 if env.is_game_over() else np.random.random()
                #pi[s][a] = rand.random()

        if rand.random() <= epsilon:
            a = np.random.choice(aa)
        else:
            a = aa[np.argmax([q[s][a] for a in aa])]

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        s_p = env.state_id()
        aa_p = env.available_actions_ids()

        if s_p not in q:
            q[s_p] = {}
            for a in aa_p:
                q[s_p][a] = 0.0 if env.is_game_over() else rand.random()

        # try :
        q[s][a] += alpha * (r + gamma * np.max([q[s_p][a] for a in aa_p]) - q[s][a])
        # except :
        #     q[s][a] += r - q[s][a]

    pi = dict()
    for (s, a_dict) in q.items():
        pi[s] = {}
        actions = []
        q_values = []
        for (a, q_value) in a_dict.items():
            actions.append(a)
            q_values.append(q_value)

        best_action_idx = np.argmax(q_values)
        for i in range(len(actions)):
            pi[s][actions[i]] = 1.0 if actions[i] == best_action_idx else 0.0


    return PolicyAndActionValueFunction(pi=pi, q=q)


def expected_sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def q_learning_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass

'''
ON LINE WORLD ENV
'''
def sarsa_on_line_world_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def q_learning_on_line_world_solo(max_iter_count = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = LineWorld()
    max_iter_count = 10000
    gamma = 0.99
    alpha = 0.1
    epsilon = 0.2

    q = dict()

    for it in range(max_iter_count):

        if env.is_game_over():
            env.reset()

        s = env.state_id()
        aa = env.available_actions_ids()

        if s not in q:
            q[s] = {}
            #pi[s] = {}
            for a in aa:
                q[s][a] = 0.0 if env.is_game_over() else np.random.random()
                #pi[s][a] = rand.random()

        if rand.random() <= epsilon:
            a = np.random.choice(aa)
        else:
            a = aa[np.argmax([q[s][a] for a in aa])]

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        s_p = env.state_id()
        aa_p = env.available_actions_ids()

        if s_p not in q:
            q[s_p] = {}
            for a in aa_p:
                q[s_p][a] = 0.0 if env.is_game_over() else rand.random()

        # try :
        q[s][a] += alpha * (r + gamma * np.max([q[s_p][a] for a in aa_p]) - q[s][a])
        # except :
        #     q[s][a] += r - q[s][a]

    pi = dict()
    for (s, a_dict) in q.items():
        pi[s] = {}
        actions = []
        q_values = []
        for (a, q_value) in a_dict.items():
            actions.append(a)
            q_values.append(q_value)

        best_action_idx = np.argmax(q_values)
        for i in range(len(actions)):
            pi[s][actions[i]] = 1.0 if actions[i] == best_action_idx else 0.0


    return PolicyAndActionValueFunction(pi=pi, q=q)


def expected_sarsa_on_line_world_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass

'''
ON LINE WORLD ENV
'''
def sarsa_on_grid_world_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def q_learning_on_grid_world_solo(max_iter_count = 10000) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = GridWorldMDP()
    max_iter_count = 10000
    gamma = 0.99
    alpha = 0.1
    epsilon = 0.2

    q = dict()

    for it in range(max_iter_count):

        if env.is_game_over():
            env.reset()

        s = env.state_id()
        aa = env.available_actions_ids()

        if s not in q:
            q[s] = {}
            #pi[s] = {}
            for a in aa:
                q[s][a] = 0.0 if env.is_game_over() else np.random.random()
                #pi[s][a] = rand.random()

        if rand.random() <= epsilon:
            a = np.random.choice(aa)
        else:
            a = aa[np.argmax([q[s][a] for a in aa])]

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        s_p = env.state_id()
        aa_p = env.available_actions_ids()

        if s_p not in q:
            q[s_p] = {}
            for a in aa_p:
                q[s_p][a] = 0.0 if env.is_game_over() else rand.random()

        # try :
        q[s][a] += alpha * (r + gamma * np.max([q[s_p][a] for a in aa_p]) - q[s][a])
        # except :
        #     q[s][a] += r - q[s][a]

    pi = dict()
    for (s, a_dict) in q.items():
        pi[s] = {}
        actions = []
        q_values = []
        for (a, q_value) in a_dict.items():
            actions.append(a)
            q_values.append(q_value)

        best_action_idx = np.argmax(q_values)
        for i in range(len(actions)):
            pi[s][actions[i]] = 1.0 if actions[i] == best_action_idx else 0.0


    return PolicyAndActionValueFunction(pi=pi, q=q)


def expected_sarsa_on_grid_world_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass

'''
DEMO
'''
def demo():
    '''Tic Tac Toe Env'''
    #print(sarsa_on_tic_tac_toe_solo())
    #print(q_learning_on_tic_tac_toe_solo(max_iter_count = 10000))
    #print(expected_sarsa_on_tic_tac_toe_solo())

    '''Secret Env 3'''
    #print(sarsa_on_secret_env3())
    #print(q_learning_on_secret_env3())
    #print(expected_sarsa_on_secret_env3())

    '''Line World Env'''
    # print(sarsa_on_line_world_solo())
    #print(q_learning_on_line_world_solo(max_iter_count=10000))
    # print(expected_sarsa_on_line_world_solo())

    '''Grid World Env'''
    # print(sarsa_on_grid_world_solo())
    #print(q_learning_on_grid_world_solo(max_iter_count=10000))
    # print(expected_sarsa_on_grid_world_solo())

demo()