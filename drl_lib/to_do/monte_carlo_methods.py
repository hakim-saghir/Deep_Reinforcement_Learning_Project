import numpy as np
import random as rand

from drl_lib.do_not_touch.result_structures import PolicyAndActionValueFunction
from drl_lib.do_not_touch.single_agent_env_wrapper import Env2
from TicTacToeEnv import TicTacToeEnv



def monte_carlo_es_on_tic_tac_toe_solo(iter_count=1000) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    # TODO
    tictactoe_env = TicTacToeEnv()
    S = tictactoe_env.S
    A = tictactoe_env.A
    pi = dict({})
    q = dict({})
    Returns = dict({})
    for it in range(iter_count):
        if it % 1000 == 0:
            print(round((it / iter_count) * 100, 1))
        tictactoe_env.reset_random()
        s0 = tictactoe_env.state_id()
        if s0 not in pi:
            temp_array_pi = np.random.random((len(tictactoe_env.available_actions_ids())))
            temp_array_q = np.random.random((len(tictactoe_env.available_actions_ids())))
            temp_array_pi /= np.sum(temp_array_pi)
            pi[s0] = dict({})
            q[s0] = dict({})
            Returns[s0] = dict({})
            for action_index, action in enumerate(tictactoe_env.available_actions_ids()):
                pi[s0][action] = temp_array_pi[action_index]
                q[s0][action] = temp_array_q[action_index]
                Returns[s0][action] = []
        a0 = np.random.choice(tictactoe_env.available_actions_ids())
        s = s0
        a = a0
        tictactoe_env.act_with_action_id(a0)
        s_p = tictactoe_env.state_id()
        if s_p not in pi:
            temp_array_pi = np.random.random((len(tictactoe_env.available_actions_ids())))
            temp_array_q = np.random.random((len(tictactoe_env.available_actions_ids())))
            temp_array_pi /= np.sum(temp_array_pi)
            pi[s_p] = dict({})
            q[s_p] = dict({})
            Returns[s_p] = dict({})
            for action_index, action in enumerate(tictactoe_env.available_actions_ids()):
                pi[s_p][action] = temp_array_pi[action_index]
                q[s_p][action] = temp_array_q[action_index]
                Returns[s_p][action] = []
        r = tictactoe_env.score()
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]
        while not tictactoe_env.is_game_over():
            a = np.random.choice(list(pi[s_p].keys()), p=list(pi[s_p].values()))
            a0 = np.random.choice(tictactoe_env.available_actions_ids())
            tictactoe_env.act_with_action_id(a0)
            s = s_p
            s_p = tictactoe_env.state_id()
            if s_p not in pi:
                temp_array_pi = np.random.random((len(tictactoe_env.available_actions_ids())))
                temp_array_q = np.random.random((len(tictactoe_env.available_actions_ids())))
                temp_array_pi /= np.sum(temp_array_pi)
                pi[s_p] = dict({})
                q[s_p] = dict({})
                Returns[s_p] = dict({})
                for action_index, action in enumerate(tictactoe_env.available_actions_ids()):
                    pi[s_p][action] = temp_array_pi[action_index]
                    q[s_p][action] = temp_array_q[action_index]
                    Returns[s_p][action] = []
            r = tictactoe_env.score()
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(s_p)
            r_history.append(r)
        G = 0
        for t in reversed(range(len(s_history))):
            G = 0.999 * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]
            appear = False
            for t_p in range(t - 1):
                if s_history[t_p] == s_t and a_history[t_p] == a_t:
                    appear = True
                    break
            if appear:
                continue
            Returns[s_t][a_t].append(G)
            q[s_t][a_t] = np.mean(Returns[s_t][a_t])
            for action_index in pi[s_t].keys():
                if action_index == max(q[s_t], key=q[s_t].get):
                    pi[s_t][action_index] = 1.0
                else:
                    pi[s_t][action_index] = 0.0
    pi_str = {}
    for s in pi.keys():
        pi_str[str(s)] = {}
        for a in pi[s].keys():
            pi_str[str(s)][str(a)] = pi[s][a]
    # with open(Path(path_monte_carlo_es_policy.format(iter_count)).absolute().as_posix(), "w") as backup_file:
    #     json.dump(dict({"S": S.tolist(), "pi": pi_str}), backup_file)
    return PolicyAndActionValueFunction(pi=pi, q=q)


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()
    # TODO
    pass


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO
    pass


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO
    pass


def demo():
    print(monte_carlo_es_on_tic_tac_toe_solo())
    # print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    # print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())
    #
    # print(monte_carlo_es_on_secret_env2())
    # print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    # print(off_policy_monte_carlo_control_on_secret_env2())
