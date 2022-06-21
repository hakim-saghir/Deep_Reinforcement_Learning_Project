import numpy as np

from .TicTacToeEnv import TicTacToeEnv
from ...do_not_touch.result_structures import PolicyAndActionValueFunction
from ...do_not_touch.single_agent_env_wrapper import Env2


def monte_carlo_es_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    iter_count = 10000
    tictactoe_env = TicTacToeEnv()
    S = tictactoe_env.S
    A = tictactoe_env.A
    pi = np.random.random((len(S), len(A))).astype("double")
    # Somme des probabilités à 1
    for s in range(len(S)):
        pi[s] /= np.sum(pi[s])
    q = np.random.random((len(S), len(A)))
    Returns = [[[] for a in A] for s in S]
    for it in range(iter_count):
        tictactoe_env.reset_random()
        s0 = tictactoe_env.state_id()
        a0 = np.random.choice(tictactoe_env.available_actions_ids())
        s = s0
        a = a0
        tictactoe_env.act_with_action_id(a0)
        s_p = tictactoe_env.state_id()
        r = tictactoe_env.score()
        terminal = tictactoe_env.is_game_over()
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]
        while not tictactoe_env.is_game_over():
            a = np.random.choice(A, p=pi[s])
            a0 = np.random.choice(tictactoe_env.available_actions_ids())
            tictactoe_env.act_with_action_id(a0)
            s = s_p
            s_p = tictactoe_env.state_id()
            r = tictactoe_env.score()
            terminal = tictactoe_env.is_game_over()
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
            q[s_t, a_t] = np.mean(Returns[s_t][a_t])
            pi[s_t, :] = 0.0
            pi[s_t, np.argmax(q[s_t])] = 1.0
    policy = {}
    for s in range(len(S)):
        policy[s] = {}
        for a in range(len(A)):
            policy[s][a] = pi[s, a]
    actionValueFunction = {}
    for s in range(len(S)):
        actionValueFunction[s] = {}
        for a in range(len(A)):
            actionValueFunction[s][a] = q[s, a]
    return PolicyAndActionValueFunction(pi=policy, q=actionValueFunction)


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
