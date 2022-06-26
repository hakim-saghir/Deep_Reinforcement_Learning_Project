import json
import numpy as np
from pathlib import Path
from .TicTacToeEnv import TicTacToeEnv
from ...do_not_touch.result_structures import PolicyAndActionValueFunction
from ...do_not_touch.single_agent_env_wrapper import Env2

path_monte_carlo_es_policy = "./drl_lib/to_do/monte_carlo_methods/backups/monte_carlo_es_on_tic_tac_toe_solo_policy-iter_count={}.json"
path_on_policy_first_visit_monte_carlo_control_policy = "./drl_lib/to_do/monte_carlo_methods/backups/on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo_policy-iter_count={}-epsilon_soft_pi={}.json"
path_off_policy_monte_carlo_control_policy = "./drl_lib/to_do/monte_carlo_methods/backups/off_policy_monte_carlo_control_on_tic_tac_toe_solo_policy-iter_count={}-epsilon_soft_b={}.json"
path_monte_carlo_es_policy_env2 = "./drl_lib/to_do/monte_carlo_methods/backups/monte_carlo_es_on_tic_tac_toe_solo_policy_env2-iter_count={}.json"
path_on_policy_first_visit_monte_carlo_control_policy_env2 = "./drl_lib/to_do/monte_carlo_methods/backups/on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo_policy_env2-iter_count={}-epsilon_soft_pi={}.json"
path_off_policy_monte_carlo_control_policy_env2 = "./drl_lib/to_do/monte_carlo_methods/backups/off_policy_monte_carlo_control_on_tic_tac_toe_solo_policy_env2-iter_count={}-epsilon_soft_b={}.json"


def monte_carlo_es_on_tic_tac_toe_solo(iter_count=100) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    tictactoe_env = TicTacToeEnv()
    pi = dict({})
    q = dict({})
    Returns = dict({})
    for it in range(iter_count):
        tictactoe_env.reset_random()
        s0 = tictactoe_env.state_id()
        if tictactoe_env.is_game_over():
            continue
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
    q_str = {}
    for s in q.keys():
        q_str[str(s)] = {}
        for a in q[s].keys():
            q_str[str(s)][str(a)] = q[s][a]
    with open(Path(path_monte_carlo_es_policy.format(iter_count)).absolute().as_posix(), "w") as backup_file:
        json.dump(dict({"S": tictactoe_env.S.tolist(), "pi": pi_str, "q": q_str}), backup_file)
    return PolicyAndActionValueFunction(pi=pi, q=q)


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(iter_count=100, epsilon_soft_pi=0.001) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    tictactoe_env = TicTacToeEnv()
    pi = dict({})
    q = dict({})
    Returns = dict({})
    for it in range(iter_count):
        tictactoe_env.reset_random()
        if tictactoe_env.is_game_over():
            continue
        s0 = tictactoe_env.state_id()
        if s0 not in pi:
            temp_array_soft_pi = np.random.uniform(epsilon_soft_pi, 1, (len(tictactoe_env.available_actions_ids())))
            temp_array_soft_pi /= np.sum(temp_array_soft_pi)
            while((temp_array_soft_pi < epsilon_soft_pi).any()):
                temp_array_soft_pi[np.argmin(temp_array_soft_pi)] *= 2
                temp_array_soft_pi /= np.sum(temp_array_soft_pi)
            temp_array_q = np.random.random((len(tictactoe_env.available_actions_ids())))
            pi[s0] = dict({})
            q[s0] = dict({})
            Returns[s0] = dict({})
            for action_index, action in enumerate(tictactoe_env.available_actions_ids()):
                pi[s0][action] = temp_array_soft_pi[action_index]
                q[s0][action] = temp_array_q[action_index]
                Returns[s0][action] = []
        a0 = max(pi[s0], key=pi[s0].get)
        s = s0
        a = a0
        tictactoe_env.act_with_action_id(a0)
        s_p = tictactoe_env.state_id()
        if s_p not in pi:
            temp_array_soft_pi = np.random.uniform(epsilon_soft_pi, 1, (len(tictactoe_env.available_actions_ids())))
            temp_array_soft_pi /= np.sum(temp_array_soft_pi)
            while((temp_array_soft_pi < epsilon_soft_pi).any()):
                temp_array_soft_pi[np.argmin(temp_array_soft_pi)] *= 2
                temp_array_soft_pi /= np.sum(temp_array_soft_pi)
            temp_array_q = np.random.random((len(tictactoe_env.available_actions_ids())))
            pi[s_p] = dict({})
            q[s_p] = dict({})
            Returns[s_p] = dict({})
            for action_index, action in enumerate(tictactoe_env.available_actions_ids()):
                pi[s_p][action] = float(temp_array_soft_pi[action_index])
                q[s_p][action] = temp_array_q[action_index]
                Returns[s_p][action] = []
        r = tictactoe_env.score()
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]
        while not tictactoe_env.is_game_over():
            choices, probabilities = np.array(list(pi[s_p].keys())), np.array(list(pi[s_p].values()))
            probabilities /= np.sum(probabilities)
            a = np.random.choice(choices, p=probabilities)
            a0 = np.random.choice(tictactoe_env.available_actions_ids())
            tictactoe_env.act_with_action_id(a0)
            s = s_p
            s_p = tictactoe_env.state_id()
            if s_p not in pi:
                temp_array_soft_pi = np.random.uniform(epsilon_soft_pi, 1, (len(tictactoe_env.available_actions_ids())))
                temp_array_soft_pi /= np.sum(temp_array_soft_pi)
                while ((temp_array_soft_pi < epsilon_soft_pi).any()):
                    temp_array_soft_pi[np.argmin(temp_array_soft_pi)] *= 2
                    temp_array_soft_pi /= np.sum(temp_array_soft_pi)
                temp_array_q = np.random.random((len(tictactoe_env.available_actions_ids())))
                pi[s_p] = dict({})
                q[s_p] = dict({})
                Returns[s_p] = dict({})
                for action_index, action in enumerate(tictactoe_env.available_actions_ids()):
                    pi[s_p][action] = temp_array_soft_pi[action_index]
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
            a_etoile = max(q[s_t], key=q[s_t].get)
            for action_index in pi[s_t].keys():
                if action_index == a_etoile:
                    pi[s_t][action_index] = 1 - epsilon_soft_pi + (epsilon_soft_pi / abs(pi[s_t][action_index]))
                else:
                    pi[s_t][action_index] = epsilon_soft_pi / abs(pi[s_t][action_index])
            for action_index in pi[s_t].keys():
                pi[s_t][action_index] /= sum(pi[s_t].values())
    pi_str = {}
    for s in pi.keys():
        pi_str[str(s)] = {}
        for a in pi[s].keys():
            pi_str[str(s)][str(a)] = pi[s][a]
    q_str = {}
    for s in q.keys():
        q_str[str(s)] = {}
        for a in q[s].keys():
            q_str[str(s)][str(a)] = q[s][a]
    with open(Path(path_on_policy_first_visit_monte_carlo_control_policy.format(iter_count, epsilon_soft_pi)).absolute().as_posix(), "w") as backup_file:
        json.dump(dict({"S": tictactoe_env.S.tolist(), "pi": pi_str, "q": q_str}), backup_file)
    return PolicyAndActionValueFunction(pi=pi, q=q)


def off_policy_monte_carlo_control_on_tic_tac_toe_solo(iter_count=100, epsilon_soft_b=0.001) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    tictactoe_env = TicTacToeEnv()
    pi, q, c = dict({}), dict({}), dict({})
    for it in range(iter_count):
        tictactoe_env.reset_random()
        if tictactoe_env.is_game_over():
            continue
        s0 = tictactoe_env.state_id()
        b = dict({})
        temp_array_soft_b = np.random.uniform(epsilon_soft_b, 1, (len(tictactoe_env.available_actions_ids())))
        temp_array_soft_b /= np.sum(temp_array_soft_b)
        while ((temp_array_soft_b < epsilon_soft_b).any()):
            temp_array_soft_b[np.argmin(temp_array_soft_b)] *= 2
            temp_array_soft_b /= np.sum(temp_array_soft_b)
        temp_array_q = np.random.random((len(tictactoe_env.available_actions_ids())))
        temp_array_p = np.zeros((len(tictactoe_env.available_actions_ids())))
        temp_array_p[np.argmax(temp_array_q)] = 1.0
        b[s0], pi[s0], q[s0], c[s0] = dict({}), dict({}), dict({}), dict({})
        for action_index, action in enumerate(tictactoe_env.available_actions_ids()):
            b[s0][action] = temp_array_soft_b[action_index]
            pi[s0][action] = temp_array_p[action_index]
            q[s0][action] = temp_array_q[action_index]
            c[s0][action] = 0.0
        a0 = max(b[s0], key=b[s0].get)
        s = s0
        a = a0
        tictactoe_env.act_with_action_id(a0)
        s_p = tictactoe_env.state_id()
        temp_array_soft_b = np.random.uniform(epsilon_soft_b, 1, (len(tictactoe_env.available_actions_ids())))
        temp_array_soft_b /= np.sum(temp_array_soft_b)
        while ((temp_array_soft_b < epsilon_soft_b).any()):
            temp_array_soft_b[np.argmin(temp_array_soft_b)] *= 2
            temp_array_soft_b /= np.sum(temp_array_soft_b)
        temp_array_q = np.random.random((len(tictactoe_env.available_actions_ids())))
        temp_array_p = np.zeros((len(tictactoe_env.available_actions_ids())))
        if len(tictactoe_env.available_actions_ids()):
            temp_array_p[np.argmax(temp_array_q)] = 1.0
        b[s_p], pi[s_p], q[s_p], c[s_p] = dict({}), dict({}), dict({}), dict({})
        for action_index, action in enumerate(tictactoe_env.available_actions_ids()):
            b[s_p][action] = temp_array_soft_b[action_index]
            pi[s_p][action] = temp_array_p[action_index]
            q[s_p][action] = temp_array_q[action_index]
            c[s_p][action] = 0.0
        r = tictactoe_env.score()
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]
        while not tictactoe_env.is_game_over():
            choices, probabilities = np.array(list(b[s_p].keys())), np.array(list(b[s_p].values()))
            probabilities /= np.sum(probabilities)
            a = np.random.choice(choices, p=probabilities)
            tictactoe_env.act_with_action_id(a0)
            s = s_p
            s_p = tictactoe_env.state_id()
            temp_array_soft_b = np.random.uniform(epsilon_soft_b, 1, (len(tictactoe_env.available_actions_ids())))
            temp_array_soft_b /= np.sum(temp_array_soft_b)
            while ((temp_array_soft_b < epsilon_soft_b).any()):
                temp_array_soft_b[np.argmin(temp_array_soft_b)] *= 2
                temp_array_soft_b /= np.sum(temp_array_soft_b)
            temp_array_q = np.random.random((len(tictactoe_env.available_actions_ids())))
            temp_array_p = np.zeros((len(tictactoe_env.available_actions_ids())))
            temp_array_p[np.argmax(temp_array_q)] = 1.0
            b[s_p], pi[s_p], q[s_p], c[s_p] = dict({}), dict({}), dict({}), dict({})
            for action_index, action in enumerate(tictactoe_env.available_actions_ids()):
                b[s_p][action] = temp_array_soft_b[action_index]
                pi[s_p][action] = temp_array_p[action_index]
                q[s_p][action] = temp_array_q[action_index]
                c[s_p][action] = 0.0
            r = tictactoe_env.score()
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(s_p)
            r_history.append(r)
        G = 0.0
        W = 1.0
        for t in reversed(range(len(s_history))):
            G = 0.999 * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]
            c[s_t][a_t] += W
            q[s_t][a_t] += ((W / c[s_t][a_t]) * (G - q[s_t][a_t]))
            for action_index in pi[s_t].keys():
                pi[s_t][action_index] = 0.0
            pi[s_t][max(q[s_t], key=q[s_t].get)] = 1.0
            if a_t != max(pi[s_t], key=pi[s_t].get):
                break
            W /= b[s_t][a_t]
    pi_str = {}
    for s in pi.keys():
        pi_str[str(s)] = {}
        for a in pi[s].keys():
            pi_str[str(s)][str(a)] = pi[s][a]
    q_str = {}
    for s in q.keys():
        q_str[str(s)] = {}
        for a in q[s].keys():
            q_str[str(s)][str(a)] = q[s][a]
    with open(Path(path_off_policy_monte_carlo_control_policy.format(iter_count, epsilon_soft_b)).absolute().as_posix(), "w") as backup_file:
        json.dump(dict({"S": tictactoe_env.S.tolist(), "pi": pi_str, "q": q_str}), backup_file)
    return PolicyAndActionValueFunction(pi=pi, q=q)


def monte_carlo_es_on_secret_env2(iter_count=100) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()
    pi = dict({})
    q = dict({})
    Returns = dict({})
    for it in range(iter_count):
        env.reset_random()
        s0 = env.state_id()
        if env.is_game_over():
            continue
        if s0 not in pi:
            temp_array_pi = np.random.random((len(env.available_actions_ids())))
            temp_array_q = np.random.random((len(env.available_actions_ids())))
            temp_array_pi /= np.sum(temp_array_pi)
            pi[s0] = dict({})
            q[s0] = dict({})
            Returns[s0] = dict({})
            for action_index, action in enumerate(env.available_actions_ids()):
                pi[s0][action] = temp_array_pi[action_index]
                q[s0][action] = temp_array_q[action_index]
                Returns[s0][action] = []
        a0 = np.random.choice(env.available_actions_ids())
        s = s0
        a = a0
        env.act_with_action_id(a0)
        s_p = env.state_id()
        if s_p not in pi:
            temp_array_pi = np.random.random((len(env.available_actions_ids())))
            temp_array_q = np.random.random((len(env.available_actions_ids())))
            temp_array_pi /= np.sum(temp_array_pi)
            pi[s_p] = dict({})
            q[s_p] = dict({})
            Returns[s_p] = dict({})
            for action_index, action in enumerate(env.available_actions_ids()):
                pi[s_p][action] = temp_array_pi[action_index]
                q[s_p][action] = temp_array_q[action_index]
                Returns[s_p][action] = []
        r = env.score()
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]
        while not env.is_game_over():
            a = np.random.choice(list(pi[s_p].keys()), p=list(pi[s_p].values()))
            a0 = np.random.choice(env.available_actions_ids())
            env.act_with_action_id(a0)
            s = s_p
            s_p = env.state_id()
            if s_p not in pi:
                temp_array_pi = np.random.random((len(env.available_actions_ids())))
                temp_array_q = np.random.random((len(env.available_actions_ids())))
                temp_array_pi /= np.sum(temp_array_pi)
                pi[s_p] = dict({})
                q[s_p] = dict({})
                Returns[s_p] = dict({})
                for action_index, action in enumerate(env.available_actions_ids()):
                    pi[s_p][action] = temp_array_pi[action_index]
                    q[s_p][action] = temp_array_q[action_index]
                    Returns[s_p][action] = []
            r = env.score()
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
    q_str = {}
    for s in q.keys():
        q_str[str(s)] = {}
        for a in q[s].keys():
            q_str[str(s)][str(a)] = q[s][a]
    with open(Path(path_monte_carlo_es_policy_env2.format(iter_count)).absolute().as_posix(), "w") as backup_file:
        json.dump(dict({"pi": pi_str, "q": q_str}), backup_file)
    return PolicyAndActionValueFunction(pi=pi, q=q)


def on_policy_first_visit_monte_carlo_control_on_secret_env2(iter_count=100, epsilon_soft_pi=0.001) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    pi = dict({})
    q = dict({})
    Returns = dict({})
    for it in range(iter_count):
        env.reset_random()
        if env.is_game_over():
            continue
        s0 = env.state_id()
        if s0 not in pi:
            temp_array_soft_pi = np.random.uniform(epsilon_soft_pi, 1, (len(env.available_actions_ids())))
            temp_array_soft_pi /= np.sum(temp_array_soft_pi)
            while((temp_array_soft_pi < epsilon_soft_pi).any()):
                temp_array_soft_pi[np.argmin(temp_array_soft_pi)] *= 2
                temp_array_soft_pi /= np.sum(temp_array_soft_pi)
            temp_array_q = np.random.random((len(env.available_actions_ids())))
            pi[s0] = dict({})
            q[s0] = dict({})
            Returns[s0] = dict({})
            for action_index, action in enumerate(env.available_actions_ids()):
                pi[s0][action] = temp_array_soft_pi[action_index]
                q[s0][action] = temp_array_q[action_index]
                Returns[s0][action] = []
        a0 = max(pi[s0], key=pi[s0].get)
        s = s0
        a = a0
        env.act_with_action_id(a0)
        s_p = env.state_id()
        if s_p not in pi:
            temp_array_soft_pi = np.random.uniform(epsilon_soft_pi, 1, (len(env.available_actions_ids())))
            temp_array_soft_pi /= np.sum(temp_array_soft_pi)
            while((temp_array_soft_pi < epsilon_soft_pi).any()):
                temp_array_soft_pi[np.argmin(temp_array_soft_pi)] *= 2
                temp_array_soft_pi /= np.sum(temp_array_soft_pi)
            temp_array_q = np.random.random((len(env.available_actions_ids())))
            pi[s_p] = dict({})
            q[s_p] = dict({})
            Returns[s_p] = dict({})
            for action_index, action in enumerate(env.available_actions_ids()):
                pi[s_p][action] = float(temp_array_soft_pi[action_index])
                q[s_p][action] = temp_array_q[action_index]
                Returns[s_p][action] = []
        r = env.score()
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]
        while not env.is_game_over():
            choices, probabilities = np.array(list(pi[s_p].keys())), np.array(list(pi[s_p].values()))
            probabilities /= np.sum(probabilities)
            a = np.random.choice(choices, p=probabilities)
            a0 = np.random.choice(env.available_actions_ids())
            env.act_with_action_id(a0)
            s = s_p
            s_p = env.state_id()
            if s_p not in pi:
                temp_array_soft_pi = np.random.uniform(epsilon_soft_pi, 1, (len(env.available_actions_ids())))
                temp_array_soft_pi /= np.sum(temp_array_soft_pi)
                while ((temp_array_soft_pi < epsilon_soft_pi).any()):
                    temp_array_soft_pi[np.argmin(temp_array_soft_pi)] *= 2
                    temp_array_soft_pi /= np.sum(temp_array_soft_pi)
                temp_array_q = np.random.random((len(env.available_actions_ids())))
                pi[s_p] = dict({})
                q[s_p] = dict({})
                Returns[s_p] = dict({})
                for action_index, action in enumerate(env.available_actions_ids()):
                    pi[s_p][action] = temp_array_soft_pi[action_index]
                    q[s_p][action] = temp_array_q[action_index]
                    Returns[s_p][action] = []
            r = env.score()
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
            # ties broken arbitrary = If multiple elements are tied for least value, the head is one of those elements
            a_etoile = max(q[s_t], key=q[s_t].get)
            for action_index in pi[s_t].keys():
                if action_index == a_etoile:
                    pi[s_t][action_index] = 1 - epsilon_soft_pi + (epsilon_soft_pi / abs(pi[s_t][action_index]))
                else:
                    pi[s_t][action_index] = epsilon_soft_pi / abs(pi[s_t][action_index])
            for action_index in pi[s_t].keys():
                pi[s_t][action_index] /= sum(pi[s_t].values())
    pi_str = {}
    for s in pi.keys():
        pi_str[str(s)] = {}
        for a in pi[s].keys():
            pi_str[str(s)][str(a)] = pi[s][a]
    q_str = {}
    for s in q.keys():
        q_str[str(s)] = {}
        for a in q[s].keys():
            q_str[str(s)][str(a)] = q[s][a]
    with open(Path(path_on_policy_first_visit_monte_carlo_control_policy_env2.format(iter_count, epsilon_soft_pi)).absolute().as_posix(), "w") as backup_file:
        json.dump(dict({"pi": pi_str, "q": q_str}), backup_file)
    return PolicyAndActionValueFunction(pi=pi, q=q)


def off_policy_monte_carlo_control_on_secret_env2(iter_count=100, epsilon_soft_b=0.001) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    pi, q, c = dict({}), dict({}), dict({})
    for it in range(iter_count):
        env.reset_random()
        if env.is_game_over():
            continue
        s0 = env.state_id()
        b = dict({})
        temp_array_soft_b = np.random.uniform(epsilon_soft_b, 1, (len(env.available_actions_ids())))
        temp_array_soft_b /= np.sum(temp_array_soft_b)
        while ((temp_array_soft_b < epsilon_soft_b).any()):
            temp_array_soft_b[np.argmin(temp_array_soft_b)] *= 2
            temp_array_soft_b /= np.sum(temp_array_soft_b)
        temp_array_q = np.random.random((len(env.available_actions_ids())))
        temp_array_p = np.zeros((len(env.available_actions_ids())))
        temp_array_p[np.argmax(temp_array_q)] = 1.0
        b[s0], pi[s0], q[s0], c[s0] = dict({}), dict({}), dict({}), dict({})
        for action_index, action in enumerate(env.available_actions_ids()):
            b[s0][action] = temp_array_soft_b[action_index]
            pi[s0][action] = temp_array_p[action_index]
            q[s0][action] = temp_array_q[action_index]
            c[s0][action] = 0.0
        a0 = max(b[s0], key=b[s0].get)
        s = s0
        a = a0
        env.act_with_action_id(a0)
        s_p = env.state_id()
        temp_array_soft_b = np.random.uniform(epsilon_soft_b, 1, (len(env.available_actions_ids())))
        temp_array_soft_b /= np.sum(temp_array_soft_b)
        while ((temp_array_soft_b < epsilon_soft_b).any()):
            temp_array_soft_b[np.argmin(temp_array_soft_b)] *= 2
            temp_array_soft_b /= np.sum(temp_array_soft_b)
        temp_array_q = np.random.random((len(env.available_actions_ids())))
        temp_array_p = np.zeros((len(env.available_actions_ids())))
        if len(env.available_actions_ids()):
            temp_array_p[np.argmax(temp_array_q)] = 1.0
        b[s_p], pi[s_p], q[s_p], c[s_p] = dict({}), dict({}), dict({}), dict({})
        for action_index, action in enumerate(env.available_actions_ids()):
            b[s_p][action] = temp_array_soft_b[action_index]
            pi[s_p][action] = temp_array_p[action_index]
            q[s_p][action] = temp_array_q[action_index]
            c[s_p][action] = 0.0
        r = env.score()
        s_history = [s]
        a_history = [a]
        s_p_history = [s_p]
        r_history = [r]
        while not env.is_game_over():
            choices, probabilities = np.array(list(b[s_p].keys())), np.array(list(b[s_p].values()))
            probabilities /= np.sum(probabilities)
            a = np.random.choice(choices, p=probabilities)
            env.act_with_action_id(a0)
            s = s_p
            s_p = env.state_id()
            temp_array_soft_b = np.random.uniform(epsilon_soft_b, 1, (len(env.available_actions_ids())))
            temp_array_soft_b /= np.sum(temp_array_soft_b)
            while ((temp_array_soft_b < epsilon_soft_b).any()):
                temp_array_soft_b[np.argmin(temp_array_soft_b)] *= 2
                temp_array_soft_b /= np.sum(temp_array_soft_b)
            temp_array_q = np.random.random((len(env.available_actions_ids())))
            temp_array_p = np.zeros((len(env.available_actions_ids())))
            temp_array_p[np.argmax(temp_array_q)] = 1.0
            b[s_p], pi[s_p], q[s_p], c[s_p] = dict({}), dict({}), dict({}), dict({})
            for action_index, action in enumerate(env.available_actions_ids()):
                b[s_p][action] = temp_array_soft_b[action_index]
                pi[s_p][action] = temp_array_p[action_index]
                q[s_p][action] = temp_array_q[action_index]
                c[s_p][action] = 0.0
            r = env.score()
            s_history.append(s)
            a_history.append(a)
            s_p_history.append(s_p)
            r_history.append(r)
        G = 0.0
        W = 1.0
        for t in reversed(range(len(s_history))):
            G = 0.999 * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]
            c[s_t][a_t] += W
            q[s_t][a_t] += ((W / c[s_t][a_t]) * (G - q[s_t][a_t]))
            for action_index in pi[s_t].keys():
                pi[s_t][action_index] = 0.0
            pi[s_t][max(q[s_t], key=q[s_t].get)] = 1.0
            if a_t != max(pi[s_t], key=pi[s_t].get):
                break
            W /= b[s_t][a_t]
    pi_str = {}
    for s in pi.keys():
        pi_str[str(s)] = {}
        for a in pi[s].keys():
            pi_str[str(s)][str(a)] = pi[s][a]
    q_str = {}
    for s in q.keys():
        q_str[str(s)] = {}
        for a in q[s].keys():
            q_str[str(s)][str(a)] = q[s][a]
    with open(Path(path_off_policy_monte_carlo_control_policy_env2.format(iter_count, epsilon_soft_b)).absolute().as_posix(),
              "w") as backup_file:
        json.dump(dict({"pi": pi_str, "q": q_str}), backup_file)
    return PolicyAndActionValueFunction(pi=pi, q=q)


def demo():
    print(monte_carlo_es_on_tic_tac_toe_solo(iter_count=1000000))
    print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(iter_count=1000000))
    print(off_policy_monte_carlo_control_on_tic_tac_toe_solo(iter_count=1000000))
    print(monte_carlo_es_on_secret_env2(iter_count=1000000))
    print(on_policy_first_visit_monte_carlo_control_on_secret_env2(iter_count=1000000))
    print(off_policy_monte_carlo_control_on_secret_env2(iter_count=1000000))
