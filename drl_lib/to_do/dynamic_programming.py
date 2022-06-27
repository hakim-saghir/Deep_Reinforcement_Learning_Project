import random as rand

from drl_lib.do_not_touch.mdp_env_wrapper import Env1
from drl_lib.do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
from LineWorld import *
from GridWorld import *


def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    # Création de l'environnement
    mdp_env = LineWorldMDP()
    S = mdp_env.states()
    R = mdp_env.rewards()
    A = mdp_env.actions()

    # Création de la policy
    policy = {}
    for s in S:
        actions = {}
        for j in A:
            actions[j] = 0.5
        policy[s] = actions
    policy[0] = {0: 0.0, 1: 0.0}
    policy[6] = {0: 0.0, 1: 0.0}

    # Algorithme
    threshold = 0.0000001

    # Création et intialisation de la value function
    value_function = {}
    for s in S:
        value_function[s] = rand.random()
    value_function[0] = 0.0
    value_function[6] = 0.0

    while True:
        delta = 0
        for s in S:
            v = value_function[s]
            value_function[s] = 0.0
            for a in A:
                total = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total += mdp_env.transition_probability(s, a, s_p, r) * (R[r] + 0.999 * value_function[s_p])
                total *= policy[s][a]
                value_function[s] += total
            delta = max(delta, np.abs(v - value_function[s]))
        if delta < threshold:
            break
    return value_function


def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """

    # Création de l'environnement
    mdp_env = LineWorldMDP()
    S = mdp_env.states()
    A = mdp_env.actions()
    R = mdp_env.rewards()

    # Algorithme
    threshold = 0.0000001

    # Création et intialisation de la value function
    value_function = {}
    for s in S:
        value_function[s] = rand.random()
    value_function[0] = 0.0
    value_function[6] = 0.0

    policy = {}
    for s in mdp_env.states():
        actions = {}
        for j in mdp_env.actions():
            actions[j] = rand.random()
        policy[s] = actions

    for s in S:
        sum = np.sum(list(policy[s].values()))
        if sum != 0:
            for a in A:
                policy[s][a] /= sum

    for k in policy[0].keys():
        policy[0][k] = 0.0
        policy[6][k] = 0.0

    while True:
        while True:
            delta = 0
            for s in S:
                v = value_function[s]
                value_function[s] = 0.0
                A = mdp_env.actions()
                for a in A:
                    total = 0.0
                    for s_p in S:
                        R = mdp_env.rewards()
                        for r in range(len(R)):
                            total += mdp_env.transition_probability(s, a, s_p, r) * (R[r] + 0.999 * value_function[s_p])
                    total *= policy[s][a]
                    value_function[s] += total
                delta = max(delta, np.abs(v - value_function[s]))
            if delta < threshold:
                break

        stable = True
        for s in S:
            old_policy = policy[s].copy()
            best_a = -1
            best_a_score = -9999999
            for a in A:
                total = 0
                for s_p in S:
                    for r in range(len(R)):
                        total += mdp_env.transition_probability(s, a, s_p, r) * (R[r] + 0.999 * value_function[s_p])
                if total > best_a_score:
                    best_a = a
                    best_a_score = total
            for k in policy[s].keys():
                policy[s][k] = 0.0
            policy[s][best_a] = 1.0
            if old_policy != policy[s]:
                stable = False
        if stable:
            return policy, value_function


def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # Création de l'environnement
    mdp_env = LineWorldMDP()
    S = mdp_env.states()
    A = mdp_env.actions()
    R = mdp_env.rewards()

    # Algorithme
    threshold = 0.0000001

    # Création et intialisation de la value function
    value_function = {}
    for s in S:
        value_function[s] = rand.random()
    value_function[0] = 0.0
    value_function[6] = 0.0

    policy = {}

    while True:
        delta = 0
        best_value_function = {}
        for s in S:
            best_value_function[s] = 0.0
        for s in S:
            v = value_function[s]
            max_reward = 0
            for a in A:
                total = 0
                for s_p in S:
                    for r in range(len(R)):
                        total += mdp_env.transition_probability(s, a, s_p, r) * (R[r] + 0.999 * value_function[s_p])

                max_val = max(max_reward, total)

                if best_value_function[s] < total:
                    actions = {}
                    for a_s in A:
                        if a_s == a:
                            actions[a] = 1.0
                        else:
                            actions[a_s] = 0.0
                    policy[s] = actions

            best_value_function[s] = max_val
            delta = max(delta, abs(value_function[s] - best_value_function[s]))
        value_function = best_value_function

        if delta < threshold:
            break

    return policy, value_function


def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """

    # Création de l'environnement
    mdp_env = GridWorldMDP()
    S = mdp_env.states()
    R = mdp_env.rewards()
    A = mdp_env.actions()

    # Création de la policy
    policy = {}
    for s in S:
        actions = {}
        for j in A:
            actions[j] = 0.25
        policy[(s[0], s[1])] = actions

    policy[(0, 0)] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    policy[(4, 4)] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    # for x in range(5):
    #     for y in range(5):
    #         if x == 4:
    #             policy[(x, y)] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    # Création et intialisation de la value function
    value_function = {}
    for s in S:
        value_function[(s[0], s[1])] = rand.random()
    value_function[(0, 0)] = 0.0
    value_function[(4, 4)] = 0.0

    # Algorithme
    threshold = 0.0000001

    while True:
        delta = 0
        for s in S:
            v = value_function[(s[0], s[1])]
            value_function[(s[0], s[1])] = 0.0
            for a in A:
                total = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total += mdp_env.transition_probability(s, a, s_p, r) * (
                                R[r] + 0.999 * value_function[(s_p[0], s_p[1])])

                total *= policy[(s[0], s[1])][a]
                value_function[(s[0], s[1])] += total
            delta = max(delta, abs(v - value_function[(s[0], s[1])]))
        if delta < threshold:
            break
    return value_function


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # Création de l'environnement
    mdp_env = GridWorldMDP()
    S = mdp_env.states()
    R = mdp_env.rewards()
    A = mdp_env.actions()


    # Création de la policy
    policy = {}
    for s in S:
        actions = {}
        for j in A:
            actions[j] = rand.random()
        policy[(s[0], s[1])] = actions

    for s in S:
        sum = np.sum(list(policy[(s[0], s[1])].values()))
        if sum != 0:
            for a in A:
                policy[(s[0], s[1])][a] /= sum

    for k in policy[(0,0)].keys():
        policy[(0,0)][k] = 0.0
        policy[(4,4)][k] = 0.0

    # Création et intialisation de la value function
    value_function = {}
    for s in S:
        value_function[(s[0], s[1])] = rand.random()
    value_function[(0, 0)] = 0.0
    value_function[(4, 4)] = 0.0

    # Algorithme
    threshold = 0.0000001


    while True:
        while True:
            delta = 0
            for s in S:
                v = value_function[(s[0], s[1])]
                value_function[(s[0], s[1])] = 0.0
                for a in A:
                    total = 0.0
                    for s_p in S:
                        for r in range(len(R)):

                            total += mdp_env.transition_probability(s, a, s_p, r) * (
                                    R[r] + 0.999 * value_function[(s_p[0], s_p[1])])

                    total *= policy[(s[0], s[1])][a]
                    value_function[(s[0], s[1])] += total
                delta = max(delta, abs(v - value_function[(s[0], s[1])]))
            if delta < threshold:
                break

        stable = True
        for s in S:
            old_policy = policy[(s[0], s[1])].copy()
            best_a = -1
            best_a_score = -9999999
            for a in A:
                total = 0
                for s_p in S:
                    for r in range(len(R)):
                        total += mdp_env.transition_probability(s, a, s_p, r) * (R[r] + 0.999 * value_function[(s_p[0], s_p[1])])
                if total > best_a_score:
                    best_a = a
                    best_a_score = total
            for k in policy[(s[0], s[1])].keys():
                policy[(s[0], s[1])][k] = 0.0
            policy[(s[0], s[1])][best_a] = 1.0
            if old_policy != policy[(s[0], s[1])]:
                stable = False
        if stable:
            return policy, value_function


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # Création de l'environnement
    mdp_env = GridWorldMDP()
    S = mdp_env.states()
    R = mdp_env.rewards()
    A = mdp_env.actions()

    # Création de la policy
    policy = {}

    # Création et intialisation de la value function
    value_function = {}
    for s in S:
        value_function[(s[0], s[1])] = rand.random()
    value_function[(0, 0)] = 0.0
    value_function[(4, 4)] = 0.0

    # Algorithme
    threshold = 0.0000001

    while True:
        delta = 0
        best_value_function = {}
        for s in S:
            best_value_function[(s[0], s[1])] = 0.0
        for s in S:
            v = value_function[(s[0], s[1])]
            max_reward = 0
            for a in A:
                total = 0
                for s_p in S:
                    for r in range(len(R)):
                        total += mdp_env.transition_probability(s, a, s_p, r) * (R[r] + 0.999 * value_function[(s_p[0], s_p[1])])

                max_val = max(max_reward, total)

                if best_value_function[(s[0], s[1])] < total:
                    actions = {}
                    for a_s in A:
                        if a_s == a:
                            actions[a] = 1.0
                        else:
                            actions[a_s] = 0.0
                    policy[(s[0], s[1])] = actions

            best_value_function[(s[0], s[1])] = max_val
            delta = max(delta, abs(value_function[(s[0], s[1])] - best_value_function[(s[0], s[1])]))
        value_function = best_value_function

        if delta < threshold:
            break

    return policy, value_function


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    S = env.states()
    A = env.actions()
    R = env.rewards()

    # Création de la policy
    policy = {}
    for s in S:
        actions = {}
        if env.is_state_terminal(s):
            for j in A:
                actions[j] = 0.0
        else:
            for j in A:
                actions[j] = 1/len(A)
        policy[s] = actions

    # Algorithme
    threshold = 0.0000001

    # Création et intialisation de la value function
    value_function = {}
    for s in S:
        value_function[s] = rand.random()
        if env.is_state_terminal(s):
            value_function[s] = 0.0

    while True:
        delta = 0
        for s in S:
            v = value_function[s]
            value_function[s] = 0.0
            for a in A:
                total = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total += env.transition_probability(s, a, s_p, r) * (R[r] + 0.999 * value_function[s_p])
                total *= policy[s][a]
                value_function[s] += total
            delta = max(delta, np.abs(v - value_function[s]))
        if delta < threshold:
            break
    return value_function


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    S = env.states()
    A = env.actions()
    R = env.rewards()

    # Création de la policy
    policy = {}
    for s in S:
        actions = {}
        if env.is_state_terminal(s):
            for j in A:
                actions[j] = 0.0
        else:
            for j in A:
                actions[j] = rand.random()
        policy[s] = actions

    for s in S:
        sum = np.sum(list(policy[s].values()))
        if sum != 0:
            for a in A:
                policy[s][a] /= sum

    # Algorithme
    threshold = 0.0000001

    # Création et intialisation de la value function
    value_function = {}
    for s in S:
        value_function[s] = rand.random()
        if env.is_state_terminal(s):
            value_function[s] = 0.0

    while True:
        while True:
            delta = 0
            for s in S:
                v = value_function[s]
                value_function[s] = 0.0
                for a in A:
                    total = 0.0
                    for s_p in S:
                        for r in range(len(R)):

                            total += env.transition_probability(s, a, s_p, r) * (
                                    R[r] + 0.999 * value_function[s_p])

                    total *= policy[s][a]
                    value_function[s] += total
                delta = max(delta, abs(v - value_function[s]))
            if delta < threshold:
                break

        stable = True
        for s in S:
            old_policy = policy[s].copy()
            best_a = -1
            best_a_score = -9999999
            for a in A:
                total = 0
                for s_p in S:
                    for r in range(len(R)):
                        total += env.transition_probability(s, a, s_p, r) * (R[r] + 0.999 * value_function[s_p])
                if total > best_a_score:
                    best_a = a
                    best_a_score = total
            for k in policy[s].keys():
                policy[s][k] = 0.0
            policy[s][best_a] = 1.0
            if old_policy != policy[s]:
                stable = False
        if stable:
            return policy, value_function


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()

    # Création de l'environnement
    S = env.states()
    R = env.rewards()
    A = env.actions()

    # Création de la policy
    policy = {}

    # Création et intialisation de la value function
    value_function = {}
    for s in S:
        value_function[s] = rand.random()
        if env.is_state_terminal(s):
            value_function[s] = 0.0

    # Algorithme
    threshold = 0.0000001

    while True:
        delta = 0
        best_value_function = {}
        for s in S:
            best_value_function[s] = 0.0
        for s in S:
            v = value_function[s]
            max_reward = 0
            for a in A:
                total = 0
                for s_p in S:
                    for r in range(len(R)):
                        total += env.transition_probability(s, a, s_p, r) * (
                                    R[r] + 0.999 * value_function[s_p])

                max_val = max(max_reward, total)
                print(total)

                if best_value_function[s] < total:
                    actions = {}
                    for a_s in A:
                        if a_s == a:
                            actions[a] = 1.0
                        else:
                            actions[a_s] = 0.0
                    print(actions)
                    policy[s] = actions

            best_value_function[s] = max_val
            delta = max(delta, abs(value_function[s] - best_value_function[s]))
        value_function = best_value_function

        if delta < threshold:
            break

    return policy, value_function


def demo():
    # print(policy_evaluation_on_line_world())
    # print(policy_iteration_on_line_world())
    # print(value_iteration_on_line_world())
    #
    # print(policy_evaluation_on_grid_world())
    # print(policy_iteration_on_grid_world())
    # print(value_iteration_on_grid_world())

    # print(policy_evaluation_on_secret_env1())
    # print(policy_iteration_on_secret_env1())

    print(value_iteration_on_secret_env1())


demo()
