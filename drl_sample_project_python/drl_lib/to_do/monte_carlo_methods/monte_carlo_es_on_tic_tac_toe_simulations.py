import json
import numpy as np
import os
import time
from drl_sample_project_python.drl_lib.to_do.monte_carlo_methods.TicTacToeEnv import TicTacToeEnv
from drl_sample_project_python.drl_lib.do_not_touch.single_agent_env_wrapper import Env2
from menu import Menu
from pathlib import Path

path_to_backups = Path("backups").absolute().as_posix()
path_to_backups_for_env2 = Path("backups_secret_envs").absolute().as_posix()


def play_with_agent():
    policies = sorted(os.listdir(path_to_backups))
    choice = -1
    while(choice < 1 or choice > (len(policies) + 1)):
        print("# Choose a policy #\n")
        for k, policy in enumerate(policies):
            print(f"{k + 1}. {policy.removeprefix('.json')}")
        choice = int(input(">> "))
    print("Choosed policy :", policies[choice - 1], "\n\n")
    with open(Path(path_to_backups / Path(policies[choice - 1])), "r") as backup_file:
        backup_content = json.load(backup_file)
        S = np.array(backup_content["S"])
        pi = backup_content["pi"]
    tictactoe_env = TicTacToeEnv(is_random_player=False, S=S)
    tictactoe_env.reset()
    print("******************** STARTING ********************")
    print("************* Agent is O, you are X **************\n")
    while not tictactoe_env.is_game_over():
        if tictactoe_env.state_id() in pi:
            pi_s = pi[str(tictactoe_env.state_id())]
            aap = dict()
            for a in tictactoe_env.available_actions_ids():
                aap[str(a)] = pi_s[str(a)]
            a_max = int(max(aap, key=aap.get))
            if a_max != int(max(pi_s, key=pi_s.get)):
                print("- Agent said : I wanted to play at position", int(max(pi_s, key=pi_s.get)) + 1, "but it was already taken")
        else:
            print("- Agent said : I played randomly because I did not find this state in my policy")
            a_max = np.random.choice(tictactoe_env.available_actions_ids())
        print("****** Agent played in position", a_max + 1, "******")
        tictactoe_env.act_with_action_id(a_max)
        time.sleep(1)
    if tictactoe_env.score() == 1.0:
        tictactoe_env.view()
        print("*********** The agent won ! ***********")
    if tictactoe_env.score() == -1.0:
        print("*********** You won ! ***********")
    if tictactoe_env.score() == 0.0:
        print("*********** Draw 0-0 ! ***********")
    print("Restart ? (O / N)")
    if input(">> ") == "O":
        os.system("cls")
        play_with_agent()
    exit(0)


def watch_the_agent_paying_against_a_random_player():
    policies = sorted(os.listdir(path_to_backups))
    choice = -1
    while(choice < 1 or choice > (len(policies) + 1)):
        print("# Choose a policy #\n")
        for k, policy in enumerate(policies):
            print(f"{k + 1}. {policy.removeprefix('.json')}")
        choice = int(input(">> "))
    print("Choosed policy :", policies[choice - 1], "\n\n")
    with open(Path(path_to_backups / Path(policies[choice - 1])), "r") as backup_file:
        backup_content = json.load(backup_file)
        S = np.array(backup_content["S"])
        pi = backup_content["pi"]
    tictactoe_env = TicTacToeEnv(is_random_player=True, S=S, watch_the_game=True)
    tictactoe_env.reset()
    print("******************** STARTING ********************")
    print("********* Agent is O, random player is X *********\n")
    while not tictactoe_env.is_game_over():
        if tictactoe_env.state_id() in pi:
            pi_s = pi[str(tictactoe_env.state_id())]
            aap = dict()
            for a in tictactoe_env.available_actions_ids():
                aap[str(a)] = pi_s[str(a)]
            a_max = int(max(aap, key=aap.get))
            if a_max != int(max(pi_s, key=pi_s.get)):
                print("- Agent said : I wanted to play at position", int(max(pi_s, key=pi_s.get)) + 1,
                      "but it was already taken")
        else:
            print("- Agent said : I played randomly because I did not find this state in my policy")
            a_max = np.random.choice(tictactoe_env.available_actions_ids())
        print("****** Agent played in position", a_max + 1, "******")
        tictactoe_env.act_with_action_id(a_max)
        time.sleep(2)
    if tictactoe_env.score() == 1.0:
        tictactoe_env.view()
        print("*********** The agent won ! ***********")
    if tictactoe_env.score() == -1.0:
        print("*********** The random player won ! ***********")
    if tictactoe_env.score() == 0.0:
        print("*********** Draw 0-0 ! ***********")
    print("Restart ? (O / N)")
    if input(">> ") == "O":
        os.system("cls")
        watch_the_agent_paying_against_a_random_player()
    exit(0)


def stats_agent_paying_against_a_random_player():
    policies = sorted(os.listdir(path_to_backups))
    choice = -1
    while(choice < 1 or choice > (len(policies) + 1)):
        print("# Choose a policy #\n")
        for k, policy in enumerate(policies):
            print(f"{k + 1}. {policy.removeprefix('.json')}")
        choice = int(input(">> "))
    print("Choosed policy :", policies[choice - 1], "\n\n")
    with open(Path(path_to_backups / Path(policies[choice - 1])), "r") as backup_file:
        backup_content = json.load(backup_file)
        S = np.array(backup_content["S"])
        pi = backup_content["pi"]
    tictactoe_env = TicTacToeEnv(is_random_player=True, S=S, watch_the_game=False)
    print("Enter the number of games ")
    iterations = int(input(">> "))
    won, losses, drawn = 0, 0, 0
    for i in range(iterations):
        tictactoe_env.reset()
        while not tictactoe_env.is_game_over():
            if tictactoe_env.state_id() in pi:
                pi_s = pi[str(tictactoe_env.state_id())]
                aap = dict()
                for a in tictactoe_env.available_actions_ids():
                    aap[str(a)] = pi_s[str(a)]
                a_max = int(max(aap, key=aap.get))
            else:
                a_max = np.random.choice(tictactoe_env.available_actions_ids())
            tictactoe_env.act_with_action_id(a_max)
        if tictactoe_env.score() == 1.0:
            print("Game", i, ": Agent won")
            won += 1
        if tictactoe_env.score() == -1.0:
            print("Game", i, ": Agent lost")
            losses += 1
        if tictactoe_env.score() == 0.0:
            print("Game", i, ": Drawn")
            drawn += 1
    print("\n###########################")
    print("Won :", won, "games |", round((won / iterations) * 100, 2), "%")
    print("Lost :", losses, "games |", round((losses / iterations) * 100, 2), "%")
    print("Draw :", drawn, "games |", round((drawn / iterations) * 100, 2), "%")
    print("###########################")
    exit(0)


def stats_agent_paying_against_a_random_player_in_env2():
    policies = sorted(os.listdir(path_to_backups_for_env2))
    choice = -1
    while(choice < 1 or choice > (len(policies) + 1)):
        print("# Choose a policy #\n")
        for k, policy in enumerate(policies):
            print(f"{k + 1}. {policy.removeprefix('.json')}")
        choice = int(input(">> "))
    print("Choosed policy :", policies[choice - 1], "\n\n")
    with open(Path(path_to_backups_for_env2 / Path(policies[choice - 1])), "r") as backup_file:
        backup_content = json.load(backup_file)
        pi = backup_content["pi"]
    env = Env2()
    print("Enter the number of games ")
    iterations = int(input(">> "))
    won, losses, drawn = 0, 0, 0
    for i in range(iterations):
        env.reset()
        while not env.is_game_over():
            # if env.state_id() in pi:
            pi_s = pi[str(env.state_id())]
            aap = dict()
            for a in env.available_actions_ids():
                aap[str(a)] = pi_s[str(a)]
            a_max = int(max(aap, key=aap.get))
            # else:
            #     a_max = np.random.choice(env.available_actions_ids())
            env.act_with_action_id(a_max)
        if env.score() == 1.0:
            print("[ENV2] Game", i, ": Agent won")
            won += 1
        if env.score() == -1.0:
            print("[ENV2] Game", i, ": Agent lost")
            losses += 1
        if env.score() == 0.0:
            print("[ENV2] Game", i, ": Drawn")
            drawn += 1
    print("\n###########################")
    print("Won :", won, "games |", round((won / iterations) * 100, 2), "%")
    print("Lost :", losses, "games |", round((losses / iterations) * 100, 2), "%")
    print("Draw :", drawn, "games |", round((drawn / iterations) * 100, 2), "%")
    print("###########################")
    exit(0)


if __name__ == "__main__":
    # Menu principal
    principal_menu = Menu(title="## Tic Tac Toe with Monte Carlo Methods ##")
    principal_menu.set_options([("Play with agent",
                                 play_with_agent),
                                ("Watch a game of the agent against a random player",
                                 watch_the_agent_paying_against_a_random_player),
                                ("Launch n games of agent against a random player and get stats",
                                 stats_agent_paying_against_a_random_player),
                                ("ENV2 - Launch n games of agent against a random player and get stats",
                                 stats_agent_paying_against_a_random_player_in_env2)
                                ])
    principal_menu.open()
