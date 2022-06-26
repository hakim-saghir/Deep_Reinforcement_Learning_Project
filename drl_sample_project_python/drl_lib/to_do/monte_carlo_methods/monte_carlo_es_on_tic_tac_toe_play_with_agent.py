import json
import numpy as np
import os
import time
from drl_sample_project_python.drl_lib.to_do.monte_carlo_methods.TicTacToeEnv import TicTacToeEnv
from menu import Menu
from pathlib import Path

path_to_backups = Path("backups").absolute().as_posix()


def play_with_agent(policy_src):
    with open(policy_src, "r") as backup_file:
        backup_content = json.load(backup_file)
        S = np.array(backup_content["S"])
        pi = backup_content["pi"]
    tictactoe_env = TicTacToeEnv(is_random_player=False, S=S)
    tictactoe_env.reset()
    print("******************** STARTING ********************")
    print("************* Agent is O, you are X **************\n")
    while not tictactoe_env.is_game_over():
        pi_s = pi[str(tictactoe_env.state_id())]
        aap = dict()
        for a in tictactoe_env.available_actions_ids():
            aap[str(a)] = pi_s[str(a)]
        a_max = int(max(aap, key=aap.get))
        if a_max != int(max(pi_s, key=pi_s.get)):
            print("- Agent said : I wanted to play at position", int(max(pi_s, key=pi_s.get)) + 1, "but it was already taken")
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
        play_with_agent(policy_src)
    exit(0)


def watch_the_agent_paying_against_a_random_player(policy_src):
    with open(policy_src, "r") as backup_file:
        backup_content = json.load(backup_file)
        S = np.array(backup_content["S"])
        pi = backup_content["pi"]
    tictactoe_env = TicTacToeEnv(is_random_player=True, S=S, watch_the_game=True)
    tictactoe_env.reset()
    print("******************** STARTING ********************")
    print("********* Agent is O, random player is X *********\n")
    while not tictactoe_env.is_game_over():
        pi_s = pi[str(tictactoe_env.state_id())]
        aap = dict()
        for a in tictactoe_env.available_actions_ids():
            aap[str(a)] = pi_s[str(a)]
        a_max = int(max(aap, key=aap.get))
        if a_max != int(max(pi_s, key=pi_s.get)):
            print("- Agent said : I wanted to play at position", int(max(pi_s, key=pi_s.get)) + 1, "but it was already taken")
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
        watch_the_agent_paying_against_a_random_player(policy_src)
    exit(0)


def stats_agent_paying_against_a_random_player(policy_src):
    with open(policy_src, "r") as backup_file:
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
            pi_s = pi[str(tictactoe_env.state_id())]
            aap = dict()
            for a in tictactoe_env.available_actions_ids():
                aap[str(a)] = pi_s[str(a)]
            a_max = int(max(aap, key=aap.get))
            if a_max != int(max(pi_s, key=pi_s.get)):
                print("- Agent said : I wanted to play at position", int(max(pi_s, key=pi_s.get)) + 1, "but it was already taken")
            print("****** Agent played in position", a_max + 1, "******")
            tictactoe_env.act_with_action_id(a_max)
            time.sleep(2)
        if tictactoe_env.score() == 1.0:
            print("Game", i, ": Agent won")
            won += 1
        if tictactoe_env.score() == -1.0:
            print("Game", i, ": Agent lost")
            losses += 1
        if tictactoe_env.score() == 0.0:
            print("Game", i, ": Drawn")
            drawn += 1
    print("Won :", won, "games |", round((won / iterations) * 100, 2))
    print("Losses :", losses, "games |", round((losses / iterations) * 100, 2))
    print("Draws :", drawn, "games |", round((drawn / iterations) * 100, 2))


if __name__ == "__main__":

    # Sous-menu : Jouer avec l'agent
    play_with_agent_policy_choice_menu = Menu(title="# Choose a policy #")
    play_with_agent_policy_choice_menu.set_options([(backup.removesuffix(".json"),
                                                     lambda: play_with_agent(Path(path_to_backups / Path(backup))))
                                                    for backup in os.listdir(path_to_backups)])

    # Sous-menu : Regarder une partie avec un joueur random
    watch_agent_playing_policy_choice_menu = Menu(title="# Choose a policy #")
    watch_agent_playing_policy_choice_menu.set_options([(backup.removesuffix(".json"),
                                                         lambda: watch_the_agent_paying_against_a_random_player(Path(path_to_backups / Path(backup))))
                                                        for backup in os.listdir(path_to_backups)])

    # Sous-menu : Regarder une partie avec un joueur random
    stats_agent_playing_policy_choice_menu = Menu(title="# Choose a policy #")
    stats_agent_playing_policy_choice_menu.set_options([(backup.removesuffix(".json"),
                                                         lambda: stats_agent_paying_against_a_random_player(
                                                             Path(path_to_backups / Path(backup))))
                                                        for backup in os.listdir(path_to_backups) if "_env2" not in backup])

    # Menu principal
    principal_menu = Menu(title="## Tic Tac Toe with Monte Carlo Methods ##")
    principal_menu.set_options([("Play with agent",
                                 play_with_agent_policy_choice_menu.open),
                                ("Watch a game of the agent against a random player",
                                 watch_agent_playing_policy_choice_menu.open),
                                ("Launch n games of agent against a random player and get stats",
                                 stats_agent_playing_policy_choice_menu.open)])

    principal_menu.open()