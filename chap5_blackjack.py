import numpy as np
from typing import Callable
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# actions: hit or stand(strike)
ACTION_HIT = 0
ACTION_STAND = 1  # strike
ACTIONS = [ACTION_HIT, ACTION_STAND]

# policy for player
POLICY_PLAYER = np.zeros(22, dtype=np.int)
for i in range(12, 22):
    POLICY_PLAYER[i] = ACTION_HIT
POLICY_PLAYER[20] = ACTION_STAND
POLICY_PLAYER[21] = ACTION_STAND


def target_policy_player(usable_ace_player, player_sum, dealer_card) -> np.int:
    """function form of target policy of player"""
    return POLICY_PLAYER[player_sum]


# policy for dealer
# sticks on any sum of 17 or greater, and hit otherwise
POLICY_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STAND


def get_card() -> int:
    """get a new card"""
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card


def card_value(card: int) -> int:
    """get the value of the card; 11 for ace"""
    return 11 if card == 1 else card


def play(policy_player: Callable, initial_state: list = None, initial_action=None) -> (list, int, list):
    """
    play a game

    :param policy_player: specify policy for player
    :param initial_state: [whether player has a usable Ace, sum of player's card, one card of dealer]
    :param initial_action: the initial action

    :return state, reward, player_trajectory
    state: list = [usable_ace_player: bool, player_sum: int, dealer_card1: int]
    reward: int = -1 or 0 or 1
    player_trajectory: list = [(usable_ace_player: bool, player_sum: int, dealer_card1: int), action: int]
    """
    # initiate sum of a player
    player_sum = 0

    # trajectory of player
    player_trajectory = []

    # whether player uses Ace as 11
    usable_ace_player = False

    # dealer status
    dealer_card1 = 0
    dealer_card2 = 0
    usable_ace_dealer = False

    if initial_state is None:
        # generate a random initial state
        while player_sum < 12:
            # if sum is less than 12, always hit
            card = get_card()
            player_sum += card_value(card)
            # if sum > 21, he may hold one or two aces (11+11)
            if player_sum > 21:
                assert player_sum == 22
                # last card must be 1 instead of 11
                player_sum -= 10
            else:
                usable_ace_player |= (1 == card)

        # initialize cards for dealer, suppose the dealer will show the first card he gets
        dealer_card1 = get_card()
        dealer_card2 = get_card()

    else:
        # use specified initial state
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()

    # initial state of the game
    state = [usable_ace_player, player_sum, dealer_card1]

    # initialize dealer's sum
    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)
    usable_ace_dealer = 1 in (dealer_card1, dealer_card2)
    # if the dealer's sum > 21, he must hold two aces
    if dealer_sum > 21:
        assert dealer_sum == 22
        # last card must be 1 instead of 11
        dealer_sum -= 10
    # final check of the sum of player and dealer
    assert dealer_sum <= 21
    assert player_sum <= 21

    # game starts

    # player's turn
    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            # get action based on current sum
            action = policy_player(usable_ace_player, player_sum, dealer_card1)

        # track player's trajectory for importance sampling
        player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])
        if action == ACTION_STAND:
            break
        # if HIT, git new card
        card = get_card()
        ace_count = int(usable_ace_player)
        if card == 1:
            ace_count += 1
        player_sum += card_value(card)

        # if player has a usable ace, use it as 1 to avoid busting and continue
        while player_sum > 21 and ace_count:
            player_sum -= 10
            ace_count -= 1
        # player buster
        if player_sum > 21:
            return state, -1, player_trajectory
        assert player_sum <= 21
        usable_ace_player = (ace_count == 1)

    # dealer's turn
    while True:
        # get actioin based on current sum
        action = POLICY_DEALER[dealer_sum]
        if action == ACTION_STAND:
            break
        # if hit, get a new card
        new_card = get_card()
        ace_count = int(usable_ace_dealer)
        if new_card == 1:
            ace_count += 1
        dealer_sum += card_value(new_card)
        # if dealer has usable ace, use it as 1 to avoid busting and continue
        while dealer_sum > 21 and ace_count:
            dealer_sum -= 10
            ace_count -= 1
        # dealer busts
        if dealer_sum > 21:
            return state, -1, player_trajectory
        usable_ace_dealer = (ace_count == 1)

    # compare the sum between player and dealer
    # make sure both sums are smaller than 21
    assert player_sum <= 21 and dealer_sum <= 21
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory


# Monte Carlo Sampling with On-Policy
def monte_carlo_on_policy(episodes: int):
    states_usable_ace = np.zeros((10, 10))
    # initialize counts to 1 to avoid 0 being divided
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    states_no_usable_ace_count = np.ones((10, 10))
    for i in tqdm(range(0, episodes)):
        _, reward, player_trajectory = play(target_policy_player)
        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
            # why??
            player_sum -= 12
            dealer_card -= 1
            if usable_ace:
                states_usable_ace[player_sum, dealer_card] += reward
                states_usable_ace_count[player_sum, dealer_card] += 1
            else:
                states_no_usable_ace[player_sum, dealer_card] += reward
                states_no_usable_ace_count[player_sum, dealer_card] += 1
    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count


def figure_5_1():
    states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(10000)
    states_usable_ace_2, states_no_usable_ace_2 = monte_carlo_on_policy(500000)

    states = [states_usable_ace_1,
              states_usable_ace_2,
              states_no_usable_ace_1,
              states_no_usable_ace_2]
    titles = ["Usable Ace, 10000 Episodes",
              "Usable Ace, 500000 Episodes",
              "No Usable Ace, 10000 Episodes",
              "No Usable Ace, 500000 Episodes"]
    # _, axes = plt.subplots(2, 2, figsize=(40, 30))
    # plt.subplots_adjust(wspace=0.1, hspace=0.2)
    # axes = axes.flatten()
    #
    # for state, title, axis in zip(states, titles, axes):
    #     fig = sns.heatmap(np.flipud(state), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
    #                       yticklabels=list(reversed(range(12, 22))))
    #     fig.set_ylabel('player sum', fontsize=30)
    #     fig.set_xlabel('dealer showing', fontsize=30)
    #     fig.set_title(title, fontsize=30)
    #
    #     plt.savefig('figure_5_1.png')
    #     plt.close()

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for state, title, axis in zip(states, titles, axes):
        fig = sns.heatmap(np.flipud(state), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig('figure_5_1.png')
    plt.close()


if __name__ == '__main__':
    figure_5_1()
