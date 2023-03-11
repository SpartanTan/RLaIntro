import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Setting up actions

# actions: hit or stand
# action: request additional card
ACTION_HIT = 0  # type: int

# action: stop request card
ACTION_STAND = 1  # type: int

# list of actions
ACTIONS = [ACTION_HIT, ACTION_STAND]  # type: list

# Setting up policies

# policy for player
# From book "Consider the policy that sticks if the player’s sum is 20 or 21, and otherwise hits."

# the sum of the cards can be 0-21, thus,
# a 1*22 table, values are the actions to take in this state
# index| 0 1 2 3 ...
# sum  | 0 1 2 3 ...
# initialize the policy table with 0s, means taking ACTION_HIT at all states
POLICY_PLAYER = np.zeros(22, dtype=np.int)
# set the states 12-19 to HIT
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT
# sticks if sum is 20 or 21
POLICY_PLAYER[20] = ACTION_STAND
POLICY_PLAYER[21] = ACTION_STAND

# policy for dealer
# policy from book "he sticks on any sum of 17 or greater, and hits otherwise."
POLICY_DEALER = np.zeros(22, dtype=np.int)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STAND


#################################################
# Functions

def target_policy_player(usable_ace_player: bool, player_sum: int, dealer_card: int) -> int:
    """
    find the action from policy of the player
    @param usable_ace_player: if the player have one usable Ace
    @param player_sum: the sum of the cards of the player
    @param dealer_card: the faced card of dealer
    @return: an int, 0 for hit, 1 for stand
    """

    return POLICY_PLAYER[player_sum]


def behavior_policy_player(usable_ace_player, player_sum, dealer_card):
    """
    Returns an anction under behavior policy.
    This policy is defined as a random choice between stand and hit.
    @param usable_ace_player:
    @param player_sum:
    @param dealer_card:
    @return:
    """
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STAND
    return ACTION_HIT


def get_card():
    """
    get one card.
    Ace will be considered as 1.
    @return: an int from 1 to 10
    """
    # cards from 2 to 10, J,Q,K, Ace
    # randint(1,14), 1 is included, 14 is not included, thus 13 options in total
    # the return will be limited in 1-10, since JQK are considered as 10
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card


def card_value(card_id):
    """
    evaluate the card and return the real card value.
    Ace will be considered as 11.
    If card_id is 1, return 11. Otherwise, return card_id.

    @param card_id: card_id
    @return: real card value
    """
    return 11 if card_id == 1 else card_id


def play(policy_player, initial_state=None, initial_action=None):
    """
    One episode of the game.
    Giving two cards to player and dealer.
    Player actions until it stands or bust.
    Then dealer actions until it stands or busts.
    Compare the sum if none of them busts.
    @param policy_player: player policy function, returns the action of player
    @param initial_state: A given initial state, (usable_ace_player, player_sum, dealer_card1)
    @param initial_action: A given initial action
    @return: (state, result, player_trajectory)
            state: a list, [usable_ace_player, player_sum, dealer_card1]
            result: -1, 0, 1, representing the result of the game
            player_trajectory: (state, action) pair
    """
    # sum of player
    player_sum = 0

    # trajectory of player
    player_trajectory = []

    # whether player uses Ace as 11
    usable_ace_player = False

    # dealer status
    dealer_card1 = 0
    dealer_card2 = 0
    usable_ace_dealer = False

    # Initialization of the game
    # player and dealer will be given two cards
    if initial_state is None:
        while player_sum < 12:
            card = get_card()
            player_sum += card_value(card)

            # The previous sum should be smaller than 12
            # If the player's sum is larger than 21,
            # The only possible sum is 22
            # extreme case: 11+11
            if player_sum > 21:
                # verify if the sum is 22, it is the only possible result
                assert player_sum == 22
                # Thus the last card must be Ace, and so we can consider
                # one of the Ace's is 1
                player_sum -= 10
            else:
                # if player_sum not exceeds 21, then
                # if the last card is Ace, then this Ace is usable.
                usable_ace_player |= (1 == card)

            # initialize cards of dealer, suppose the dealer will show the first card
            dealer_card1 = get_card()
            dealer_card2 = get_card()
    else:
        # use specified initial state
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()

    # initialize the state of the game
    state = [usable_ace_player, player_sum, dealer_card1]

    # initialize dealer's sum
    dealer_sum = card_value(dealer_card1) + card_value(dealer_card2)
    usable_ace_dealer = 1 in (dealer_card1, dealer_card2)

    # if the dealer sum is larger than 21, which means he gets two Aces, 22
    if dealer_sum > 21:
        assert dealer_sum == 22
        # use one Ace as 1
        dealer_sum -= 10
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
        # structure: a union with an integer
        # [(usable_ace_player, player_sum, dealer_card1), action]
        player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])

        # if action is stand, then player's turn is over
        if action == ACTION_STAND:
            break

        # if hit, get a new card
        card = get_card()

        # Before taking the new card, player could have at most 1 usable Ace
        ace_count = int(usable_ace_player)
        if card == 1:
            ace_count += 1
        player_sum += card_value(card)

        # If the sum is larger than 21, and the player has usable Ace,
        # then use one of the Ace's as 1 to avoiding busting and continue
        while player_sum > 21 and ace_count:
            player_sum -= 10
            ace_count -= 1

        # player busts
        if player_sum > 21:
            return state, -1, player_trajectory
        assert player_sum <= 21
        usable_ace_player = (ace_count == 1)

    # dealer's turn
    while True:
        # get action based on current sum
        action = POLICY_DEALER[dealer_sum]

        # if the dealer's action is stand, his turn is over
        if action == ACTION_STAND:
            break

        # if hit, get dealer a new card
        new_card = get_card()
        ace_count = int(usable_ace_dealer)
        if new_card == 1:
            ace_count += 1
        dealer_sum += card_value(new_card)
        # if dealer has usable Ace and the sum is larger than 21
        while dealer_sum > 21 and ace_count:
            dealer_sum -= 10
            ace_count -= 1

        # dealer busts
        if dealer_sum > 21:
            return state, 1, player_trajectory
        usable_ace_dealer = (ace_count == 1)

    # compare the sum between player and dealer
    assert player_sum <= 21 and dealer_sum <= 21
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory


def monte_carlo_on_policy(episodes: int):
    """
    Monte-carlo method for calculating the state-value function of a static policy.
    @rtype: object
    """
    states_usable_ace = np.zeros((10, 10))
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    states_no_usable_ace_count = np.ones((10, 10))

    for i in tqdm(range(0, episodes)):
        _, reward, player_trajectory = play(target_policy_player)
        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
            player_sum -= 12
            dealer_card -= 1
            if usable_ace:
                states_usable_ace_count[player_sum, dealer_card] += 1
                states_usable_ace[player_sum, dealer_card] += reward
            else:
                states_no_usable_ace_count[player_sum, dealer_card] += 1
                states_no_usable_ace[player_sum, dealer_card] += reward
    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count


def monte_carlo_es(episodes):
    """
    Monte Carlo Exploring Starts
    """

    # (playerSum, dealerCard, usableAce, action)
    state_action_values = np.zeros((10, 10, 2, 2))
    state_action_pair_count = np.ones((10, 10, 2, 2))

    # behavior policy is greedy
    def behavior_policy(usable_ace: bool, player_sum: int, dealer_card: int):
        usable_ace = int(usable_ace)
        player_sum -= 12
        dealer_card -= 1
        values_ = state_action_values[player_sum, dealer_card, usable_ace, :] / \
                  state_action_pair_count[player_sum, dealer_card, usable_ace, :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    for episode in tqdm(range(episodes)):
        # for each episode, use a random starting state and action
        # (usable_ace_player, player_sum, dealer_card1)
        initial_state = [bool(np.random.choice([0, 1])),
                         np.random.choice(range(12, 22)),
                         np.random.choice(range(1, 11))]
        initial_action = np.random.choice(ACTIONS)

        current_policy = behavior_policy if episode else target_policy_player
        _, reward, trajectory = play(current_policy, initial_state, initial_action)
        first_visit_check = set()
        for (usable_ace, player_sum, dealer_card), action in trajectory:
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            state_action = (usable_ace, player_sum, dealer_card, action)

            # in this episode, if the state-action is visited before
            # then jump
            if state_action in first_visit_check:
                continue
            first_visit_check.add(state_action)

            # in this episode, if the state-action pair is not first-visited
            # update values of state-action pairs
            # The reward will be accumulated from all the episodes
            state_action_values[player_sum, dealer_card, usable_ace, action] += reward
            state_action_pair_count[player_sum, dealer_card, usable_ace, action] += 1

    return state_action_values / state_action_pair_count


def monte_carlo_off_policy(episodes):
    # (usable_ace_player, player_sum, dealer_card1)
    initial_state = [True, 13, 2]

    rhos = []
    returns = []

    for i in range(0, episodes):
        _, reward, player_trajectory = play(behavior_policy_player, initial_state, )

        # get the importance ratio
        # numerator: pi(Ak|Sk)
        # denominator: b(Ak|Sk)
        numerator = 1.0
        denominator = 1.0

        for (usable_ace, player_sum, dealer_card), action in player_trajectory:
            if action == target_policy_player(usable_ace, player_sum, dealer_card):
                denominator *= 0.5
            else:
                # if the action under b policy is different from the action
                # under pi policy, then pi(Ak|Sk) = 0, so that the numerator
                # is 0, then abort from this trajectory
                numerator = 0.0
                break

        # the importance sampling ratio
        rho = numerator / denominator
        rhos.append(rho)
        returns.append(reward)

    # after all the episodes
    rhos = np.asarray(rhos)
    returns = np.asarray(returns)
    weighted_returns = rhos * returns
    weighted_returns = np.add.accumulate(weighted_returns)
    rhos = np.add.accumulate(rhos)

    # ordinary importance sampling
    # V(s) = Sig(rhot * Gt)/|J(s)|
    # the state s is given by the initial condition
    # It means every time we start from this state, followed by policy b, then
    # termination. One episode is one time step of visiting S.
    # Thus the set J(s) stores 1....number_of_episodes
    ordinary_sampling = weighted_returns / np.arange(1, episodes + 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        weighted_sampling = np.where(rhos != 0, weighted_returns / rhos, 0)
    return ordinary_sampling, weighted_sampling


def figure_5_1():
    """
    Approximate state-value functions for the blackjack policy that sticks
    only or 21. Computed by MC policy evaluation
    """
    states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(10000)
    states_usable_ace_2, states_no_usable_ace_2 = monte_carlo_on_policy(500000)

    states = [states_usable_ace_1,
              states_usable_ace_2,
              states_no_usable_ace_1,
              states_no_usable_ace_2]

    titles = ['Usable Ace, 10000 Episodes',
              'Usable Ace, 500000 Episodes',
              'No Usable Ace, 10000 Episodes',
              'No Usable Ace, 500000 Episodes']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for state, title, axis in zip(states, titles, axes):
        fig = sns.heatmap(np.flipud(state), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.show()


def figure_5_2():
    """
    Monte carlo exploring starting method
    """
    state_action_values = monte_carlo_es(500000)

    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)

    # get the optimal policy
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)

    images = [action_usable_ace,
              state_value_usable_ace,
              action_no_usable_ace,
              state_value_no_usable_ace]

    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for image, title, axis in zip(images, titles, axes):
        fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.show()


def figure_5_3():
    """
    Off-policy Estimation of a Blackjack State Value
    """
    true_value = -0.27726
    episodes = 10000
    runs = 100

    error_ordinary = np.zeros(episodes)
    error_weighted = np.zeros(episodes)

    for i in tqdm(range(0, runs)):
        ordinary_sampling_, weighted_sampling_ = monte_carlo_off_policy(episodes)
        # get the squared error
        error_ordinary += np.power(ordinary_sampling_ - true_value, 2)
        error_weighted += np.power(weighted_sampling_ - true_value, 2)

    error_ordinary /= runs
    error_weighted /= runs

    plt.plot(np.arange(1, episodes + 1), error_ordinary, color='green', label='Ordinary Importance Sampling')
    plt.plot(np.arange(1, episodes + 1), error_weighted, color='red', label='Weighted Importance Sampling')
    plt.ylim(-0.1, 5)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel(f'Mean square error\n(average over {runs} runs)')
    plt.xscale('log')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # figure_5_1()
    # figure_5_2()
    figure_5_3()
