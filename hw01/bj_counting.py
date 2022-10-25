import gym
from gym import spaces
from gym.utils import seeding


counter = 0 #переменная для подсчета карт

def cmp(a, b):
    return float(a > b) - float(a < b)


def create_deck():
    global deck
    # 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
    deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4


def reset_deck():
    global deck
    global counter
    #как гвоорится в разных источниках, обычно колоду начинают перемешивать когда в колоде(шузе) ~ 50% использованых карт, однако этот порог слишком уж большой
    # для обучения алгоритма, чтобы подсчет вообще что то значал, поэтому возьмем отметку в 75% использованой колоды как тригер для перемешивания.
    if len(deck) < 52*0.25:
        create_deck()
        counter = 0


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):

    def __init__(self, natural=False, sab=False, counting_method='halves'):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2), spaces.Discrete(32))
        )
        self.seed()
        self.natural = natural
        self.sab = sab
        # SRC: https://ru.wikipedia.org/wiki/Блэкджек#Подсчёт_карт
        if counting_method == 'halves': # метод половинки 
            self.cnt_mapping = {
                2: .5,
                3: 1,
                4: 1,
                5: 1.5,
                6: 1,
                7: .5,
                8: 0,
                9: -0.5,
                10: -1,
                1: -1
            }
        elif counting_method == 'PN': #метод "Плюс-Минус"
            self.cnt_mapping = {
                2: 1,
                3: 1,
                4: 1,
                5: 1,
                6: 1,
                7: 0,
                8: 0,
                9: 0,
                10: -1,
                1: -1
            }

        else:
            NotImplementedError('Current method is not emplemented yet')
        
        create_deck()

    def reset(self):
        self.dealer = self.draw_hand(self.np_random)
        self.player = self.draw_hand(self.np_random)
        return self._get_obs()
    
    
    def draw_card(self, np_random):
        global counter
        global deck
        reset_deck()
        card = int(np_random.choice(deck))
        counter += self.cnt_mapping[card]
        deck.remove(card)
        return card


    def draw_hand(self, np_random):
        return [self.draw_card(np_random), self.draw_card(np_random)]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit: add a card to players hand and return
            self.player.append(self.draw_card(self.np_random))
            if is_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
                
        else:  # stick: play out the dealers hand, and score
            if action == 2: # for double stake
                self.player.append(self.draw_card(self.np_random))
            terminated = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if action == 2: # for double stake
                reward *= 2
            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
            elif (
                not self.sab
                and self.natural
                and is_natural(self.player)
                and reward == 1.0
            ):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
            
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), counter)