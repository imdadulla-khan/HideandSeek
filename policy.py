import random
import numpy
import operator

valid_action = ["LEFT", "RIGHT", "UP", "DOWN"]
rot = ["r_rot", "l_rot"]

block_size = 32

class Policy(object):
    # for all players

    def __init__(self, player):
        self.player = player
        self.q_table = {}
        self.reward = 0
        self.alpha = 0.3
        self.gamma = 0.1
        self.epsilon = 0.1  
        self.penalties = []
        self.total_reward = 0.0
        self.counts = 0.0


    def reset(self):
        self.reward = 0

    def get_action(self):
        max_q = 0
        self.state_unformatted = self.player.get_state()
        self.curr_distance = self.player.distance()
        self.state = (
        self.state_unformatted[0], self.state_unformatted[1], round(self.curr_distance, 3), round(self.total_reward, 3))
        # head x, head y, dist from original cords, total reward till now
        if not self.state in self.q_table:
            self.q_table[self.state] = {ac: 0 for ac in valid_action}  # updating (left right up down ) for each state
            print(self.q_table)

        action = random.choice(valid_action)
        rotation = random.choice(rot)
        # random_action = action

        # Exploration v/s exploitation
        if numpy.random.random() > self.epsilon:
            if len(set(self.q_table[self.state].values())) == 1:
                pass
            else:
                action = max(self.q_table[self.state].items(), key=operator.itemgetter(1))[0]
                print(action)
        return action, rotation

    def update(self, action, reward):

        self.total_reward += reward

        print(self.total_reward)
        self.next_state_unformatted = self.player.get_state()
        self.curr_distance = self.player.distance()
        self.next_state = (self.next_state_unformatted[0], self.next_state_unformatted[1], round(self.curr_distance, 3),
                           round(self.total_reward, 3))
        # x coordinate,y coordinate, Distance from original cords, Cummulative Reward

        if self.next_state not in self.q_table:  # Checking if q_values has already been calculated 
            self.q_table[self.next_state] = {ac: 0 for ac in valid_action}

        old_q_value = self.q_table[self.state][action]  # Learning policy based on state, action, reward


        next_max = max(self.q_table[self.next_state].values())  # Maximum q_value for next_state actions

        new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (
                reward + self.gamma * next_max)  # Calculating the q_value for the next_max action.
        self.q_table[self.state][action] = new_q_value
        print("Agent.update(): state = {}".format(self.state))  

