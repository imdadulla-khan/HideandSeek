from Player import *
import random
import math
import pickle
from policy import *

# Intializing seeker
class Seeker(Player):
    initial_positions = [(280, 250), (400, 400), (250, 300), (500, 100), (620, 450)]

    def __init__(self):
        # super().__init__()
        rect = pygame.Rect(0, 0, 32, 32)
        rect.center = random.choice(Seeker.initial_positions)
        self.rect = rect
        self.original_cords = [self.rect.x, self.rect.y]
        self.angle = 0
        self.checkwalls(850)
        self.orientation = 0
        self.movement_type = [1, 3][0]
        self.game_no = 0
        self.game_prevno = 0
        self.dist_covered = 0
        self.agent_seeker = Policy(self)
        seeker_pickle = open("seeker_qtable.pickle","rb")
        self.agent_seeker.q_table=pickle.load(seeker_pickle)

    def distance_reward(self, hider_cords):
        x = self.rect.x
        y = self.rect.y
        for i in hider_cords:
            x1 = [x, y]
            x2 = [i[0], i[1]]
            dist = math.dist(x1, x2)
            if dist > 100:
                return -0.01
            else:
                return dist / 1000

    def distance(self):
        # Reward function for more distance covered
        x1 = self.original_cords
        x2 = [self.rect.x, self.rect.y]
        self.dist_covered = math.dist(x1, x2)
        return self.dist_covered

    def area_coverage(self):
        # Reward function for covering more area
        self.dist_covered = self.distance()
        if self.dist_covered < 100:
            coverage_reward = - self.dist_covered / 1000
        else:
            coverage_reward = self.dist_covered / 1000
        return coverage_reward

    def wall_collision(self):
        wall_info = self.is_wall_nearby()
        wall_reward = 0
        for key in wall_info:
            if wall_info[key]:
                wall_reward = -0.001
        return wall_reward

    def reward(self, hider_objs, hider_cords):
        reward = -1
        co_list = []
        self.v_startpoints, self.v_endpoints = Vision(45,180,50).get_intersect(self.rect.center, self.near_walls,
                                                                      self.orientation)
        loser_hider = 0
        seek_rew=0
        for h in hider_objs:
            for line in self.vision: 
                start = h.rect.clipline(line)
                if start:
                    seek_rew = seek_rew + 1000
                    print("hey, they collided!!!!!!!!!!! ", seek_rew)
                    loser_hider = h
                    self.game_prevno = self.game_no
                    self.game_no += 1
                    break
        reward = seek_rew # If hider spots the seeker
        reward += self.distance_reward(hider_cords)  # If distance from seeker is higher then positive reward and vice versa
        reward += self.area_coverage()   # Reward due to distance from intial coordinates
        reward += self.wall_collision()  # If continious collisons with walls
        # print(reward)
        return reward, loser_hider