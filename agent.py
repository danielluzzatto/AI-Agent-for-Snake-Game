import torch
import random
import numpy as np
from game import snake_game_ai, Directions, Point
from collections import deque
from model import Linear_Qnet, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20


class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Qnet(11, 256, 3)  
        self.trainer = QTrainer(self.model, lr = LR, gamma =self.gamma)  
        self.total_games = 1000

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_r = game.direction == Directions.RIGHT
        dir_l = game.direction == Directions.LEFT
        dir_d = game.direction == Directions.DOWN
        dir_u = game.direction == Directions.UP

        state = [  # danger straight ahead
            (dir_r and game.check_collision(point_r)) or
            (dir_l and game.check_collision(point_l)) or 
            (dir_u and game.check_collision(point_u)) or 
            (dir_d and game.check_collision(point_d)),
            # danger to the right
            (dir_r and game.check_collision(point_d)) or 
            (dir_l and game.check_collision(point_u)) or 
            (dir_u and game.check_collision(point_r)) or
            (dir_d and game.check_collision(point_l)),
            # danger to the left
            (dir_r and game.check_collision(point_u)) or
            (dir_l and game.check_collision(point_d)) or
            (dir_u and game.check_collision(point_l)) or
            (dir_d and game.check_collision(point_r)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80-self.num_games
        final_move = [0, 0, 0]
        if random.randint(0, self.total_games) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move [move] = 1
        return final_move



def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = snake_game_ai()
    epochs = 100  
 
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Game: {agent.num_games}, Score: {score}, Record: {record}")
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)

            if agent.num_games == agent.total_games:
                break


    plot(plot_scores, plot_mean_scores)

def test():
    total_score = 0
    record = 0
    agent = Agent()
    game = snake_game_ai()
    agent.model.load_state_dict(torch.load(".\\model\\model.pth"))
    agent.model.eval()
 
    state = agent.get_state(game)
    game_over = False
    score = 0
    trials = 10

    for _ in range(trials):    
        while True:
            final_move = agent.get_action(state)
            _ , game_over, score = game.play_step(final_move)
            state = agent.get_state(game)
                
            if score > record:
                record = score
            
            total_score += score
            if game_over:
                game.reset()
                print(f"Game Over! Score: {score}, Record: {record}")
                break
            

if __name__ == "__main__":
    choice = input("train/test: ").strip().lower()
    
    if choice == "train":
        train()
    elif choice == "test":
        test()
    else:
        print("Invalid choice. Please enter 'train' or 'test'.")
