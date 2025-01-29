import pygame as pg
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pg.init()
font = pg.font.SysFont("arial", 25)


class Directions(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40


class snake_game_ai:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pg.display.set_mode((self.w, self.h))
        self.clock = pg.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Directions.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0

    def place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()

    def play_step(self,action):
        self.frame_iteration += 1
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

        self._move(action)
        self.snake.insert(0, self.head)

        game_over = False
        reward = 0
        if self.check_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.place_food()
        else:
            self.snake.pop()
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def check_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w- BLOCK_SIZE or pt.x < 0 or pt.y > self.h-BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pg.draw.rect(
                self.display, BLUE1, pg.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
        pg.draw.rect(
            self.display, RED, pg.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        )
        text = font.render("Score:" + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pg.display.flip()

    def _move(self, action):
        clock_wise = [Directions.RIGHT, Directions.DOWN, Directions.LEFT, Directions.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_ind = (idx + 1) % 4
            new_dir = clock_wise[next_ind]
        else:
            next_ind = (idx - 1) % 4
            new_dir = clock_wise[next_ind]

        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Directions.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Directions.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Directions.UP:
            y -= BLOCK_SIZE
        elif self.direction == Directions.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)
