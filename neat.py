import numpy as np
import nn
import brain
import random
import pygame # type: ignore
import time

black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

pygame.init()

score_to_achieve = 10

windox_width = 500
window_height = 500

initial_snakes = 100
speed = 20

# Define a NEAT algorithm, with the brain as genes and the fitness as the score

# we will define the fitness function as follows:
# +1 every time snake gets closer to the fruit
# -1 every time snake gets further from the fruit
# +100 if snake eats the fruit
# +0.1 for every step the snake takes

# use pygame to create a game, where the snake is controlled by the brain, and the fitness is calculated as above
# the game will be displayed on the screen, initally all the snakes of genreation 1 will be displayed on the screen and the best snake will be displayed in a different color
# continue the game until all the snakes die, then calculate the fitness of each snake and display the next generation of snakes
# we'll keep track of the generation number and the best fitness of each generation
# and keep increasing the generation number until the game score reaches score_to_achieve

pygame.display.set_caption('Heet Snakes')
game_window = pygame.display.set_mode((windox_width, window_height))

fps = pygame.time.Clock()

max_score = 0

class State:
    
    def __init__(self, generation):
        self.snake_position = [100, 50]
        self.snake_body = [ [100, 50],
                        [90, 50],
                        [80, 50],
                        [70, 50]
                    ]
        self.head = self.snake_body[0]
        self.fruit_position = [random.randrange(1, (windox_width//10)) * 10,
                        random.randrange(1, (window_height//10)) * 10]
        self.fruit_spawn = True
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.snake_speed = speed
        self.brain = brain.Brain()
        self.generation = generation
        self.window_x = windox_width
        self.window_y = window_height
        
        
    def fitness(self):
        return self.score + 0.1 * self.steps
    
    def snake_movement(self):
        if self.direction == 'UP':
            self.snake_position[1] -= 10
        if self.direction == 'DOWN':
            self.snake_position[1] += 10
        if self.direction == 'LEFT':
            self.snake_position[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_position[0] += 10
        self.head = self.snake_body[0]
        self.forward()
        
    def forward(self):
        inputs = self.brain.get_inputs(state=self)
        self.brain.forward(inputs)
        output = self.brain.output[0]
        if output[0] > output[1] and self.direction != 'DOWN':
            self.change_to = 'UP'
        if output[1] > output[0] and self.direction != 'UP':
            self.change_to = 'DOWN'
        if output[2] > output[3] and self.direction != 'RIGHT':
            self.change_to = 'LEFT'
        if output[3] > output[2] and self.direction != 'LEFT':
            self.change_to = 'RIGHT'
        self.direction = self.change_to
    
        
snakes = []
for i in range(initial_snakes):
    # create a state object
    state = State(0)
    snakes.append(state)
    

def show_highest_score(color, font, size, snakes):
    highest_score = 0
    for snake in snakes:
        if snake.score > highest_score:
            highest_score = snake.score
    
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(highest_score), True, color)
    score_rect = score_surface.get_rect()
    game_window.blit(score_surface, score_rect)

def get_next_gen():
    # sort the snakes by fitness
    snakes.sort(key=lambda x: x.fitness(), reverse=True)
    # get the top 20% of the snakes
    top_snakes = snakes[:int(0.2 * len(snakes))]
    # get the top 20% of the snakes and create a new generation
    new_snakes = []
    for i in range(initial_snakes):
        new_snake = State(snakes[0].generation + 1)
        new_snake.brain = top_snakes[i % len(top_snakes)].brain
        new_snakes.append(new_snake)
    # add mutation to the new generation, by changing weights of brain, in the other 80% of the old snakes
    for i in range(int(0.2*len(snakes)),int(len(snakes))):
        snakes[i].brain.mutate()
        new_snakes.append(snakes[i])
    # exchange random weights of the brains of 5% of the new generation
    for i in range(int(0.95 * len(new_snakes)), int(len(new_snakes))):
        new_snakes[i].brain = top_snakes[i % len(top_snakes)].brain
    return new_snakes



while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    
    game_window.fill(black)
    
    for state in snakes:
        if not state.game_over:
            state.steps += 1
            state.snake_body.insert(0, list(state.snake_position))
            if state.snake_position[0] == state.fruit_position[0] and state.snake_position[1] == state.fruit_position[1]:
                state.score += 100
                state.fruit_spawn = False
            else:
                state.snake_body.pop()
                
            if not state.fruit_spawn:
                state.fruit_position = [random.randrange(1, (windox_width//10)) * 10,
                        random.randrange(1, (window_height//10)) * 10]
                state.fruit_spawn = True
                
            state.snake_movement()
            
            if state.snake_position[0] < 0 or state.snake_position[0] > windox_width-10:
                state.game_over = True
            if state.snake_position[1] < 0 or state.snake_position[1] > window_height-10:
                state.game_over = True
                
            for block in state.snake_body[1:]:
                if state.snake_position[0] == block[0] and state.snake_position[1] == block[1]:
                    state.game_over = True
        else:
            if state.score > max_score:
                max_score = state.score
            state.fitness()
            
    if max_score > score_to_achieve:
        break
    
    snakes = get_next_gen()
    
    for state in snakes:
        if not state.game_over:
            pygame.draw.rect(game_window, green, pygame.Rect(state.fruit_position[0], state.fruit_position[1], 10, 10))
            for block in state.snake_body:
                pygame.draw.rect(game_window, white, pygame.Rect(block[0], block[1], 10, 10))
        else:
            for block in state.snake_body:
                pygame.draw.rect(game_window, red, pygame.Rect(block[0], block[1], 10, 10))
    
    show_highest_score(white, 'times new roman', 20, snakes)
    
    pygame.display.flip()
    fps.tick(speed)
    print(max_score)
pygame.quit()


