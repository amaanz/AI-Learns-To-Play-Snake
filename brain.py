# First lets try and give our snake some vision, we will give it the ability to see the fruit, the walls and its own body.
# Let's say the snake can look in 8 directions, up, down, left, right, up-left, up-right, down-left, down-right.

# We set the vision range as a variable
vision_range = 10

# We will use the following inputs for the snake:
# 1. Distance to the fruit in each of the 8 directions
# 2. Distance to the wall in each of the 8 directions
# 3. Distance to the body in each of the 8 directions
# So we will have 3 * 8 = 24 inputs

# We will use the following outputs for the snake:
# 1. Move up
# 2. Move down
# 3. Move left
# 4. Move right
# So we will have 4 outputs

# We will initialise the following hidden layers for the snake:
# 1. 12 neurons
# 2. 12 neurons

# We will use the ReLU activation functions

# We will use the Softmax activation function for the output layer

import numpy as np
import nn # Import the neural network from nn.py

# Define the neural network
class Brain:
    def __init__(self):
        self.lay1 = nn.Layer_Dense(24, 12)
        self.act1 = nn.Activation_ReLU()
        self.lay2 = nn.Layer_Dense(12, 48)
        self.act2 = nn.Activation_ReLU()
        self.lay3 = nn.Layer_Dense(48, 4)
        self.act3 = nn.Activation_Softmax()
    def forward(self, inputs):
        self.lay1.forward(inputs)
        self.act1.forward(self.lay1.output)
        self.lay2.forward(self.act1.output)
        self.act2.forward(self.lay2.output)
        self.lay3.forward(self.act2.output)
        self.act3.forward(self.lay3.output)
        self.output = self.act3.output
    def get_inputs(self, state):
        #   the directions are as follows, respective to the snake's head: u d l r ul ur dl dr
        out = np.array([[0,0,0,0,0,0,0,0, # distance to fruit
                        0,0,0,0,0,0,0,0, # distance to wall
                        0,0,0,0,0,0,0,0]], dtype=float) # distance to body
        
        head = state.head # assuming head is a tuple (x, y)
        fruit = state.fruit_position # assuming fruit is a tuple (x, y)
        
        diff_x = head[0] - fruit[0]
        diff_y = head[1] - fruit[1]

        if diff_x > 0:
            out[0][3] = diff_x
        else:
            out[0][2] = -diff_x
        if diff_y > 0:
            out[0][1] = diff_y
        else:
            out[0][0] = -diff_y
        # diagonal directions
        if diff_x > 0 and diff_y > 0:
            out[0][7] = diff_x
        elif diff_x > 0 and diff_y < 0:
            out[0][6] = diff_x
        elif diff_x < 0 and diff_y > 0:
            out[0][5] = -diff_x
        elif diff_x < 0 and diff_y < 0:
            out[0][4] = -diff_x
            
        # distance to wall
        out[0][8] = head[1]
        out[0][9] = state.window_y - head[1]
        out[0][10] = head[0]
        out[0][11] = state.window_x - head[0]
        # diagonal directions
        out[0][12] = min(head[1], head[0])
        out[0][13] = min(state.window_y - head[1], head[0])
        out[0][14] = min(head[1], state.window_x - head[0])
        out[0][15] = min(state.window_y - head[1], state.window_x - head[0])
        
        # distance to body
        for i in range(1, len(state.snake_body)):
            diff_x = head[0] - state.snake_body[i][0]
            diff_y = head[1] - state.snake_body[i][1]
            if diff_x > 0:
                out[0][19] = diff_x
            else:
                out[0][18] = -diff_x
            if diff_y > 0:
                out[0][17] = diff_y
            else:
                out[0][16] = -diff_y
            # diagonal directions
            if diff_x > 0 and diff_y > 0:
                out[0][23] = diff_x
            elif diff_x > 0 and diff_y < 0:
                out[0][22] = diff_x
            elif diff_x < 0 and diff_y > 0:
                out[0][21] = -diff_x
            elif diff_x < 0 and diff_y < 0:
                out[0][20] = -diff_x
        return out
    def get_outputs(self):
        return self.output
    def mutate(self):
        self.lay1.weights += 0.1 * np.random.randn(24, 12)
        self.lay2.weights += 0.1 * np.random.randn(12, 48)
        self.lay3.weights += 0.1 * np.random.randn(48, 4)
        self.lay1.biases += 0.1 * np.random.randn(1, 12)
        self.lay2.biases += 0.1 * np.random.randn(1, 48)
        self.lay3.biases += 0.1 * np.random.randn(1, 4)
    
        
        
        
        




