#!/usr/bin/python
import os

#******************************
#******** Enviroment **********
#******************************

ENV_NAME = 'PriorityLane_MARl'


PATH_SAVE_MODEL = "model/{}/".format(ENV_NAME)
PATH_LOAD_FOLDER = None

BUFFER_CAPACITY = 100000
BATCH_SIZE = 64
MIN_SIZE_BUFFER = 64

CRITIC_HIDDEN_0 = 64
CRITIC_HIDDEN_1 = 64
ACTOR_HIDDEN_0 = 64 
ACTOR_HIDDEN_1 = 64

ACTOR_LR = 0.0001
CRITIC_LR = 0.0001
GAMMA = 0.95
TAU = 0.01

MAX_GAMES = 300
TRAINING_STEP = MAX_GAMES/20
MAX_STEPS = 5
EVALUATION_FREQUENCY = MAX_GAMES/10
SAVE_FREQUENCY = MAX_GAMES/10