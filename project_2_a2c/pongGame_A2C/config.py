#!/usr/bin/python
import os

# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'PongDeterministic-v4'

INPUT_SHAPE = (84, 84, 4)
BATCH_SIZE = 64
N_UPDATES = 2000
HIDDEN = 512
LR = 3e-4
GAMMA = 0.99
VALUE_C = 0.5
ENTROPY_C = 0.01
CLIP_RATIO = 0.2
STD_ADV = True
AGENT = "PPO"

PATH_SAVE_MODEL = f"../model/{AGENT.lower()}"

# PATH_LOAD_MODEL = "./save_agent_202506021305/model.tf"
PATH_LOAD_MODEL = None

CONFIG_WANDB = dict (
  input_shape = INPUT_SHAPE,
  batch_size = BATCH_SIZE,
  n_updates = N_UPDATES,
  hidden = HIDDEN,
  learning_rate = LR,
  # learning_rate = LR,
  gamma = GAMMA,
  value_c = VALUE_C,
  entropy_c = ENTROPY_C,
  clip_ratio = CLIP_RATIO,
  std_adv = STD_ADV,
  agent = AGENT,
  operating_system = os.name
)
