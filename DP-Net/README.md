# DP-Net

The code is for "Kill two birds with one stone: generalized and robust AI-generated text detection via dynamic perturbations."

# Dataset
The dataset used in the experiments can be obtained from https://github.com/mbzuai-nlp/M4.
The related paper is: [M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection](https://aclanthology.org/2024.eacl-long.83)

# Train
`python RL_encoder.py`

The sequence of random seeds used in RLNE is [18, 30, 52, 88, 100].

# Test
`python test.py`

The reinforcement learning environment setup, DDPG agent, and training process can be found in RL_encoder.py.