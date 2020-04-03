Tasks for Differentiable Harvard Machine

For every folder, main.py is the runner script which will interact with the user.

All the folders have the following structure
  
    1. harvard_machine.py has the code for the proposed Harvard Machine
    2. ntm.py has the code for a Neural Turing Machine with a feed-forward controller
    3. lstm.py has the code for a single layer LSTM network
    4. learned_params/ has trained weights of Harvard Machine, NTM and LSTM
    5. program_memory/ has trained weights of task networks
    6. nets.py has the code for task networks
  
The data is supposed to be loaded from a directory data/ which is not included here due to size limits.

For the addMuliply task, data is generated right before training. Therefore, no additional files are required for training.
