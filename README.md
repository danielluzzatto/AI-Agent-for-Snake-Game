# AI Agent for Snake Game
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
Deep Q Learning Algorithm to beat the Snake Game

![Snake Game AI Screenshot](assets/Screenshot2.png)

This project implements a Snake Game with an AI agent that can be trained and tested to play the game effectively. The AI uses reinforcement learning to learn how to maximize its score by navigating the environment and collecting food without colliding with itself or the walls.
# Features:
- Train Mode: Train the AI to play the Snake Game.
- Test Mode: Test the AI's performance after training.
- Interactive Gameplay: Watch the AI play the game and observe its progress.

# Getting Started
Prerequisites
Ensure you have Python installed on your system. You can install the required libraries using:

 ```bash
   pip install -r requirements.txt
   ```
# Installation
Clone this repository:
```bash
git clone https://github.com/danielluzzatto/AI-Agent-for-Snake-Game
cd snake-game-ai
```
# Usage

```bash
   python agent.py
   ```

Input train to start training. This will initialize the training process, saving the best model as model.pth in the models/ folder. For testing, type 'test'.


# Results
- The AI typically reaches scores of 60+ after training for about 80 episodes.
- Visualization: During testing, you can visualize how the AI moves and improves over time.

![Snake Game AI Screenshot](assets/plot.png)


# Troubleshooting
If you encounter differences in training and testing performance:

- Verify that the environment setup is consistent in both modes.
- Experiment with hyperparameters such as learning rate, batch size, or exploration decay.

# Contributing
Contributions are welcome! If you'd like to enhance the Snake Game AI, follow these steps:
- Fork the repository.
- Create a new branch for your feature/bugfix.
- Submit a pull request explaining your changes.


# License

This project is licensed under the MIT License. See the `LICENSE` file for details.
