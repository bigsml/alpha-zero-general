import Arena
from MCTS import MCTS
from Coach import Coach
from tictactoe_3d.TicTacToeGame import TicTacToeGame as Game
from tictactoe_3d.keras.NNet import NNetWrapper as NNet
from tictactoe_3d.TicTacToePlayers import HumanTicTacToePlayer, RandomPlayer
import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

NUMBER = 3

g = Game(NUMBER)

# all players
rp = RandomPlayer(g).play
hp = HumanTicTacToePlayer(g, NUMBER).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./pretrained_models/tictactoe_3d/keras/','best-200eps-200sim-10epch.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

player2 = hp

arena = Arena.Arena(n1p, player2, g, display=Game.display)

print(arena.playGames(2, verbose=True))
