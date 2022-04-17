import logging

logging.basicConfig(level=logging.DEBUG)

#import coloredlogs

from Coach import Coach
from tictactoe_3d.TicTacToeGame import TicTacToeGame as Game
from tictactoe_3d.keras.NNet import NNetWrapper as NNet
from tictactoe_3d.TicTacToePlayers import HumanTicTacToePlayer, RandomPlayer
import numpy as np
from utils import *

log = logging.getLogger(__name__)

#coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 100,
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 2000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp','checkpoint_2.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

TIC_NUMBER = 4

def game_factory():
    return Game(TIC_NUMBER)

def main():
    log.info('Loading %s...', Game.__name__)
    g = game_factory()

    log.info('Loading %s...', NNet.__name__)
    nnet = NNet(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args, game_factory=game_factory)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
