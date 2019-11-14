import random
import math
import numpy as np


class tictactoe_methods:

    def __init__(self):
        self.win_positions = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]

    def get_initial_board(self):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def get_next_board(self, board, action, turn):
        new_board = board.copy()
        new_board[action] = turn
        return new_board

    def get_possible_actions(self, board):
        actions = []
        for location in range(len(board)):
            if (board[location] == 0):
                actions.append(location)
        return actions

    # used for policy, just returns 9x1 with 0's at impossible moves
    def get_possible_actions_mask(self, board):
        actions = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for location in range(len(board)):
            if (board[location] == 0):
                actions[location] = 1
        return actions

    def is_board_full(self,board):
        for piece in board:
            if(piece == 0):
                return False
        return True

    def get_winner(self, board):
        if(self.is_board_full(board)):
            return 0

        for board_state in self.win_positions:
            if (board[board_state[0]] != 0 and board[board_state[0]] == board[board_state[1]] ):
                if board[board_state[1]] == board[board_state[2]]:
                    return board[board_state[0]]
        return -1
    
    def pretty_print(self,board):
        print("%s|%s|%s"%(board[0],board[1],board[2]))
        print("---------")
        print("%s|%s|%s"%(board[3],board[4],board[5]))
        print("---------")
        print("%s|%s|%s"%(board[6],board[7],board[8]))
