# We want to simulate a mcts tree
# To do this we have to do many simulation steps
# and in each simulation step we:

# 1: Traverse the tree down to a leaf
#   We traverse via this equation
"""
a_t = argmax(Q(s,a) + u(s,a))

Q(s,a) = mcts tree's value of the node reached when moving from state s with action a

u(s,a) = c_puct * P(s,a) * sqrt(sum(visits_to_all_nodes))/ (1+visits_to_state_reached_from_action_a)
c_puct is our exploration rate
"""


# IMPORTANT, We have a small flaw that moving to the same state from
# different directions will copy the tree

import random
import math
import numpy as np

class tictactoe_methods:

    def __init__():
        self.win_positions = [
                [0,1,2],
                [3,4,5], 
                [6,7,8],
                [0,3,6],
                [1,4,7],
                [2,5,8],
                [0,4,8],
                [2,4,6],
        ]

    def starting_board(self):
        return [0,0,0,0,0,0,0,0,0]

    def get_next_board(self, board, action, turn):
        new_board = board.copy()
        new_board[action] = turn
        return new_board

    def get_possible_actions(self, board):
        actions = []
        for location in range(len(board)):
            if(board[location] == 0):
                actions.append(location)
        return actions

    # used for policy, just returns 9x1 with 0's at impossible moves
    def get_possible_actions_mask(self,board):
        actions = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for location in range(len(board)):
            if(board[location] == 0):
                actions[location] = 1
        return actions

    def get_winner(self,board):
        for board_state in self.win_positions:
            if(board[board_state[0] != 0 and board[board_state[0]] == board[board_state[1]] ):
                if(board[board_state[1]] == board[board_state[2]]):
                    return board[board_state[0]]
        return 0
            





class mcts_node:
    def __init__(self,board):
        self.N = 0 #visits
        self.value = -1
        self.win_percentage = -1 #Q
        self.children = None # will get initialized when node is reached
        self.board = board

class mcts_tree:
    # functions to traverse mcts_nodes
    def __init__(self):
        self.tictactoe_methods = tictactoe_methods()
        start_node = self.tictactoe_methods.starting_board()
        self.head = mcts_node(start_node)

    def recursive_simulate(self,node,turn):
        print("visited", node.N)

        # The node has not been expanded
        if(node.children == None):
            print("deepest")
            node.children = self.expand_children(node,turn)

            node.value = self.compute_win_percentage(node)
            node.win_percentage = node.value

            node.N += 1

            return

        if(node.children == []):
            # this is the end of the game
            # discover who won
            winner = self.tictactoe_methods(node.board)
            return winner

        # This node has been expanded we want to keep going further down
        # To do this we have to compute this equation
        """
        a_t = argmax(Q(s,a) + u(s,a))

        Q(s,a) = mcts tree's value of the node reached when moving from state s with action a

        u(s,a) = c_puct * P(s,a) * sqrt(sum(visits_to_all_nodes))/ (1+visits_to_state_reached_from_action_a)
        c_puct is our exploration rate
        """
        print("keeping going down")

        board = node.board

        win_percentages = np.array([child_node.win_percentage for child_node in node.children])

        # Computing U
        c_puct = 1.0
        policy_actions = self.compute_policy(board)
        total_number_of_visits = sum([child_node.N for child_node in node.children])
        visit_term = np.array([math.sqrt(total_number_of_visits)/(1+child_node.N) for child_node in node.children])
        # print("posible actions", self.tictactoe_methods.get_possible_actions_mask(board))
        # print("policy:", policy_actions)
        # print('visit terms', visit_term)
        "expected reward for going into each state"
        Q_term = [child_node.

        U_term = c_puct*policy_actions*visit_term
        # print(U_term)

        # move down the tree following this
        tree_action = np.argmax(win_percentages+U_term)
        # print("ACTION:",tree_action, "On board", board)

        turn = 1 if turn == 2 else 2
        node_to_expand = node.children[tree_action]
        self.recursive_simulate(node_to_expand,turn)


        #update this node because we have traversed through it
        node.N += 1
        

    def compute_policy(self,board):
        policy = np.array([1,1,1,1,1,1,1,1,1])

        # remove the non possible actions
        possible_actions = self.tictactoe_methods.get_possible_actions_mask(board)
        possible_policy = []
        for index in range(len(possible_actions)):
            if possible_actions[index] != 0:
                possible_policy.append(policy[index])
        policy = np.array(possible_policy)
        # renormalize the policy
        policy = policy / np.mean(policy)

        return policy



    def expand_children(self,node,turn):
        # look at board state and expand all the children
        current_board = node.board.copy()
        actions = self.tictactoe_methods.get_possible_actions(current_board)

        children = []
        for action in actions:
            new_board = self.tictactoe_methods.get_next_board(current_board,action,turn)
            children.append(mcts_node(new_board))
        return children


    def compute_win_percentage(self,node):
        # look at board state and determine with what
        # percentage we will win
        return random.random()

    def start_simulation(self):
        self.recursive_simulate(self.head, 1)

if __name__ == "__main__":
    tree = mcts_tree()
    for i in range(100000):
        print("simulation #"+str(i))
        tree.start_simulation()
