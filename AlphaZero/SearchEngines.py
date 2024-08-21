from random import choice
from math import sqrt,log
import numpy as np


class Human:

    def __init__(self,game,args):
        self.game = game
        self.arga = args
    
    def __call__(self,state,player):
        valid_actions = self.game.get_valid_actions(state,player)
        print("# Valid Actions : ", valid_actions)
        try:
            action = eval(input(f"# Enter Action (Player {player}) : "))
        except:
            action = None
        return action


class Random:

    def __init__(self,game,args):
        self.game = game
        self.arga = args
    
    def __call__(self,state,player):
        valid_actions = self.game.get_valid_actions(state,player)
        action = choice(valid_actions)
        print(f"# Action Chosen (Player {player}) : {action}")
        return action


class MiniMax:
    
    def __init__(self,game,args):
        self.game = game
        self.args = args
    
    def __call__(self,state,player):
        return self.get_best_action(state,player)

    def get_best_action(self,state,player):
        game = self.game
        depth = self.args["depth"]
        best_value = -player*float("inf")
        actions_value = {}
        best_actions = []

        valid_actions = game.get_valid_actions(state,player)
        for action in valid_actions:
            new_state = game.get_next_state(state,player,action)
            value = self.search(new_state,-player,action,depth)
            actions_value[action] = value
            if (player*value > player*best_value):
                best_value = value

        for action in actions_value:
            if actions_value[action]==best_value:
                best_actions.append(action)
        best_action = choice(best_actions)
        print(f"# Action Chosen (Player {player}) : {best_action}")
        return best_action
    
    def search(self,state,player,action,depth,alpha=-float("inf"),beta=float("inf")):
        
        game = self.game
        if depth==0 or game.is_terminated(state,-player,action):
            return self.evaluate(state,player,action)
        
        if player == 1:
            max_value = -float("inf")
            valid_actions = game.get_valid_actions(state,player)
            for action in valid_actions:
                next_state = game.get_next_state(state,player,action)
                value = self.search(next_state,-player,action,depth-1,alpha,beta)
                max_value = max(value,max_value)
                alpha = max(value,alpha)
                if beta<=alpha:
                    break
            return max_value
        
        if player == -1:
            min_value = float("inf")
            valid_actions = game.get_valid_actions(state,player)
            for action in valid_actions:
                next_state = game.get_next_state(state,player,action)
                value = self.search(next_state,-player,action,depth-1,alpha,beta)
                min_value = min(value,min_value)
                beta = min(value,beta)
                if beta<=alpha:
                    break
            return min_value
    
    def evaluate(self,state,player,action):
        return self.game.get_value(state,-player,action)


class MCTS:

    def __init__(self,game,args):
        self.game = game
        self.args = args

    def __call__(self,state,player):
        return self.get_best_action(state,player)
    
    class Node:

        def __init__(self,state,player,parent=None,action_taken=None):
            self.state = state
            self.player = player
            self.parent = parent
            self.action_taken = action_taken
            self.children = []
            self.visit_count = 0
            self.value_sum = 0
    
    def get_best_action(self,state,player):
        root = self.search(state,player)
        max_visit_count = 0
        best_action = None
        for child in root.children:
            if child.visit_count>max_visit_count:
                max_visit_count = child.visit_count
                best_action = child.action_taken
        print(f"# Action Chosen (Player {player}) : {best_action}")
        return best_action

    def get_action_probs(self,state,player):
        game = self.game
        root = self.search(state,player)
        action_probs = np.zeros(game.action_size)
        for child in root.children:
            action_probs[child.action_taken]=child.visit_count

        action_probs/=np.sum(action_probs)
        return action_probs
                
    def search(self,state,player):
        game = self.game
        num_searches = self.args["num_searches"]
        state = game.change_perspective(state,player)
        player = player*player
        root = self.Node(state,player)
        for _ in range(num_searches):
            node = root
            while self.is_fully_expanded(node):
                node = self.select(node)
            
            if node.action_taken==None or not game.is_terminated(node.state,-node.player,node.action_taken):
                node = self.expand(node)

            value = self.simulate(node)
            self.backpropagate(node,value)
        return root
    
    def is_fully_expanded(self,node):
        game = self.game
        return len(node.children) == len(game.get_valid_actions(node.state,node.player)) and len(node.children)>0
    
    def select(self,node):
        best_child = None
        best_ucb = -float("inf")
        for child in node.children:
            ucb = self.get_ucb(node,child)
            if ucb>best_ucb:
                best_ucb = ucb
                best_child = child
        return best_child

    def get_ucb(self,node,child):
        exploitation = child.value_sum/child.visit_count
        exploration = sqrt(log(node.visit_count)/child.visit_count)
        ucb_value = exploitation+self.args["c"]*exploration
        return ucb_value
    
    def expand(self,node):
        game = self.game
        valid_actions = game.get_valid_actions(node.state,node.player)
        expandable_actions = [action for action in valid_actions if action not in [child.action_taken for child in node.children]] 
        action = choice(expandable_actions)
        new_child_state = game.get_next_state(node.state,node.player,action)
        new_child_node = self.Node(new_child_state,-node.player,node,action)
        node.children.append(new_child_node)
        return new_child_node
    
    def simulate(self,node):
        game = self.game
        rollout_state = node.state
        rollout_player = node.player
        rollout_action_taken = node.action_taken
        while not game.is_terminated(rollout_state,-rollout_player,rollout_action_taken) :
            valid_actions = game.get_valid_actions(rollout_state,rollout_player)
            action = choice(valid_actions)
            rollout_state = game.get_next_state(rollout_state,rollout_player,action)
            rollout_action_taken = action
            rollout_player = - rollout_player
        return game.get_value(rollout_state,-rollout_player,rollout_action_taken)          
    
    def backpropagate(self,node,value):
        node.visit_count+=1
        node.value_sum += -node.player*value
        if node.parent is not None:
           self.backpropagate(node.parent,value) 


class AdvSearch(MiniMax):

    def __init__(self, game, args):
        super().__init__(game, args)


    def evaluate(self, state, player, action):
        if self.game.is_terminated(state,-player,action):
            return super().evaluate(state,player,action)
        return MCTS(self.game,self.args).search(state,player).value_sum/self.args["num_searches"]    
    
