import numpy as np
import colorama
from colorama import Fore

colorama.init()


class TicTacToe:

    def __init__(self):
        self.row_count = 3
        self.col_count = 3
        self.action_size = self.row_count*self.col_count
        self.initial_state = self.get_initial_state()
    
    def __repr__(self):
        return "TicTacToe"
    
    def get_initial_state(self):
        return np.zeros(shape=(self.row_count,self.col_count))
    
    def display(self,state):
        print()
        for i in range(self.row_count):
            for j in range(self.col_count):
                xo = int(state[i,j])
                if xo == 1:
                    xo = Fore.GREEN+" O "
                elif xo == -1:
                    xo = Fore.RED+" X "
                else:
                    xo = "   "
                
                if j<2:
                    print(xo,end=Fore.BLUE+"|")
                else:
                    print(xo)
            if i<2:
                print(Fore.BLUE+"-"*11)
        print(Fore.WHITE)


    def are_actions_available(self,state,player):
        return (state.flatten() == 0).astype(np.uint8)
    
    def get_valid_actions(self,state,player):
        actions_space = self.are_actions_available(state,player) 
        return [i for i in range(self.action_size) if actions_space[i] == 1]
    
    def is_action_valid(self,state,player,action):
        return action in self.get_valid_actions(state,player)

    def get_next_state(self,state,player,action):
        row = action // self.row_count
        col = action % self.col_count
        next_state = state.copy()
        next_state[row,col] = player
        return next_state

    def check_win(self,state,player,action):
        row = action // self.row_count
        col = action % self.col_count
        return (
            np.sum(state[row, :]) == player * self.col_count
            or np.sum(state[:, col]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or state[0][2]+state[1][1]+state[2][0] == player * self.row_count
        )

    def check_draw(self,state,player,action):
        if not self.check_win(state,player,action):
            if np.sum(self.are_actions_available(state,player))==0:
                return True
            else:
                return False
        else:
            return False
    
    def is_terminated(self,state,player,action):
        if self.check_win(state,player,action) or self.check_draw(state,player,action):
            return True
        else:
            return False
        
    def get_value(self,state,player,action):
        if self.check_win(state,player,action):
            return player
        else:
            return 0
        
    def get_opponent(self,player):
        return -player

    def change_perspective(self,state,player):
        return (state.copy())*player
    
    def get_encoded_state(self,state):
        estate = np.stack(
            (state==-1,state==0,state==1)
        ).astype(np.uint8)
        return estate


class ConnectFour:

    def __init__(self):
        self.row_count = 6
        self.col_count = 7
        self.action_size = self.col_count
        self.connect_len = 4
        self.initial_state = self.get_initial_state()

    def get_initial_state(self):
        return np.zeros((self.row_count,self.col_count))
    
    def __repr__(self):
        return "ConnectFour"

    def display(self,state):
        print()
        print("","  0  ","  1  ","  2  ","  3  ","  4  ","  5  ","  6  ")
        for i in range(6):
            print(Fore.BLUE+"|",end="")
            for j in range(7):
                x = int(state[i][j])
                if x == 1:
                    x =  Fore.GREEN+"  O  "
                elif x == -1:
                    x = Fore.RED+"  X  "
                else:
                    x = "     "
                print(x, end=Fore.BLUE+"|")
            print()
            print(Fore.BLUE+"-"*(5*7+8))
        print(Fore.WHITE)
    
    def are_actions_available(self,state,player):
        return (state[0] == 0).astype(np.uint8)
    
    def get_valid_actions(self,state,player):
        actions_space = self.are_actions_available(state,player) 
        return [i for i in range(self.action_size) if actions_space[i] == 1]
    
    def is_action_valid(self,state,player,action):
        return action in self.get_valid_actions(state,player)
    
    def get_next_state(self,state,player,action):
        for i in range(self.row_count):
            if state[i,action]!=0:
                i = i-1
                break
        row = i
        column = action
        next_state = state.copy()
        next_state[row,column] = player
        return next_state

    
    
    def check_win(self,state,player,action):
        
        row = np.min(np.where(state[:,action] !=0))
        column = action

        def count(offset_row,offset_column):
            for i in range(1, self.connect_len):
                r = row + offset_row*i
                c = column+offset_column*i
                if (
                    r<0
                    or r>= self.row_count
                    or c<0
                    or c>=self.col_count
                    or state[r,c] != player
                ):
                    return i-1
            return self.connect_len -1

        return(
            count(1,0) >= self.connect_len -1
            or (count(0,1) + count(0,-1)) >= self.connect_len -1
            or (count(1,1) + count(-1,-1)) >= self.connect_len -1
            or (count(1,-1) +count(-1,1)) >= self.connect_len -1
            )
    
    def check_draw(self, state, player, action):
        if not self.check_win(state,player,action):
            if np.sum(self.are_actions_available(state,player))==0:
                return True
            else:
                return False
        else:
            return False

    def is_terminated(self,state,player,action):
        if self.check_win(state,player,action) or self.check_draw(state,player,action):
            return True
        else:
            return False
        
    def get_value(self,state,player,action):
        if self.check_win(state,player,action):
            return player
        else:
            return 0

    def get_opponent(self, player):
        return -player
    
    def change_perspective(self, state, player):
        return (state.copy())*player
    
    def get_encoded_state(self,state):
        estate = np.stack(
            (state==-1,state==0,state==1)
        ).astype(np.uint8)
        return estate
    
