import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from random import choice,shuffle
from math import sqrt,log
from GameModels import TicTacToe,ConnectFour
from SearchEngines import MCTS,MiniMax



class ResNet(nn.Module):
    
    def __init__(self,game,args):
        super().__init__()
        self.game = game
        self.args = args
        nhc = args["num_hidden_channels"]
        nrb = args["num_resBlocks"]

        self.startBlock = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=nhc,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=nhc),
            nn.ReLU()
        )

        self.backBone =  nn.ModuleList(
            [ResBlock(nhc) for _ in range(nrb)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(in_channels=nhc,out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=32*game.row_count*game.col_count,out_features=game.action_size)

        )

        self.vaueHead = nn.Sequential(
            nn.Conv2d(in_channels=nhc,out_channels=3,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=3*game.row_count*game.col_count,out_features=1),
            nn.Tanh()
        )


    def forward(self,isate):
        fstate = self.startBlock(isate)
        for resBlock in self.backBone:
            fstate = resBlock(fstate)
        policy = self.policyHead(fstate)
        value = self.vaueHead(fstate)
        return policy,value


class ResBlock(nn.Module):
    
    def __init__(self,nhc):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=nhc,out_channels=nhc,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=nhc),
            nn.ReLU(),
            nn.Conv2d(in_channels=nhc,out_channels=nhc,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=nhc)
        )


    def forward(self,x):
        fx =  self.layer_stack(x)
        fx+=x
        fx = F.relu(fx)
        return fx   


class RMCTS:

    def __init__(self,game,model,args):
        self.game = game
        self.model = model
        self.args = args

    def __call__(self,state,player):
        return self.get_best_action(state,player)
    
    class Node:

        def __init__(self,state,player,parent=None,action_taken=None,prior=0):
            self.state = state
            self.player = player
            self.parent = parent
            self.action_taken = action_taken
            self.prior = prior
            self.children = []
            self.visit_count = 0
            self.value_sum = 0
    
    def get_best_action(self,state,player):
        game = self.game
        model = self.model

        nstate = game.change_perspective(state,player)
        estate = torch.tensor(game.get_encoded_state(nstate),dtype=torch.float32).unsqueeze(0)
        model.eval()
        policy,value = model(estate)
        value = round(player*value.item(),4)

        root = self.search(state,player)
        best_action = None
        action_probs = np.zeros(game.action_size)
        max_visit_count = 0
        for child in root.children:
            action_probs[child.action_taken]=child.visit_count
            if child.visit_count>max_visit_count:
                max_visit_count = child.visit_count
                best_action = child.action_taken
        action_probs/=np.sum(action_probs)
        """
        best_ucb = 0
        for child in root.children:
            child_ucb = self.get_ucb(root,child)
            if child_ucb>best_ucb:
                best_ucb = child_ucb
                best_action = child.action_taken
        """
        print("# Action Probs :",np.round(action_probs,2))
        print("# Value Pred. :", value)
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

    
    @torch.inference_mode()
    def search(self,state,player):
        game = self.game
        model = self.model
        num_searches = self.args["num_searches"]
        state = game.change_perspective(state,player)
        player = player*player
        root = self.Node(state,player)
        for _ in range(num_searches):
            node = root
            while self.is_fully_expanded(node):
                node = self.select(node)
            
            if node.action_taken==None or not game.is_terminated(node.state,-node.player,node.action_taken):
                
                nstate = game.change_perspective(node.state,node.player)
                estate = torch.tensor(game.get_encoded_state(nstate),dtype=torch.float32).unsqueeze(0)
                policy,value = model(estate)

                policy = torch.softmax(policy, axis=1).squeeze(0).numpy()
                actions_space = game.are_actions_available(node.state,node.player)
                policy *= actions_space

                policy /= np.sum(policy)
                value = node.player*value.item()

                self.expand(node,policy)

            self.backpropagate(node,value)

        return root
    
    def is_fully_expanded(self,node):
        return len(node.children)>0
    
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
        exploitation = child.value_sum/child.visit_count if child.visit_count>0 else 0
        exploration = child.prior*sqrt(node.visit_count/(child.visit_count+1))
        ucb_value = exploitation+self.args["c"]*exploration
        return ucb_value
    
    def expand(self,node,policy):
        game = self.game
        for action, prob in enumerate(policy):
            if prob>0:
                new_child_state = game.get_next_state(node.state,node.player,action)
                new_child_node = self.Node(new_child_state,-node.player,node,action,prob)
                node.children.append(new_child_node)

        
    def backpropagate(self,node,value):
        node.visit_count+=1
        node.value_sum += -node.player*value
        if node.parent is not None:
           self.backpropagate(node.parent,value) 

class RMiniMax:
    
    def __init__(self,game,args):
        self.game = game
        self.args = args
    
    def __call__(self,state,player):
        return self.get_best_action(state,player)

    def get_result(self,state,player):
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
    
        action_probs = np.zeros(game.action_size)
        action_probs[best_action] = 1


        return action_probs,best_value
    
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


class AlphaZero:

    def __init__(self,game,model,engine,optimizer,args):
        self.game = game
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.engine = engine

    def selfPlay(self):
        game = self.game
        engine = self.engine
        
        gMemory = []
        player = 1
        state = game.initial_state
        while True:

            ##game.display(state)
            action_probs = engine.get_action_probs(state,player)
            ##print(">> Action Probs :",action_probs)
            ##print(">> Best Value :",best_value)

            nstate = game.change_perspective(state,player)
            estate = game.get_encoded_state(nstate)
            gMemory.append((estate,action_probs,player))

            #action = np.random.choice(game.get_valid_actions(state,player))
            action = np.random.choice(game.action_size,p=action_probs)
        
            state = game.get_next_state(state,player,action)

            if game.is_terminated(state,player,action):
                
                rMemory = []
                value = game.get_value(state,player,action)

                for hist_state,hist_action_probs,hist_player in gMemory:
                    rMemory.append((
                        hist_state,
                        hist_action_probs,
                        value*hist_player
                    ))
                
                return rMemory
            player = -player


    def train(self,memory):
        args = self.args
        model = self.model
        optimizer = self.optimizer
        bsize = args["batch_size"]
        mlen = len(memory)

        count = 0
        total_loss = 0

        shuffle(memory)
        for bidx in range(0, mlen,bsize):
            fidx = min(mlen-1,bidx+bsize) 
            sample = memory[bidx : fidx]

            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets),np.array(value_targets).reshape(-1,1)
            
            state = torch.tensor(state, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)
            
            policy_outs, value_outs = model(state)
            policy_loss = F.cross_entropy(policy_outs,policy_targets)
            value_loss = F.mse_loss(value_outs,value_targets)
            loss = policy_loss+value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss
            count+=1
        
        return total_loss/count
       

    def learn(self):
        args = self.args
        
        print("-"*94)
        for iteration in range(args["num_iterations"]):

            print("# Iteration :",iteration+1)
            print()

            iMemory = []
            self.model.eval()
            for selfPlay_games in range(args["num_self_plays"]):
                print("  >> Self Play :",selfPlay_games+1)
                iMemory += self.selfPlay()
            print()
            self.model.train()
            for epoch in range(args["num_epochs"]):
                loss = self.train(iMemory)
                print(f"  >> Epoch {epoch+1}:",round(loss.item(),8))

            
            
            print()
            print("-"*94)
            torch.save(self.model.state_dict(),f"{game}_model_save.pt")
            torch.save(self.optimizer.state_dict(),f"{game}_optim_save.pt")



class EvalRMCTS(RMCTS):

    def __init__(self,game,args):
        inargs = {
            "num_resBlocks":4,
            "num_hidden_channels":64,
            }
        model = ResNet(game,inargs)
        try:
            model.load_state_dict(torch.load(f"{game}_model_save.pt"))
        except:
            pass
        model.eval()
        super().__init__(game,model,args)



class Master:

    def __init__(self,game,args):
        self.game = game
        inargs = {
            "num_resBlocks":4,
            "num_hidden_channels":64,
            }
        self.model = ResNet(self.game,inargs)
        #try:
        self.model.load_state_dict(torch.load(f"{self.game}_model_save.pt"))
        #except:
            #pass

    def __call__(self,state,player):
        return self.get_best_action(state,player)
    

    
    def get_best_action(self,state,player):
        game = self.game
        model = self.model
        nstate = game.change_perspective(state,player)
        estate = torch.tensor(game.get_encoded_state(nstate), dtype=torch.float32).unsqueeze(0)
    
        model.eval()
        with torch.inference_mode():
            policy_out,value_out = model(estate)
            policy_pred = torch.softmax(policy_out, axis=1).squeeze(0).numpy()

        action_space = game.are_actions_available(state,player)
        action_probs = policy_pred*action_space
        action_probs /= np.sum(action_probs)
        best_action = np.argmax(action_probs)
        print(f"# Action Chosen (Player {player}) : {best_action}")
        #return policy_pred,value_out.item()
        return best_action 


def Trainer(game,args):
    
    inargs = {
        "num_resBlocks":4,
        "num_hidden_channels":64,
        }

    model = ResNet(game,inargs)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    engine = RMCTS(game,model,args)
    #engine = RMiniMax(game,args)
    
    try:
        model.load_state_dict(torch.load(f"{game}_model_save.pt"))
        optimizer.load_state_dict(torch.load(f"{game}_optim_save.pt"))
    
    except:
        pass
  
    trainee = AlphaZero(game,model,engine,optimizer,args)
    trainee.learn()


if __name__=="__main__":


    args = {
        "depth":float("inf"),
        "c":2,
        "num_searches":1000,

        "num_iterations":5,
        "num_self_plays":100,
        "num_epochs":100,
        "batch_size":64

    }

    game  = TicTacToe()
   
    
    Trainer(game,args)

    state = np.array([
        [1,0,0],
        [0,0,0],
        [0,0,0]
    ])
    player = -1

    """
    engine = Master(game,args)
    
    policy,value = engine(state,player)
    print("# Policy :",np.round(policy,2))
    print("# Value :",value)

    engine = RMiniMax(game,args)
    policy,value = engine.get_result(state,player)
    print("# Policy :",np.round(policy,2))
    print("# Value :",value)

    MiniMax(game,args)(state,player)
    """
   
    

   

    

    

    
    
    

     

    
    
    



   
        
        


    
