from GameModels import TicTacToe, ConnectFour
from SearchEngines import MiniMax,MCTS,AdvSearch,Human, Random
from AdvEngines import Master,EvalRMCTS

sqrt_2 = 1.4142135623731

def playGame(game,engine1,engine2,args1,args2):

    game = GameClass()
    engine1 = Engine1(game,args1)
    engine2 = Engine2(game,args2)
    move_count = 0

    state = game.initial_state

    player = 1

    while True:
        game.display(state)

        if player==1:
            action = engine1(state,player)
        else:
            action = engine2(state,player)     

        if not game.is_action_valid(state,player,action):
            print()
            print(">> $ Invalid Action, Enter Again $")
            print()
            continue

        state = game.get_next_state(state,player,action)

        if game.is_terminated(state,player,action):
            game.display(state)
            if game.check_draw(state,player,action):
                print(">> ! The Game has been Drawn !")
            else:
                print(f">> ! Player {player} has won the game !" )
            move_count+=1
            print(f">> Total Moves : {move_count}")
            print()
            return game.get_value(state,player,action), move_count

        player = game.get_opponent(player)
        move_count+=1


def GameLoop(GameClass,Engine1,Engine2,args1,args2,duration):
    
    game_count = 0
    win_count1 = 0
    win_count2 = 0
    draw_count = 0
    total_moves = 0

    while game_count<duration:
        print("-"*94)
        print(f"!!! GAME {game_count+1} STARTED !!!")
        score, move_count = playGame(GameClass,Engine1,Engine2,args1,args2)
        print(f"!!! GAME {game_count+1} FINISHED !!!")
        print("-"*94)
        game_count+=1
        if score==1:
            win_count1+=1
        elif score==-1:
            win_count2+=1
        else:
            draw_count+=1
        total_moves+=move_count
    print()
    print("# Total Games :",game_count)
    print("# Player 1 Wins :",win_count1)
    print("# Total Draws :",draw_count)
    print("# Player 2 Wins :",win_count2)
    print("# Average Game Length :",total_moves/game_count)
    print()
    print("-"*94)



GameClass = TicTacToe
Engine1 = MiniMax
Engine2 = Master

args1 = {

    "depth":float("inf"),
    "num_searches":1000,
    "c":2

    }


args2 = {
    "depth":float("inf"),
    "num_searches":1000,
    "c":2
    }





duration = 10

GameLoop(GameClass,Engine1,Engine2,args1,args2,duration)

