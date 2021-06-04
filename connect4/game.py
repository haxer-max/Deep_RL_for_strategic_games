import numpy as np
from copy import deepcopy

class Game():
    def __init__(self,state=np.zeros((6,7)),player=1):
        self.state = deepcopy(state)
        self.player = player
        #self.top = self.gettop(state)        
        pass
    
    def gettop(self,state):
        top = np.zeros((7,),dtype=np.int32)
        for i in range(7):
            for j in range(6):
                if state[j,i]==0:
                    top[i]=j
                    break
                top[i]=6
        return top
    
    def ValidMoves(self,state,player):
        top=self.gettop(state)
        return top<6    
    
    def Next(self,state,player,action):
        state=deepcopy(state)
        for j in range(6):
            if state[j,action]==0:
                state[j,action]=player
                break
        return state, -player
    
    def play(self,action):
        self.state[self.gettop(self.state)[action],action]=self.player
        self.player*=-1
        pass
    
    def end(self,state,player):
        tok=player
        for i in range(4):
            for j in range(6):
                if state[j][i]==tok and state[j][i+1]==tok and state[j][i+2]==tok and state[j][i+3]==tok:
                    return 1

            for j in range(2):
                if state[j][i]==tok and state[j+1][i+1]==tok and state[j+2][i+2]==tok and state[j+3][i+3]==tok:
                    return 1

            for j in range(3,6):
                if state[j][i]==tok and state[j-1][i+1]==tok and state[j-2][i+2]==tok and state[j-3][i+3]==tok:
                    return 1

        for i in range(7):
            for j in range(2):
                if state[j][i]==tok and state[j+1][i]==tok and state[j+2][i]==tok and state[j+3][i]==tok:
                    return 1

        tok=-player
        for i in range(4):
            for j in range(6):
                if state[j][i]==tok and state[j][i+1]==tok and state[j][i+2]==tok and state[j][i+3]==tok:
                    return -1

            for j in range(2):
                if state[j][i]==tok and state[j+1][i+1]==tok and state[j+2][i+2]==tok and state[j+3][i+3]==tok:
                    return -1

            for j in range(3,6):
                if state[j][i]==tok and state[j-1][i+1]==tok and state[j-2][i+2]==tok and state[j-3][i+3]==tok:
                    return -1   
        for i in range(7):
            for j in range(2):
                if state[j][i]==tok and state[j+1][i]==tok and state[j+2][i]==tok and state[j+3][i]==tok:
                    return -1

        if min(self.gettop(state))==6:
            return 0
        return -2
    
    def stringrep(self,state):
        return state.tobytes()
            
    def PrintBoard(self,state):
        for i in range(5,-1,-1):
            for j in range(7):
                if state[i,j]==1:
                    print("|X", end="")
                elif state[i,j]==-1:
                    print("|O", end="")
                else:
                    print("| ", end="")
            print("|")