from pacman import Directions
from game import Agent, Actions
from pacmanAgents import LeftTurnAgent


class TimidAgent(Agent):
    """
    A simple agent for PacMan that escapes the ghosts when conditionns are met 
    and turns left at every opportunity otherwise
    """

    def __init__(self):
        super().__init__()  # Call parent constructor
        # Add anything else you think you need here

    def inDanger(self, pacman, ghost, dist=3):
        """inDanger(pacman, ghost) - Is the pacman in danger
        For better or worse, our definition of danger is when the pacman and
        the specified ghost are:
           in the same row or column,
           the ghost is not scared,
           and the agents are <= dist units away from one another

        If the pacman is not in danger, we return Directions.STOP
        If the pacman is in danger we return the direction to the ghost.
        """
        #Get position of pacman and ghost
        pacmanPosition = pacman.getPosition()
        ghostPosition = ghost.getPosition()

        #if pacman and ghost are in the same column (x coordinate)
        if(pacmanPosition[0] == ghostPosition[0]): 
            if(abs(pacmanPosition[1] - ghostPosition[1]) <= dist):  #if y distance is less than dist
                if( not ghost.isScared()):  #ghost is not frigtened
                    vecdist = pacmanPosition[1] - ghostPosition[1]  #calculate the vector distance between pacman and ghost
                    if vecdist < 0: 
                        ghostDirection = Directions.NORTH  #negative value implies ghost is above pacman
                    else: 
                        ghostDirection = Directions.SOUTH  #positive value implies ghost is below pacman

                    return ghostDirection
                else:  #ghost is not frigtened, pacman is not in danger
                    return Directions.STOP  
            else:  #y distance is greater than dist, pacman is not in danger
                return Directions.STOP

        #if pacman and ghost are in the same row (y coordinate)
        elif(pacmanPosition[1] == ghostPosition[1]): 
            if(abs(pacmanPosition[0] - ghostPosition[0]) <= dist):  #if x distance is less that dist
                if( not ghost.isScared()):  #ghost is not frigtened
                    vecdist = pacmanPosition[0] - ghostPosition[0]  #calculate vector distance
                    if vecdist < 0: 
                        ghostDirection = Directions.EAST  #negative value implies ghost is to the right of pacman
                    else: 
                        ghostDirection = Directions.WEST  #positive value implies ghost is to the left of pacman
                        
                    return ghostDirection
                else:  #ghost is not frigtened, pacman is not in danger
                    return Directions.STOP 
            else:  #y distance is greater than dist, pacman is not in danger
                return Directions.STOP
        
        #pacman is not in danger
        else: 
            return Directions.STOP
                

    def getAction(self, state):
        """
        state - GameState
        getAction(state) - Make a decsion based on the current game state and 
                           whether the pacman is in danger based on inDanger(pacman, ghost, dist) method
        state - pacman.GameState instance
        returns a valid action direction:  North, East, South, West or
        a Stop action when no legal actions are possible

        Part of the code is referenced from LeftTurnAgent
        
        """

        # List of directions the agent can choose from
        legal = state.getLegalPacmanActions()

        # Get the pacman's state and ghosts' states from the game state and find agent heading
        agentState = state.getPacmanState()
        ghostStates = state.getGhostStates()
        heading = agentState.getDirection()


        if heading == Directions.STOP:
            # Pacman is stopped, assume North (true at beginning of game)
            heading = Directions.NORTH
        
        #Check if pacman is in danger with each of the ghosts
        for i in range(len(ghostStates)): 
            danger = self.inDanger(agentState, ghostStates[i])

            #pacman is not in danger, moves like LeftTurnAgent
            if danger == Directions.STOP: 
                action = LeftTurnAgent.getAction(self,state)
                
            #pacman is in danger
            else:
                if Directions.REVERSE[danger] in legal:  #reverse current direction with respect to direction of ghost from pacman
                    action = Directions.REVERSE[danger]    
                elif Directions.LEFT[danger] in legal:  #turn left with respect to direction of ghost
                    action = Directions.LEFT[danger]
                elif Directions.RIGHT[danger] in legal:  #turn right with respect to direction of ghost 
                    action = Directions.RIGHT[danger]    
                elif heading in legal:  #continue in current direction
                    action = heading
                else:
                    action = Directions.STOP  #Can't move, no move is legal
                return action  #reacts as soon as pacman is in danger 

        return action

