# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance, mazeDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        print(legalMoves[chosenIndex])
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        walls = successorGameState.getWalls()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        for ghostState in newGhostStates:
            if ghostState.getPosition() == newPos:
                return -1
        if currentGameState.getScore() < successorGameState.getScore():
            return 100
        foodDist = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        if len(foodDist) == 0:
            return 9999
        return 1 / min(foodDist)
def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        nextIsPacman = lambda agentIdx: agentIdx == gameState.getNumAgents() - 1
        
        def miniMax(currentState, agentIdx, depth):
            if currentState.isWin() or currentState.isLose() or depth == self.depth + 1:
                return (self.evaluationFunction(currentState), 'Stop')
            valueFunc = None
            nextAgentIdx = (agentIdx + 1) % currentState.getNumAgents()
            if nextIsPacman(agentIdx):
                depth += 1
            if agentIdx == 0:
                valueFunc = max
            else:
                valueFunc = min
            legalActions = currentState.getLegalActions(agentIdx)
            successorValues = [miniMax(currentState.generateSuccessor(agentIdx, action), nextAgentIdx, depth) for action in legalActions]
            retValue = valueFunc(successorValues)
            print(retValue)
            action = legalActions[successorValues.index(retValue)]
            return (retValue[0], action)
        return miniMax(gameState, 0, 1)[1]
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        nextIsPacman = lambda agentIdx: agentIdx == gameState.getNumAgents() - 1
        myMax = lambda x, y: x if x[0] > y[0] else y
        myMin = lambda x, y: x if x[0] < y[0] else y
        def maxValue(currentState, agentIdx, nextDepth, alpha, beta):
            maximal = (-9999, None)
            nextAgentIdx = (agentIdx + 1) % currentState.getNumAgents()
            legalActions = currentState.getLegalActions(agentIdx)
            for action in legalActions:
                successorState = currentState.generateSuccessor(agentIdx, action)
                maximal = myMax(maximal, (genericValue(successorState, nextAgentIdx, nextDepth, alpha, beta)[0], action))
                if maximal[0] > beta:
                    return maximal
                alpha = max(alpha, maximal[0])
            return maximal

        def minValue(currentState, agentIdx, nextDepth, alpha, beta):
            minimal = (9999, None)
            nextAgentIdx = (agentIdx + 1) % currentState.getNumAgents()
            legalActions = currentState.getLegalActions(agentIdx)
            for action in legalActions:
                successorState = currentState.generateSuccessor(agentIdx, action)
                minimal = myMin(minimal, (genericValue(successorState, nextAgentIdx, nextDepth, alpha, beta)[0], action))
                if minimal[0] < alpha:
                    return minimal
                beta = min(beta, minimal[0])
            return minimal

        def genericValue(currentState, agentIdx, depth, alpha, beta):
            if currentState.isWin() or currentState.isLose() or depth == self.depth + 1:
                return (self.evaluationFunction(currentState), 'Stop')
            if nextIsPacman(agentIdx):
                depth += 1
            if agentIdx == 0:
                return maxValue(currentState, agentIdx, depth, alpha, beta)
            else:
                return minValue(currentState, agentIdx, depth, alpha, beta)
        return genericValue(gameState, 0, 1, -9999, 9999)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        nextIsPacman = lambda agentIdx: agentIdx == gameState.getNumAgents() - 1
        myMax = lambda x, y: x if x[0] > y[0] else y
        def maxValue(currentState, agentIdx, nextDepth):
            maximal = (-9999, None)
            nextAgentIdx = (agentIdx + 1) % currentState.getNumAgents()
            legalActions = currentState.getLegalActions(agentIdx)
            for action in legalActions:
                successorState = currentState.generateSuccessor(agentIdx, action)
                maximal = myMax(maximal, (genericValue(successorState, nextAgentIdx, nextDepth)[0], action))
            return maximal
        
        def chanceValue(currentState, agentIdx, nextDepth):
            expection = 0
            nextAgentIdx = (agentIdx + 1) % currentState.getNumAgents()
            legalActions = currentState.getLegalActions(agentIdx)
            for action in legalActions:
                successorState = currentState.generateSuccessor(agentIdx, action)
                expection += genericValue(successorState, nextAgentIdx, nextDepth)[0]
            return (expection / len(legalActions), None)

        def genericValue(currentState, agentIdx, depth):
            if currentState.isWin() or currentState.isLose() or depth == self.depth + 1:
                return (self.evaluationFunction(currentState), 'Stop')
            if nextIsPacman(agentIdx):
                depth += 1
            if agentIdx == 0:
                return maxValue(currentState, agentIdx, depth)
            else:
                return chanceValue(currentState, agentIdx, depth)
        value = genericValue(gameState, 0, 1)
        return value[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    ghostStates = currentGameState.getGhostStates()
    foods = currentGameState.getFood().asList()
    currentScore = currentGameState.getScore()
    currentPos = currentGameState.getPacmanPosition()
    walls = currentGameState.getWalls()
    ghostDists = [manhattanDistance(currentPos, ghost.getPosition()) for ghost in ghostStates]
    foodDists = []
    if len(foods) >= 5:
        foodDists = [manhattanDistance(currentPos, foodPos) for foodPos in foods]
    else:
        foodDists = [mazeDistance(currentPos, foodPos, walls) for foodPos in foods]
    ghostScore = 0
    foodScore = 0
    for g in ghostDists:
        if g == 0:
            ghostScore -= 9999999
        else:
            ghostScore += g
    for f in foodDists:
        if f == 0:
            foodScore += 10000
        else:
            foodScore += 1 / f
    if len(foods) == 1:
        foodScore *= 100000
    if len(foods) == 0:
        foodScore += 100000
    return currentScore + 0.1 * ghostScore + foodScore

# Abbreviation
better = betterEvaluationFunction
