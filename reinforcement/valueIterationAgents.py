# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(0, self.iterations):
            incrementVector = util.Counter()
            for s in self.mdp.getStates():
                maxValue = -9999
                actions = self.mdp.getPossibleActions(s)
                if len(actions) == 0:
                    maxValue = 0
                for a in actions:
                    sum = 0
                    for sp_tuple in self.mdp.getTransitionStatesAndProbs(s, a):
                        succ, prob = sp_tuple
                        reward = self.mdp.getReward(s, a, succ)
                        sum += prob * (reward + self.discount * self.values[succ])
                    maxValue = max(sum, maxValue)
                incrementVector[s] = maxValue - self.values[s]
            self.values = self.values + incrementVector

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        sum = 0
        for sp_tuple in self.mdp.getTransitionStatesAndProbs(state, action):
            succ, prob = sp_tuple
            reward = self.mdp.getReward(state, action, succ)
            sum += prob * (reward + self.discount * self.values[succ])
        return sum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        cnt = util.Counter()
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        for action in actions:
            for sp_tuple in self.mdp.getTransitionStatesAndProbs(state, action):
                succ, prob = sp_tuple
                reward = self.mdp.getReward(state, action, succ)
                cnt[action] += prob * (reward + self.discount * self.values[succ])
        return cnt.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(0, self.iterations):
            state = states[i % len(states)]
            self.values[state] = computeMaxQvalue(state, self.mdp, self.discount, self.values)

        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        #compute all predecessors.
        predecessors = {}
        for state in states:
            successors = set()
            for action in self.mdp.getPossibleActions(state):
                for succ in self.mdp.getTransitionStatesAndProbs(state, action)[0]:
                    successors.add(succ)
            for succ in successors:
                if not succ in predecessors:
                    predecessors[succ] = set()
                predecessors[succ].add(state)

        #initialize priority queue.
        priority_queue = util.PriorityQueue()
        for state in states:
            diff = abs(computeMaxQvalue(state, self.mdp, self.discount, self.values) - self.values[state])
            priority_queue.push(state, -diff)

        #iteration
        for i in range(0, self.iterations):
            if priority_queue.isEmpty():
                break
            state = priority_queue.pop()
            self.values[state] = computeMaxQvalue(state, self.mdp, self.discount, self.values)
            for pred in predecessors[state]:
                diff = abs(computeMaxQvalue(pred, self.mdp, self.discount, self.values) - self.values[pred])
                if diff > self.theta:
                    priority_queue.update(pred, -diff)


            

def computeMaxQvalue(state, mdp, discount, values):
    actions = mdp.getPossibleActions(state)
    maxValue = -9999
    if mdp.isTerminal(state):
        return 0
    for action in actions:
        sum = 0
        for sp_tuple in mdp.getTransitionStatesAndProbs(state, action):
            succ, prob = sp_tuple
            reward = mdp.getReward(state, action, succ)
            sum += prob * (reward + discount * values[succ])
        maxValue = max(sum, maxValue)
    return maxValue
