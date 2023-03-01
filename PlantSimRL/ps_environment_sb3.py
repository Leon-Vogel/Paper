import itertools
from random import seed
from time import sleep

import gym
import numpy as np
from gym import spaces

from plantsim.plantsim import Plantsim
from problem import Problem


class PlantSimulationProblem(Problem):

    def __init__(self, plantsim: Plantsim, states=None, actions=None, id=None, evaluation=0, goal_state=False):
        """

        :param plantsim: Simulation model must include two tables "Actions" and "States" with both column and row index.
                         column index should be "Index" and "Name" for Actions, where "Name" contains the names of the
                         actions. For state the column index should be the names of the attributes. Also add "id" as
                         integer where the object id is saved, "evaluation" as a float value and "goal_state" as a
                         boolean value.
                         Also include two tables for the data exchange:
                         Table1: "CurrentState" with "id", one column for each state, "evaluation" and "goal_state"
                         Table2: "ActionControl" with "id",one column for each state and "action"

        """
        self.plantsim = plantsim
        if actions is not None:
            self.actions = actions
        else:
            self.actions = self.plantsim.get_object("Actions").get_columns_by_header("Name")
        if states is not None:
            self.states = states
        else:
            self.states = {}
            states = self.plantsim.get_object("States")
            for header in states.header:
                if header != "Index":
                    self.states[header] = states.get_columns_by_header(header)
                    # removing empty cells - entfernt auch 0
                    # self.states[header] = list(filter(None, self.states[header]))
        self.state = None
        self.id = id
        self.evaluation = evaluation
        self.goal_state = goal_state
        self.next_event = True
        self.step = 0

    def copy(self):
        ps_copy = PlantSimulationProblem(self.plantsim, self.state.copy(), self.actions[:], self.id, self.evaluation,
                                         self.goal_state)
        return ps_copy

    def act(self, action):
        self.plantsim.set_value("ActionControl[\"id\",1]", self.id)
        for label, values in self.states.items():
            for value in self.state:
                if value in values:
                    self.plantsim.set_value(f"ActionControl[\"{label}\",1]", value)
        self.plantsim.set_value("ActionControl[\"action\",1]", action)
        # print("Step "+str(self.step)+": "+action+"\n")
        self.plantsim.execute_simtalk("AIControl")
        if not self.plantsim.plantsim.IsSimulationRunning():
            self.plantsim.start_simulation()

        self.next_event = True
        self.step += 1

    def to_state(self):
        return tuple(self.state)

    def is_goal_state(self, state):
        return state.goal_state

    def get_applicable_actions(self, state):
        return self.actions

    def get_current_state(self):
        """
        possible actions list named "actions" must be returned be simulation within the message
        :return:
        """
        while not self.plantsim.get_value("ready"):
            sleep(0.00001)
            print("sleep")
        if self.next_event:
            self.state = []
            # states = self.plantsim.get_next_message()
            states = self.plantsim.get_current_state()
            if states == None:
                print("kein state")
            for key, value in states.items():
                if key == "id":
                    self.id = value
                elif key == "evaluation":
                    self.evaluation = value
                elif key == "goal_state":
                    self.goal_state = value
                else:
                    self.state.append(value)
                    # print(key +":  "+ str(value))
            if states.items() == None:
                print("kein state")
            self.next_event = False
        return self

    def eval(self, state):
        return state.evaluation

    def get_all_actions(self):
        return self.actions

    def get_all_states(self):
        all_states = list(itertools.product(*list(self.states.values())))
        all_states = [tuple(x) for x in all_states]
        return all_states

    def get_reward(self, state):
        reward = -self.eval(state)
        # print("Reward: "+str(reward))
        return reward

    def reset(self):
        self.state = None
        self.id = None
        self.evaluation = 0
        self.goal_state = False
        self.next_event = True


class Environment(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, plantsim: Plantsim, seed_value=1):
        self.done = None
        self.current_state = None
        self.observation = None
        self.new_observation = None
        self.reward = None
        self.info = None

        if seed_value is not None:
            seed(seed_value)
        plantsim.reset_simulation()
        self.problem = PlantSimulationProblem(plantsim)
        plantsim.start_simulation()

        actions = self.problem.actions
        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(low=np.array([liste[0] for liste in self.problem.states.values()]),
                                            high=np.array([liste[1] for liste in self.problem.states.values()]),
                                            dtype=float)

    def step(self, action):
        a = int(action[0])
        a = self.problem.actions[a]
        self.problem.act(a)
        self.current_state = self.problem.get_current_state()
        self.new_observation = np.array(self.current_state.to_state())
        self.reward = -1 * self.problem.get_reward(self.current_state)
        self.done = self.problem.is_goal_state(self.current_state)
        self.info = {}
        '''print(self.new_observation)
        print('-- Return %.1f' % self.reward)
        print('-- done: '+ str(self.done))'''
        return self.new_observation, self.reward, self.done, self.info

    def reset(self):
        self.problem.plantsim.execute_simtalk("reset")
        self.problem.plantsim.reset_simulation()
        self.problem.reset()
        self.problem.plantsim.start_simulation()
        self.problem.plantsim.execute_simtalk("GetCurrentState")
        self.current_state = self.problem.get_current_state()
        self.observation = np.array(self.current_state.to_state())
        print('#####Reset')
        # self.done = False
        return self.observation

    def render(self, mode='human'):
        ...

    def close(self):
        self.problem.plantsim.quit()
