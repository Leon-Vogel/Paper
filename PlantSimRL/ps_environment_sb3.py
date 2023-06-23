import itertools
import random
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
            sleep(0.0000001)
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
        self.plan = {
            'Typ1': [93, 79, 90, 93, 91, 70, 95, 89, 78, 98,
                     100, 100, 90, 80, 90, 75, 84, 97, 80,90, 83],

            'Typ2': [86, 73, 69, 89, 85, 74, 71, 61, 86, 67,
                     60, 80, 70, 60, 75, 80, 76, 73, 80,85, 77],

            'Typ3': [53, 75, 59, 56, 65, 73, 66, 76, 60, 61,
                     50, 70, 60, 80, 60, 75, 65, 60, 75,55, 65],

            'Typ4': [35, 40, 47, 33, 33, 43, 35, 43, 40, 36,
                     50, 30, 45, 45, 45, 40, 45, 40, 35,35, 40],

            'Typ5': [33, 33, 35, 29, 26, 40, 33, 31, 36, 38,
                     40, 20, 35, 45, 30, 30, 40, 30, 30,35, 45],

            'Total': [300, 300, 300, 300, 300, 300, 300, 300, 300, 300,
                      300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
        }
        self.eval_plan = {
            'Typ1': [95, 84, 95, 90, 100],
            'Typ2': [83, 78, 73, 69, 65],
            'Typ3': [50, 70, 59, 56, 65],
            'Typ4': [37, 35, 42, 46, 35],
            'Typ5': [35, 33, 31, 39, 35],
            'Total': [300, 300, 300, 300, 300]
        }

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
        self.possible_actions = [''] #ToDo Namen aller gültigen Aktionen, aus Plantsim auslesen

    def valid_action_mask(self): #ToDo valid action ist eine liste mit boolean Werten ob die Aktion an entsprechender Stelle gültig ist. invalid_actions, eine Liste mit den Namen der ungültigen Aktionen muss aus Plantsim gelesen werden
        valid_actions = [action not in self.invalid_actions for action in self.possible_actions]
        return np.array(valid_actions)

    def step(self, action):
        # a = int(action[0]) # für die eigene Lern Methode
        a = action  # für die sb3 model.learn
        a = self.problem.actions[a]
        self.problem.act(a)
        self.current_state = self.problem.get_current_state()
        self.new_observation = np.array(self.current_state.to_state())
        self.reward = -1 * self.problem.get_reward(self.current_state)
        self.done = self.problem.is_goal_state(self.current_state)
        self.info = {}
        if self.done:
            # print('Done Erfolg')
            self.info['Typ1'] = self.problem.plantsim.get_value("Bewertung[\"Typ1\",1]")  # Tabelle für Metrik
            self.info['Typ2'] = self.problem.plantsim.get_value("Bewertung[\"Typ2\",1]")
            self.info['Typ3'] = self.problem.plantsim.get_value("Bewertung[\"Typ3\",1]")
            self.info['Typ4'] = self.problem.plantsim.get_value("Bewertung[\"Typ4\",1]")
            self.info['Typ5'] = self.problem.plantsim.get_value("Bewertung[\"Typ5\",1]")
            self.info['Warteschlangen'] = self.problem.plantsim.get_value("Bewertung[\"Warteschlangen\",1]")
            self.info['Auslastung'] = self.problem.plantsim.get_value("Bewertung[\"Auslastung\",1]")
            self.info['Av_Warteschlangen'] = self.problem.plantsim.get_value("Bewertung[\"Av_Warteschlangen\",1]")
            self.info['Av_Auslastung'] = self.problem.plantsim.get_value("Bewertung[\"Av_Auslastung\",1]")
            self.info['Soll'] = self.problem.plantsim.get_value("Bewertung[\"Soll\",1]")
        '''print(self.new_observation)
        print('-- Return %.1f' % self.reward)
        print('-- done: '+ str(self.done))'''
        return self.new_observation, self.reward, self.done, self.info

    def reset(self, eval_mode=None, eval_step=None):
        '''if self.done:
            print('Done Erfolg')
            print('Typ1: ' + str(self.problem.plantsim.get_value("Bewertung[\"Typ1\",1]")))  # Tabelle für Metrik
            print('Typ2: ' + str(self.problem.plantsim.get_value("Bewertung[\"Typ2\",1]")))
            print('Typ3: ' + str(self.problem.plantsim.get_value("Bewertung[\"Typ3\",1]")))
            print('Typ4: ' + str(self.problem.plantsim.get_value("Bewertung[\"Typ4\",1]")))
            print('Typ5: ' + str(self.problem.plantsim.get_value("Bewertung[\"Typ5\",1]")))
            print('Evaluation Warteschlangen: ' + str(self.problem.plantsim.get_value("Bewertung[\"Warteschlangen\",1]")))
            print('Evaluation Auslastung: ' + str(self.problem.plantsim.get_value("Bewertung[\"Auslastung\",1]")))'''
        self.problem.plantsim.execute_simtalk("reset")
        self.problem.plantsim.reset_simulation()
        self.problem.reset()
        # Plan schreiben
        if eval_mode:
            self.problem.plantsim.set_value("Typ_Soll[\"Typ1\",1]", self.eval_plan['Typ1'][eval_step])
            self.problem.plantsim.set_value("Typ_Soll[\"Typ2\",1]", self.eval_plan['Typ2'][eval_step])
            self.problem.plantsim.set_value("Typ_Soll[\"Typ3\",1]", self.eval_plan['Typ3'][eval_step])
            self.problem.plantsim.set_value("Typ_Soll[\"Typ4\",1]", self.eval_plan['Typ4'][eval_step])
            self.problem.plantsim.set_value("Typ_Soll[\"Typ5\",1]", self.eval_plan['Typ5'][eval_step])
            self.problem.plantsim.set_value("Typ_Soll[\"Total\",1]", self.eval_plan['Total'][eval_step])
        else:
            i = random.randint(0, 20)
            self.problem.plantsim.set_value("Typ_Soll[\"Typ1\",1]", self.plan['Typ1'][i])
            self.problem.plantsim.set_value("Typ_Soll[\"Typ2\",1]", self.plan['Typ2'][i])
            self.problem.plantsim.set_value("Typ_Soll[\"Typ3\",1]", self.plan['Typ3'][i])
            self.problem.plantsim.set_value("Typ_Soll[\"Typ4\",1]", self.plan['Typ4'][i])
            self.problem.plantsim.set_value("Typ_Soll[\"Typ5\",1]", self.plan['Typ5'][i])
            self.problem.plantsim.set_value("Typ_Soll[\"Total\",1]", self.plan['Total'][i])
        self.problem.plantsim.start_simulation()
        self.problem.plantsim.execute_simtalk("GetCurrentState")
        self.current_state = self.problem.get_current_state()
        self.observation = np.array(self.current_state.to_state())
        print('#####Reset')
        self.done = False
        return self.observation

    def render(self, mode='human'):
        ...

    def close(self):
        self.problem.plantsim.quit()
