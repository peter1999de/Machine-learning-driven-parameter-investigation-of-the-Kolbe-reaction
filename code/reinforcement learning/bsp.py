import math
import pickle
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import pandas as pd

columns = ['best total yield', 'best CE', 'best S', 'pH Start', 'j [mA/cm2]', 'flow rate [l/min]', 't [min]']
df_summary = pd.DataFrame(columns=columns)

class Parameteroptimierung(Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, filename, max_variable, max_episodes, maxsize):

        self.variable = np.array(['pH Start', 'j [mA/cm2]', 'flow rate [l/min]', 't [min]'])
        self.max_variable = max_variable
        self.parameter = np.random.rand(len(max_variable)) # start with random parameters
        self.parameter = np.clip(self.parameter, 0.0, 1.0)
        #print(self.parameter)
        
        self.action_space = Discrete(17) # 2x4 quick in/decrease, 2x4 slow in/decrease, 1 pass

        high = np.array(max_variable)
        low = np.zeros(len(max_variable))
        self.observation_space = Box(low, high, dtype=np.float64)  # observation = adjusted parameters

        self.knn = pickle.load(open(filename, 'rb'))

        # Es wird die beste Ausbeute festgelegt, am Anfang natürlich 0
        self.bestCE = 0
        self.bestyield = 0
        self.bestS = 0
        self.maxsize = maxsize
        self.max_episodes = max_episodes
        self.episode = max_episodes
        # Ein Zähler der hochzählt wenn kein besserer Wert gefunden wurde
        self.yieldcount = 0
        
        # Die Besten Resultate
        self.bestOverallCE = 0
        self.bestOverallYield = 0
        self.bestOverallS = 0
        self.bestParameters = np.zeros((4,))
        # Ein Zähler der verhindert dass die besten Resultate das Terminal sprengen
        self.debugCounter = 0
    

    # Methode zur Berechnung der neuen observation
    def actionCalculation(self, action, observation):
        if action == 8:
            pass
        elif action <= 3:
            observation[action] = np.clip(observation[action] + 0.1, 0.0, 1.0)
        elif action >= 4 and action <= 7:
            observation[action-4] = np.clip(observation[action-4] + 0.001, 0.0, 1.0)
        elif action >= 9 and action <= 12:
            observation[action-9] = np.clip(observation[action-9] - 0.1, 0.0, 1.0)
        elif action >= 13:
            observation[action-13] = np.clip(observation[action-13] - 0.001, 0.0, 1.0)
        return observation



    def step(self, action):

        # Apply action
        self.observation = self.actionCalculation(action, self.parameter)
        # Checken ob Parameter über dem Maximum oder Minumum liegen, sonst wird bestraft
        
        # reduce episode by 1
        self.episode -= 1

        # Umstrukturieren der Daten, damit sie ins knn passen, da sie mit feature namen trainiert wurde
        predict_values = pd.DataFrame(
            (self.parameter), index=self.variable).transpose()
        # calculate reward
        
        CE_prediction, S_prediction, yield_prediction = self.knn.predict(predict_values)[0]
        #yield_stdev = float(self.knn.predict(predict_values)[1])
        # check if done
        terminated = False
        truncated = False

        # terminated wenn es nach 200 versuchen keine verbesserung findet
        if self.bestyield + self.bestCE + self.bestS < yield_prediction + CE_prediction + S_prediction:
            self.bestCE = CE_prediction
            self.bestyield = yield_prediction
            self.bestS = S_prediction
            self.yieldcount = 0
        else:
            self.yieldcount += 1

        if self.yieldcount >= 200:
            terminated = True

        # truncated wenn alle Episoden durchgechekt wurden
        if self.episode <= 0:
            truncated = True

        
        reward = yield_prediction + CE_prediction + S_prediction
        # reshape weil es damit einfacher ist
        observation = self.parameter.reshape(4,)
        # vermutlich eher observation.reshape
        info = {}

        # Die beste Option wird geprüft
        if self.bestOverallYield + self.bestOverallCE + self.bestOverallS < self.bestyield + self.bestCE + self.bestS:
            self.bestOverallCE = self.bestCE
            self.bestOverallYield = self.bestyield
            self.bestOverallS = self.bestS
            self.bestParameters = self.parameter
        self.debugCounter += 1
       

        # Damit das Terminal nicht jeden step gesprengt wird
        if self.debugCounter >= 20000:
            print(
                f"best yield: {self.bestOverallYield} best_CE: {self.bestOverallCE} best_S: {self.bestOverallS} \nwith parameters{self.bestParameters}")
            new_row_data = [self.bestOverallYield, self.bestOverallCE, self.bestOverallS, self.bestParameters[0], self.bestParameters[1], self.bestParameters[2], self.bestParameters[3]]
            df_summary.loc[len(df_summary)] = new_row_data
            #print(df_summary)
            df_summary.to_csv('summary_test.csv', index=False, mode='w')
            self.debugCounter = 0

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # reset parameter
        self.parameter = np.zeros((4,))
        self.parameter = np.array(self.parameter, dtype=np.float32)
        self.parameter = np.clip(self.parameter, 0.0, 1.0)
        self.yieldcount = 0
        self.bestCE = 0
        self.bestS = 0

        # reset episode
        self.episode = self.max_episodes

        # info, pretty much useless
        info = {}

        # observation = self._get_obs()
        return (self.parameter, info)

    def close(self):
        pass
