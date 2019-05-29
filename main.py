#!/usr/bin/env python3
#########################################################################
# > File Name: main.py
# > Author: Samuel
# > Mail: enminghuang21119@gmail.com
# > Created Time: Sun May 26 00:36:48 2019
#########################################################################
import numpy as np
from PIL import Image
import cv2 #opencv
import io
import time
import pandas as pd
import numpy as np
# from IPython.display import clear_output
from random import randint
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

import tensorflow as tf

#keras imports
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD , Adam
from tensorflow.keras.callbacks import TensorBoard
from collections import deque
import random
import pickle
from io import BytesIO
import base64
import json

#game parameters
ACTIONS = 2 # 0: nothing; 1: jump
GAMMA = 0.99 # Decay rate of past observations original 0.99
OBSERVATION = 200 # Timesteps to observe before training
EXPLORE = 100000  # Frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 16 # size of minibatch
FRAME_PER_ACTION = 1
FPS_LIMIT = 7.6
LEARNING_RATE = 1e-4
img_rows , img_cols = 80,80
img_channels = 4 # Stack 4 frames

# Path variables
from pathlib import Path
game_url = 'file://' + str(Path().absolute()) + '/game/index.html'
loss_file_path = "./objects/loss_df.csv"
actions_file_path = "./objects/actions_df.csv"
q_value_file_path = "./objects/q_values.csv"
scores_file_path = "./objects/scores_df.csv"


# Tensorflow-gpu
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# Scripts
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

# Get image from canvas
getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"

'''
* Game class: Selenium interface
* __init__():  Launch firefox
* get_crashed() : True if crash
* get_playing(): True if playing
* restart() : Restart
* press_up(): Send up key
* get_score(): Get current score
* pause(): Pause the game
* resume(): Resume a pause game
* end(): Close browser
'''
class Game:
    def __init__(self,custom_config=True):
        self._driver = webdriver.Firefox()
        self._driver.set_window_position(x=-10,y=0)
        self._driver.get(game_url)
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(init_script)
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")
    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array) # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)
    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")
    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")
    def end(self):
        self._driver.close()

class DinoAgent:
    def __init__(self,game):
        self._game = game; 
        self.jump(); # Start game
    def is_running(self):
        return self._game.get_playing()
    def is_crashed(self):
        return self._game.get_crashed()
    def jump(self):
        self._game.press_up()

def prRed(string): print("\033[91m{}\033[00m".format(string)) 

class Game_sate:
    def __init__(self, agent,game):
        self._agent = agent
        self._game = game
        self._display = show_img() # Display the processed image on screen using openCV
        self._display.__next__() # Init the display
    def get_state(self, actions):
        actions_df.loc[len(actions_df)] = actions[1] # Storing actions in a dataframe
        score = self._game.get_score() 
        reward = 0.1
        is_over = False # game over
        if actions[1] == 1:
            self._agent.jump()
            reward = -6
        image = grab_screen(self._game._driver) 
        self._display.send(image) # Display the image on screen
        if self._agent.is_crashed():
            scores_df.loc[len(loss_df)] = score # Log the score when game is over
            self._game.restart()
            reward = -1000
            is_over = True
        return image, reward, is_over # Return the Experience tuple
    def get_score(self):
        return self._game.get_score()

def save_obj(obj, name):
    with open('objects/'+ name + '.pkl', 'wb') as f: # Dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name ):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def grab_screen(_driver):
    image_b64 = _driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)#
    return image

def process_img(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # RGB to Grey Scale
    image = image[:300, :500] # Crop Region of Interest(ROI)
    image = cv2.resize(image, (80, 80))
    return  image

def show_img(graphs = False):
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
        imS = cv2.resize(screen, (800, 400)) 
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

#Intialize log structures from file if exists else create new
loss_df = pd.read_csv(loss_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns =['loss'])
scores_df = pd.read_csv(scores_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns = ['scores'])
actions_df = pd.read_csv(actions_file_path) if os.path.isfile(actions_file_path) else pd.DataFrame(columns = ['actions'])
q_values_df =pd.read_csv(actions_file_path) if os.path.isfile(q_value_file_path) else pd.DataFrame(columns = ['qvalues'])

def init_cache():
    """Initial cache"""
    save_obj(INITIAL_EPSILON,"epsilon")
    t = 0
    save_obj(t,"time")
    Log = deque()
    save_obj(Log,"Log")

def buildmodel():
    print("=======Building the model=======")
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same',strides=(4, 4),input_shape=(img_cols,img_rows,img_channels)))  #80*80*4
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4),strides=(2, 2),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3),strides=(1, 1),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    
    # Create model file if dosen't exist
    if not os.path.isfile(loss_file_path):
        model.save_weights('./objects/model.h5')
    print("=======Finish building the model=======")
    return model

def get_fps(pre, now):
    return 1 / (now - pre)

def trainNetwork(model, game_state, training = True, observe=False):
    last_time = time.time()
    # Store the previous observations in replay memory
    Log = load_obj("Log") #load from file system
    # Get the first state by doing nothing
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1 # 0 => do nothing,
                      # 1 => jump
    
    x_t, r_0, terminal = game_state.get_state(do_nothing) # Get next step
    

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) # Stack 4 images to create placeholder
    

    
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*20*40*4
    
    initial_state = s_t 

    if observe :
        OBSERVE = 999999999    # Keep observe, never train
        epsilon = FINAL_EPSILON
        print ("=======Loading weight=======")
        model.load_weights("./objects/model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                       # Training mode
        OBSERVE = OBSERVATION
        epsilon = load_obj("epsilon") 
        model.load_weights("./objects/model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)

    t = load_obj("time") # Resume from the previous time step stored in file system
    tt = t
    last_score = 0;
    while (True):
        
        print_red = 0
        loss = 0
        Q_sa = 0
        action_index = -1
        r_t = 0
        a_t = np.zeros([ACTIONS])
        # Choose an action
        if t % FRAME_PER_ACTION == 0:
            if  random.random() <= epsilon * int(training): # Random action
                prRed("----------Random Action----------")
                print_red = 1
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else: # Predict output
                q = model.predict(s_t)       
                max_Q = np.argmax(q)         # Chose index with maximum q value
                action_index = max_Q 
                a_t[action_index] = 1        # 0=> do nothing; 1=> jump
                
        # Reduced the epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE 

        x_t1, r_t, terminal = game_state.get_state(a_t)
        while get_fps(last_time, time.time()) > FPS_LIMIT:
            pass
        print('fps: {0}'.format(get_fps(last_time, time.time()))) # Print fps
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x20x40x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) # Update image array
        
        
        # Store the transition in Log
        Log.append((s_t, action_index, r_t, s_t1, terminal))
        if len(Log) > REPLAY_MEMORY:
            Log.popleft()
        # time.sleep(.04)
        # Train if done observing
        if t > OBSERVE: 
            
            # Sample a minibatch to train
            minibatch = random.sample(Log, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 20, 40, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            # Train minibatch
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]    # 4D stack of images
                action_t = minibatch[i][1]   # Action index
                reward_t = minibatch[i][2]   # Reward at state_t due to action_t
                state_t1 = minibatch[i][3]   # Next state
                terminal = minibatch[i][4]   # Dino status
                

                inputs[i:i + 1] = state_t    

                targets[i] = model.predict(state_t)  # Predict q values
                Q_sa = model.predict(state_t1)      # Predict q values for next step
                
                if terminal:
                    targets[i, action_t] = reward_t # If terminated, only equals reward
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
            q_values_df.loc[len(q_values_df)] = np.max(Q_sa)
        s_t = initial_state if terminal else s_t1 # Reset game to initial frame if died
        t += 1 

        # Save progress every 1000 iterations
        if training and t % 1000 == 0:
            print("Saving model")
            game_state._game.pause() # Pause game while saving to filesystem
            model.save_weights("./objects/model.h5", overwrite=True)
            save_obj(Log, "Log") # Savie episodes
            save_obj(t, "time") # Cache time steps
            save_obj(epsilon, "epsilon") # Cache epsilon to avoid repeated randomness in actions
            loss_df.to_csv("./objects/loss_df.csv", index = False)
            scores_df.to_csv("./objects/scores_df.csv", index = False)
            actions_df.to_csv("./objects/actions_df.csv", index = False)
            q_values_df.to_csv(q_value_file_path, index = False)
            with open("./objects/model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)
            # clear_output()
            # os.system('clear')
            game_state._game.resume()

        # Print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        # print("TIMESTEP", t, "/ STATE", state,             "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,             "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)
        string = str(t) + '# / ACTION: ' + str(action_index) + ' / SCORE: ' + str(game_state.get_score()) + ' / TERMINAL: ' + str(int(r_t) == -1000) + ' / EPSILON:' + str(epsilon)
        if print_red:
            prRed(string)
        else:
            print(string)

    print("************************")
    print("*       Finished!      *")
    print("************************")

#main function
def playGame(training = True):
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_sate(dino, game)    
    model = buildmodel()
    try:
        trainNetwork(model, game_state, training = training, observe = not training)
    except StopIteration:
        game.end()

#init_cache()             # Must be called to create cache file(INIT log file)
playGame(training = False); # Change True to False to run program without training
