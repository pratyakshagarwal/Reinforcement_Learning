# Mss used for screen cap
from mss import mss
# Sending commands
import pydirectinput
# Opencv allows us to do frame processing
import cv2
import numpy as np
# OCR for game over extraction
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\\Users\\praty\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"
# visulize captured frames
from matplotlib import pyplot as plt
import time
# Enviroment components
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete


# Building The Custom Enviorment
class WebGame(Env):
    # setup the env action and obs shapes
    def __init__(self):
        super().__init__()
        # setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # Define Extraction parameters for the game
        self.cap = mss()
        self.game_location = {'top':300, 'left':0, 'width':600, 'height':500}
        self.done_location = {'top':405, 'left':630, 'width':660, 'height':100}

    # What is called to do something in the game
    def step(self, action):
        # Action key - 0 -> Space, 1 -> Duck(down), 2 -> No action(np op)
        action_map = {
            0:'space',
            1:'down',
            2:'no_op'
        }
        if action != 2:
            pydirectinput.press(action_map[action])

        # Checking whether the game is done 
        done, done_cap  = self.get_done()
        # Get the next observation
        new_observation = self.get_observation()
        # Reward we get a point 1 for every frame alive
        reward =  1
        # Info dictionary
        info = {}

        return new_observation, reward, done, info
    
    # visulize the game
    def render(self):
        while True:
            frame = np.array(self.cap.grab(self.game_location))[:, :, :3]
            cv2.imshow('GAME', frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                self.close()
                break


    # This closes down the observation
    def close(self):
        cv2.destroyAllWindows()

    # Restart the game
    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_observation()

    # Get the part of the obs of the game that we want
    def get_observation(self):
        # Get screen Capture of game
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3].astype(np.uint8)
        # Grayscale 
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # resize
        resized = cv2.resize(gray, (100, 83))
        # Add channel first
        channel = np.reshape(resized, (1, 83, 100))

        return channel
    
    # Get the done text using ocr
    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))[:, :, :3]
        # Valid done text
        done_string = ['GAME', 'GAHE']

        # Apply OCR
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_string:
            done = True

        return done, done_cap

if __name__ == "__main__":
    env = WebGame()
    
    for episode in range(10):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            total_reward += reward
        print(f"total reward for episode {episode} is {total_reward}")