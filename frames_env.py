import gym
import numpy as np
import cv2

def show_img(observation):
    window_title = "Juego"

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
    cv2.imshow(window_title, observation)
    
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        cv2.destroyAllWindows()

class FramesEnv(gym.Wrapper):
    def __init__(self, env):
        super(FramesEnv, self).__init__(env)
    
    def reset(self, **kwargs):
        self.last_life_count = 0
        observation = self.env.reset(**kwargs)
        observation = self.process_img(observation)
        observation = np.stack([observation])

        return observation

    def step(self, action):
        one_hot_action = np.zeros(3)
        one_hot_action[action] = 1
            
        observation, reward, done = self.env.step(one_hot_action)
        observation = self.process_img(observation)
        show_img(observation)

        observation = np.stack([observation])

        reward = 4 if reward > 0 else -15

        return observation, reward, done

    def process_img(self, image):
        image = cv2.resize(image, (80,80))
        return  image

