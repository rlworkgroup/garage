import gym, roboschool
import numpy as np
import random

class NaivePolicy:
    "Simple multi-layer perceptron policy, no internal state"
    def __init__(self, observation_space, action_space):
        return

    def act(self, ob):
        x = np.array([0.5-random.random()])
        return x

def demo_run():
    env = gym.make("RoboschoolInvertedPendulum-v1")
    pi = NaivePolicy(env.observation_space, env.action_space)

    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()

        while 1:
            a = pi.act(obs)
            obs, r, done, _ = env.step(a)
            score += r
            frame += 1
            still_open = env.render("human")
            if still_open==False:
                return
            if not done: continue
            if restart_delay==0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60*2  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay==0: break


if __name__=="__main__":
    demo_run()