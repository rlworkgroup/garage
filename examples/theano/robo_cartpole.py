import gym, roboschool
import numpy as np
import random
import garage.misc.logger as logger


def demo_run():
    env = gym.make("RoboschoolInvertedPendulum-v1")
    iter_time = 20
    for _ in range(iter_time):
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            obs, r, done, _ = env.step(a)
            score += r
            frame += 1
            still_open = env.render("human")
            if still_open==False:
                return
        logger.log("score=%0.2f in %i frames" % (score, frame))

demo_run()
