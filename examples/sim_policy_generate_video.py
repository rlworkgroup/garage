#!/usr/bin/env python3
"""Simulates pre-learned policy."""
import argparse
import os
import sys

import cloudpickle
from gym.wrappers import Monitor, TimeLimit
import numpy as np
import tensorflow as tf

import cv2
from garage import StepType
from garage.envs import GymEnv
from garage.sampler.utils import rollout
from garage.torch import set_gpu_mode


def trajectory_generator(env, policy, res=(640, 480)):

    env.reset()
    env.reset_model()
    o = env.reset()[0]

    for _ in range(env.max_path_length):
        a = policy.get_action(o)[0]
        timestep = env.step(a)
        o = timestep.observation
        done = timestep.step_type == StepType.TERMINAL
        yield timestep.reward, done, timestep.env_info, env.sim.render(
            *res, mode='offscreen', camera_name='corner')[:, :, ::-1]


def writer_for(tag, fps, res):
    if not os.path.exists('movies'):
        os.mkdir('movies')
    return cv2.VideoWriter(f'movies/{tag}.mp4',
                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                           res)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.compat.v1.Session():
    #     [rest of the code]
    with tf.compat.v1.Session() as sess:
        set_gpu_mode(True)
        with open(args.file, mode='rb') as fi:
            data = cloudpickle.load(fi)
        policy = data['algo'].policy
        env = data['env']
        resolution = (640, 480)
        writer = writer_for(
            'pick-place-solved-5m',
            env.metadata['video.frames_per_second'], resolution)
        for _ in range(10):
            for r, done, info, img in trajectory_generator(
                    env, policy, resolution):
                # img = cv2.rotate(img, cv2.ROTATE_180)
                writer.write(img)
        writer.release()
