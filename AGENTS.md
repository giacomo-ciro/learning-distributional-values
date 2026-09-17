# Project: Implementing the Value Function from the π*0.6 paper

The goal is to implement the value function described in [π*0.6: a VLA that Learns from Experience](docs/arXiv-2511.14759v2/main.tex), which takes as input an observation (set of frames from different robot-mounted cameras) and returns the estimated value.

At training time, this will be used to estimate the advantage of an action (difference of values at two states before an after the action) to be used as label during VLA pretraining so that bad trajectories can be exploited as well.

We focus solely on the value function pretraining.

We have a dataset of 800 episodes for an actuators unboxing task. Each sample is made of 3 frames from the left wrist, rigth wrist and top camera.

In this project we have to first create the labels for each frame (value), and then train with supervised learning a model to predict the value from the frames. The original paper trains a Gemma 3 (670M) model. We try that and a simpler convnet, which might be enough for our simpler setting.

Each episode is labeled with 0/1 for failure/success. The reward r_t for an episode of length T is defined as follows.

r_t = 0 if t=T and success
r_t = -C if t=T and fail
r_t = -1 otherwise

Where C is a big constant (>= max episode length). This is to get a frame-level value signal and incentive short successes.