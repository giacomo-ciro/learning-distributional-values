<h1 align="center">Learning Distributional Value Functions</h1>

> We learn a distributional value function from robot rollouts labeled only as successes or failures, producing dense, frame-level signals that make failed trajectories useful for VLA training.

<p align="center">
  <img src="dump/example.png" alt="Camera frames and predicted value over a successful episode" width="800">
</p>

**Figure 1: Value along an episode.** *Predicted value over a successful validation episode. When the policy fails to grasp the actuator, the value drops sharply; once it recovers with a correct grasp, the value rises again.*

Robot rollouts are easy to collect, but they usually come with a single binary label: whether the episode succeeded or failed. This project learns a dense, frame-level value function from such data, so that every state of every rollout, failed ones included, gets an estimate of how close it is to a quick success. The difference in value between two states gives the advantage of the actions in between, which can be used to exploit suboptimal data during VLA training, as done in π\*0.6 [1].

We work on an [`actuator unboxing task`](https://huggingface.co/datasets/DreamMachines/20h_fullft_eval_success) (800 rollouts, roughly half successful). Each observation is a set of three frames (left wrist, right wrist, top camera), and the model predicts the value of that state.

The binary outcome is turned into a dense reward. For an episode of length $T$, the reward at step $t$ is

<p align="center">
  <img src="dump/reward.png" alt="Reward definition: 0 at a successful terminal step, -C at a failed terminal step, and -1 otherwise" width="400">
</p>

where $C_{fail}$ is a large constant. The value target of each frame is its return-to-go, normalized to $[-1, 0]$, so the value is higher for states closer to a quick success.

The key design choice that made training succeed was setting $C_{fail}$ to the maximum episode length in the training set. Returns from failed episodes then fall between $-2C$ and $-C$. We divide each return by $C_{fail}$ and clip values below $-1$, mapping every frame from a failed episode to $-1$. This is intentional: the model should learn that an episode will fail, not how long the operator will wait before unplugging the robot. This normalization substantially improves training and enables the model to learn a meaningful value function.

Instead of regressing the value directly, the model predicts a distribution over discretized value bins and is trained with cross-entropy against HL-Gauss soft targets [3]. The architecture is a ResNet-101 [2] backbone that encodes each frame independently. The features are then concatenated and fed to a two-layer GELU MLP that predicts the logits for the bins.

<p align="center">
  <img src="dump/eval.png" alt="Agreement between human judgement and predicted advantages" width="400">
</p>

**Figure 2: Human agreement with predicted advantages.** *An annotator watches two intervals from the same episode, and picks the one that went better. Agreement is how often that pick matches the interval with the higher predicted advantage. Random is chance (50%). ResNet101 is trained with one-hot cross-entropy, and +HLGauss uses soft targets with σ = 0.75.*

> **Disclaimer:** these are preliminary results. Each model was scored on only 14 annotated pairs.

## Reproducing the Results

Download the dataset into `data/`, where [`configs/train.yaml`](configs/train.yaml) expects it:

```bash
hf download DreamMachines/20h_fullft_eval_success --repo-type dataset --local-dir data/20h_fullft_eval_success
```

Train the two models in Figure 2:

```bash
# ResNet101 (+HLGauss), σ = 0.75
python scripts/train.py trainer.hl_gauss_sigma=0.75 run_name=resnet101_ffnn_hlgauss_0.75

# ResNet101, one-hot cross-entropy
python scripts/train.py trainer.hl_gauss_sigma=0.0 run_name=resnet101_ffnn
```

Then set `RUN_NAME` in [`scripts/annotate.py`](scripts/annotate.py) to each run and annotate the pairs with `python scripts/annotate.py`.

## Repository Structure

```
├── configs/      # Hydra configs
├── src/          # data module, models, trainer
├── scripts/      # train.py, test.py, annotate.py
├── notebooks/    # data exploration, evaluation, loss analysis
├── dump/splits/  # fixed train/val/test episode splits
├── data/         # datasets (git-ignored)
└── checkpoints/  # trained models (git-ignored)
```

A presentation of this work is available in [the slides](dump/slides.pdf).

## References

1. Physical Intelligence. [π\*0.6: a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759). arXiv:2511.14759, 2025.
2. K. He, X. Zhang, S. Ren, J. Sun. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). CVPR, 2016.
3. J. Farebrother et al. [Stop Regressing: Training Value Functions via Classification for Scalable Deep RL](https://arxiv.org/abs/2403.03950). ICML, 2024.
