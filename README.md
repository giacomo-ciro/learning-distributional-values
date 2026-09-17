# Learning Distributional Value Functions

Pretraining of the value function from [π*0.6: a VLA that Learns from Experience](docs/arXiv-2511.14759v2/main.tex) on an actuator unboxing task, using the [`20h_fullft_eval_success`](https://huggingface.co/datasets/DreamMachines/20h_fullft_eval_success) dataset (~800 rollouts, roughly half successful).

Each observation is a set of three frames (left wrist, right wrist, top camera), and the model predicts the value of that state.

The labels come from the episode outcome: every step has reward -1, and the last step has reward 0 on success or -C on failure, so the value is higher for states closer to a quick success.

The model is trained as a classifier over discretized value bins, using cross-entropy with HL-Gauss soft targets. The architecture is a ResNet-101 backbone that encodes each frame independently. The features are then concatenated and fed to a two-layer GELU MLP that predicts the logits for the bins.

The predicted values are used to estimate action advantages for VLA training.

![Agreement between human judgement and predicted advantages](dump/eval.png)

*An annotator watches two intervals from the same episode, and picks the one that went better. Agreement is how often that pick matches the interval with the higher predicted advantage. Random is chance (50%). ResNet101 is trained with one-hot cross-entropy, and +HLGauss uses soft targets with σ = 0.75.*

> **Disclaimer:** these are preliminary results. Each model was scored on only 14 annotated pairs.

## Repository Structure

```
├── configs/      # Hydra configs
├── src/          # data module, models, trainer
├── scripts/      # train.py, test.py, annotate.py
├── notebooks/    # data exploration, evaluation, loss analysis
├── docs/         # the π*0.6 paper
├── dump/splits/  # fixed train/val/test episode splits
├── data/         # datasets (git-ignored)
└── checkpoints/  # trained models (git-ignored)
```
