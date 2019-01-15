# Tensor2Robot

Tensor2Robot implements distributed machine learning and reinforcement learning infrastructure for several research projects within Robotics @ Google.

## Projects using this Code

- QT-Opt.
- Grasp2Vec

## Features and Design Decisions

- Scalability: This codebase is designed for training large-scale robotic perception models with algorithms that do not require a tight perception-action-learning loop (supervised learning, off-policy reinforcement learning of large computer vision models). For example, one setup might involve multiple GPUs pulling data asynchronously from a replay buffer and training with an off-policy RL algorithm, while data collection agents periodically update their checkpoints and push experiences to the same replay buffer. This can also be run on a single workstation for smaller-scale experiments.

- Due to the sizes of models we work with (e.g. grasping from vision) Inference is assumed to be within 1-10Hz and without real-time guarantees. If you are learning small networks (e.g. two-layer perceptron) with on-policy RL (e.g. PPO), this is probably not the right codebase to use.

- Minimize boilerplate: A common pattern in robotics research involves adding a new sensor modality to a neural network. This involves 1) changing what data is saved, 2) changing data pipeline code to read in new modalities at training time 3) adding a new `tf.placeholder` to handle the new input modality at test time. Tensor2Robot Models declare their own "Feature Specifications", which allow (2) and (3) to happen automatically. Another feature of this codebase is static verification of run-time data with TFModels and their promised input and output specifications, making it easy to debug discrepancies.

- TPU support: Easy to deploy a model to train on TPUs and GPUs.

- gin-configurable: [Gin-Config](https://github.com/google/gin-config) is used to flexibly configure experiments and model definitions.


## Disclaimer

This is not an official Google product. External support not guaranteed. Please email us before working on a pull request.
