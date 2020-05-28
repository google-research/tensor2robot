# QT-Opt

This directory contains network architecture definitions for the Grasping critic
architecture described in [QT-Opt: Scalable Deep Reinforcement Learning for
Vision-Based Robotic Manipulation](https://arxiv.org/abs/1806.10293).

## Running the code

The following command trains the QT-Opt critic architecture for a few gradient
steps with mock data (real data is not included in this repo). The learning
obective resembles supervised learning, since Bellman targets in QT-Opt are
computed in a separate process (not open-sourced).

```
git clone https://github.com/google/tensor2robot
# Optional: Create a virtualenv
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -r tensor2robot/requirements.txt
python -m tensor2robot.research.qtopt.t2r_models_test
```

## PCGrad

This directory also contains a multi-task optimization method
[PCGrad](https://arxiv.org/pdf/2001.06782.pdf) that is implemented in the form
of a optimization wrapper. This is based on the open-source implementation
[here](https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py).
