# VRGripper Environment Models

Contains code for training models in the VRGripper environment from
"Watch, Try, Learn: Meta-Learning from Demonstrations and Rewards."

Includes the models used in the Watch, Try, Learn (WTL) gripping experiments.

Links

-   [Project Website](https://sites.google.com/corp/view/watch-try-learn-project)
-   [Paper Preprint](https://arxiv.org/abs/1906.03352)

## Authors

Allan Zhou<sup>1</sup>, Eric Jang<sup>1</sup>, Daniel Kappler<sup>2</sup>,
Alex Herzog<sup>2</sup>, Mohi Khansari<sup>2</sup>,
Paul Wohlhart<sup>2</sup>, Yunfei Bai<sup>2</sup>,
Mrinal Kalakrishnan<sup>2</sup>, Sergey Levine<sup>1,3</sup>,
Chelsea Finn<sup>1</sup>

<sup>1</sup> Google Brain, <sup>2</sup>X, <sup>3</sup>UC Berkeley

## Training the WTL gripping experiment models.

WTL experiment models are located in `vrgripper_env_wtl_models.py`.
Data is not included in this repository, so you will have to provide your own
training/eval datasets. Training is configured by the following gin configs:

* `configs/run_train_wtl_statespace_trial.gin`: Train a trial policy on
state-space observations.
* `configs/run_train_wtl_statespace_retrial.gin`: Train a retrial policy
on state-space observations.
* `configs/run_train_wtl_vision_trial.gin`: Train a trial policy on image
observations.
* `configs/run_train_wtl_vision_retrial.gin`: Train a retrial policy on
image observations.
