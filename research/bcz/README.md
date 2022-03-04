# BC-Z

Source codes for reproducing "BC-Z: Zero-Shot Task Generalization with Robotic
Imitation Learning".

Links

-   [Project Website](https://sites.google.com/view/bc-z/home/)
-   [Paper](https://arxiv.org/abs/2202.02005)
-   [Google AI Blog Post](https://ai.googleblog.com/2022/02/can-robots-follow-instructions-for-new.html)

## Training the Model

Download the data in the TFRecords format is open-sourced here:
https://www.kaggle.com/google/bc-z-robot

```
TRAIN_DATA="/path/to/bcz-21task_v9.0.1.tfrecord/train*,/path/to/bcz-79task_v16.0.0.tfrecord/train*"
EVAL_DATA="/path/to/bcz-21task_v9.0.1.tfrecord/val*,/path/to/bcz-79task_v16.0.0.tfrecord/val*"
python3 -m tensor2robot.bin.run_t2r_trainer --logtostderr \
  --gin_configs="tensor2robot/research/bcz/configs/run_train_bc_langcond_trajectory.gin" \
  --gin_bindings="train_eval_model.model_dir='/tmp/bcz/'" \
  --gin_bindings="TRAIN_DATA='${TRAIN_DATA}' \
  --gin_bindings="EVAL_DATA='${EVAL_DATA}'"
```
