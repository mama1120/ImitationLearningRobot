# Robot programming HKA WS 2024

This repository contains the code for the robot programming course (the policy learning part) at the University of Applied Sciences Karlsruhe.

We will use and update this repository throughout the course. Hello.

## Quick start

### Environment setup

**Requirements:** have docker installed including the post-installation steps.

**Note:** The default settings are for nvidia GPU support. If you don't have an nvidia GPU, open up `build_image.sh` and set the `render` argument to `base`. Also, remove the `--gpus all` flag from the `docker run` command in `run_container.sh`.

Build the docker image with

```bash
./build_image.sh
```

Run the container with
```bash
./run_container.sh
```

Check whether you can open a window from the container by running
```bash
python stack_cubes.py
```

To test the tensorflow functionality:
Check whether you can open a window from the container by running
```bash
python test_model.py
```

To create a new dataset, run:
```bash
python auto_create_dataset_full.py
```
Combine the future state in the last state file:
```bash
python cmobine_json.py
```
Train the model:
```bash
python Cubes_Pose_gripper_train.py
```
Test the model output by giving an example input:
```bash
python test_model.py
```
Test the model in the enviromment:
```bash
python env_test_model.py
```
