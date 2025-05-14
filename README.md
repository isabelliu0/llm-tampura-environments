# TAMPURA Environments
The environments associated with the [TAMPURA planner](https://github.com/aidan-curtis/tampura)

## Installation
Set up the conda environment
```
conda create -n tampura_env python=3.11.10
python -m pip install -e .
conda install -c conda-forge pygraphviz
```

## Paper environments
```
python run_planner.py --config=./env_configs/tool_use.yml --vis=1 --global-seed=0 --vis-graph=1

python run_planner.py --config=./env_configs/slam_collect.yml --vis=1 --global-seed=0 --vis-graph=1

python run_planner.py --config=./env_configs/find_dice.yml --vis=1 --global-seed=0 --vis-graph=1

python run_planner.py --config=./env_configs/class_uncertain.yml --vis=1 --global-seed=0 --vis-graph=1

python run_planner.py --config=./env_configs/puck_slide.yml --vis=1 --global-seed=0 --vis-graph=1
```
