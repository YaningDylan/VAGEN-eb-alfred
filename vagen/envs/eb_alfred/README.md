# EB-ALFRED Environment

EB-ALFRED integrates [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench)'s AI2-THOR household tasks into the VAGEN framework.

## Installation

```bash
conda env create -f environment.yaml && conda activate embench

git clone https://github.com/EmbodiedBench/EmbodiedBench && pip install -e EmbodiedBench

pip install --no-deps -e ../../..

git lfs install && git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
```

## Setup (per machine restart)

```bash
# Install Xvfb if needed
apt install -y xvfb

# Start one virtual display per GPU
Xvfb :0 -screen 0 1024x768x24 &
Xvfb :1 -screen 0 1024x768x24 &
Xvfb :2 -screen 0 1024x768x24 &
Xvfb :3 -screen 0 1024x768x24 &

# Start env server
conda activate embench
python -m vagen.envs.eb_alfred.serve --port 8000 --x-displays 0,1,2,3
```

## Evaluation

```bash
conda activate vagen
python -m vagen.evaluate.run_eval --config <config_file>
```

Available configs:

| Config | n_envs | max_concurrent_jobs |
|--------|:------:|:-------------------:|
| `tests/eval_eb_alfred_gpt41_20ep.yaml` | 20 | 3 |
| `tests/eval_eb_alfred_gpt41_128ep_parallel_500.yaml` | 128 | 100 |
| `examples/evaluate/eb_alfred/config.yaml` | 64 | 16 |

## Training

```bash
conda activate vagen
bash examples/eb_alfred/train_ppo_no_concat_qwen25vl3b.sh
```
