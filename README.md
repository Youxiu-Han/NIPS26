Code for "Memory-Driven Contrastive Embedding Enhancement for Fine-Grained Open-Set Semi-Supervised Learning", built on the `semilearn` training framework. The main algorithm see `semilearn/algorithms/textmatch/`). Vision–language components are configured via CLIP options in the YAML files.

## Requirements

- Python 3
- PyTorch (match your CUDA version)
- Other dependencies: `numpy`, `torchvision`, and common CV/ML utilities

Prepare each dataset under the path set in the config (`data_dir`, default `./data`). Obtain dataset from each dataset’s official release.

## Running

The primary entry point is:

```bash
bash train.sh
```

Before running, edit `train.sh` as needed:

- `CUDA_VISIBLE_DEVICES`: GPU index(es) to use
- `config_file`: path to a config (e.g. `config/textmatch_cub200_2000_0.yaml`)

Equivalent direct invocation:

```bash
python train.py --c config/textmatch_cub200_2000_0.yaml
```

Training hyperparameters are taken from the YAML; `train.py` can override some options from the command line. 

## Config files

Example presets under `config/` for different datasets and label budgets:

| Config | Dataset | Notes |
| ------ | ------- | ----- |
| `textmatch_cub200_2000_0.yaml`, `textmatch_cub200_500_0.yaml` | CUB-200-2011 | Different label counts |
| `textmatch_dog120_1200_0.yaml`, `textmatch_dog120_300_0.yaml` | Stanford Dogs | Different label counts |
| `textmatch_car196_1960_0.yaml`, `textmatch_car196_490_0.yaml` | Stanford Cars | Different label counts |
| `textmatch_aircraft100_1000_0.yaml`, `textmatch_aircraft100_250_0.yaml` | FGVC Aircraft | Different label counts |

For a new run, you typically adjust `save_name`, `dataset`, `num_classes`, `num_labels`, `data_dir`, `gpu`.

## Repository layout

- `train.py` — training entry
- `train.sh` — example launcher for `train.py` with a chosen YAML
- `config/` — experiment configs
- `semilearn/` — algorithms, datasets, training loop
- `clip/` — CLIP-related code

## Citation

Official code release for the paper. Add BibTeX here once the paper is public.
