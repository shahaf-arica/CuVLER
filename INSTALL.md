
# Installation

## Requirements
- Linux with Python ≥ 3.9
- PyTorch ≥ 2.0 and torchvision that matches the PyTorch installation. It is highly recommended to install them together at [pytorch.org](https://pytorch.org)
- Detectron2: follow [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) installation instructions.
- Dense CRF: you can get the latest version from [lucasb-eyer/pydensecrf](https://github.com/lucasb-eyer/pydensecrf)

## Environment setup instructions
```bash
conda create --name cuvler python=3.9 -y
conda activate cuvler
conda install pytorch torchvision -c pytorch
pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# under your working directory
pip install git+https://github.com/facebookresearch/detectron2.git
git clone --recursive git@github.com:shahaf-arica/CuVLER.git
cd CuVLER
pip install -r requirements.txt
```

## datasets
Next step is to prepare the data for training and evaluating, please see [datasets/README.md](datasets/README.md).
