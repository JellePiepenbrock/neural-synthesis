# Machine Learning Meets The Herbrand Universe

## Installation Instructions

### What you need
- It would be best if you install the Conda (https://docs.conda.io/en/latest/) package management system .
- A GPU with NVIDIA drivers that at least support CUDA 10.1
- Linux (tested on Ubuntu)
```
cd mlmthu/
conda create --prefix ./cenv python=3.8.5 tqdm
conda activate ./cenv
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
pip install --no-cache-dir --no-index torch-scatter==2.0.6 -f https://data.pyg.org/whl/torch-1.7.0+cu101.html
```
To run the system on 103 test problems, run

    python run_system.py

And wait a few minutes. If that doesn't work, perhaps you don't have Perl installed on your system. If so, try:

    conda install -c conda-forge perl


