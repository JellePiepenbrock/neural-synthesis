# Machine Learning Meets The Herbrand Universe

## Installation Instructions

### What you need
- It would be best if you install the Conda (https://docs.conda.io/en/latest/) package management system .
- A GPU with NVIDIA drivers that at least support CUDA 10.1
- Linux (tested on both Ubuntu 20.04 and NVIDIA DGX Server Version 5.2.0)
- GLIBC_2.29 version or higher (check with "ldd --version"); if you have a recent OS (2019) it should be there.

### Installs & Commands
```
cd mlmthu/
conda create --prefix ./cenv python=3.8.5 tqdm
conda activate ./cenv
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
pip install --no-cache-dir --no-index torch-scatter==2.0.6 -f https://data.pyg.org/whl/torch-1.7.0+cu101.html
```

Now, you need to find out where your conda install stores the "etc/profile.d/conda.sh" file and change 
CONDASH_LOCATION in inst_config.py to point to the right file (you can use "which conda" to find out). \


To run the system on 103 test problems (with 2 levels of instantiation), run

    python run_system.py

Usually (depending on the sampling), this will solve around 18% of the sample files.

And wait a few minutes. If that doesn't work, perhaps you don't have Perl installed on your system. If so, try:

    conda install -c conda-forge perl

### Other information

The sample data folder contains CNF files (a random 1000 samples from the whole dataset). 

After reviewing, this will be hosted on Github, with the whole dataset; during reviewing we can only upload 50MB. Code for the looping experiments is also here.

