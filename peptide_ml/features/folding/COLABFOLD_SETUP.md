For getting colab fold ready fro installing and setting up databases.

1. Ensure system has CUDA 12.6 and driver versions >560 and gcc = 9.4.0

then 

```
# Update package lists
sudo apt-get update
sudo apt-get install -y zlib1g-dev

# Install cmake and rhash
sudo apt-get install -y cmake rhash

# Clone MMseqs2 repository
git clone https://github.com/soedinglab/MMseqs2.git

# Build MMseqs2
cd MMseqs2
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=. ..
make -j$(nproc)
make install

# Add MMseqs2 to PATH
export PATH=$(pwd)/bin:$PATH
echo 'export PATH=$(pwd)/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify MMseqs2 installation
mmseqs version

# Install aria2c
sudo apt-get install -y aria2

# Verify aria2c installation
which aria2c

# Install additional tools if needed
sudo apt-get install -y curl wget rsync awscli

# Make setup script executable
chmod +x setup_databases.sh

# Run setup script
./setup_databases.sh
```

Now we must ensure we have latest CudNN

```
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.5.1.17_cuda12-archive.tar.xz

tar -xvf cudnn-linux-x86_64-9.5.1.17_cuda12-archive.tar.xz

# Copy header files
sudo cp cudnn-linux-x86_64-9.5.1.17_cuda12-archive/include/cudnn*.h /usr/local/cuda/include/

# Copy library files
sudo cp -P cudnn-linux-x86_64-9.5.1.17_cuda12-archive/lib/libcudnn* /usr/local/cuda/lib64/


# ============================
# Step 4: Set Appropriate File Permissions
# ============================
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*


echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Apply the changes to the current session
source ~/.bashrc

# ============================
# Step 6: Update the Library Cache
# ============================
sudo ldconfig
```

we must now install colab fold and setup the environment

install some dependancies

```
sudo apt-get update
sudo apt-get install -y curl git wget
```

Ensure gcc version (should be 9.4.0)

```
gcc --version
```
re-install to 9.4.0 if its less than 9

install ColabFold
To fisrt get the code:
```
it clone https://github.com/sokrypton/ColabFold.git
```

```
wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh
```

Run install script


```
bash install_colabbatch_linux.sh
```

Add to PATH
head into localcolabfold directory 
```
/localcolabfold/colabfold-conda/
```
```
echo 'export PATH=$(pwd)/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

set ENV variables

```
echo 'export TF_FORCE_UNIFIED_MEMORY="1"' >> ~/.bashrc
echo 'export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"' >> ~/.bashrc
echo 'export XLA_PYTHON_CLIENT_ALLOCATOR="platform"' >> ~/.bashrc
echo 'export TF_FORCE_GPU_ALLOW_GROWTH="true"' >> ~/.bashrc
source ~/.bashrc
```

you will then need to use this as a conda environment OK
so you willl have a conda env at
```

/home/{USER}/peptide_templating/peptide_templating/features/folding/ColabFold/localcolabfold/
```
so you can actually register this as a named conda env with 

```
conda env list  # To confirm it's not listed
conda config --append envs_dirs /home/{USER}/peptide_templating/peptide_templating/features/folding/ColabFold/localcolabfold/
```
REPLACE USER with you user

then if you want a shorter terminal conda env you can create a symlink to the regular conda envs on yoru system eg 

```
ln -s /path/to/ColabFold/localcolabfold/colabfold-conda ~/anaconda3/envs/colabfold-conda

conda config --set env_prompt '({name}) '
```

and then to actiavte 
```
conda activate colabfold-conda
```

Also install pymol for vis
```
conda install -c conda-forge pymol-open-source
```

this pymol dosnt actually run in the python env as you can see in teh fold_all.py script its triggered as a subprocess.

then also install dssp 

```
sudo apt-get install dssp
```

Then lastly to run:

```
colabfold_batch   --msa-mode mmseqs2_uniref_env   --data /storage-1/peptide_templating/databases   --templates   --amber   input_sequences.fasta   temp_out_dir/
```
assuming you have a fasta file something like 
```
>MyProtein
MALKSLVLLSLLVLVLLLVRVQPSLGKETAAAKFERQHMDSSTSAASSSNYCNQMMKSRN
LTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPN
CAYKTTQANKHIIVACEGNPYVPVHFDASV
```
However it should manage multiple proteins 


The to visualize first we create a visualization conda env

```
conda create -n pymol-env python=3.8
conda activate pymol-env
```

install vis packages
```
conda install -c schrodinger pymol-bundle
sudo apt-get update
sudo apt-get install -y pymol
```
MAKE SURE YOU HAVE DEACTIVATED ALL OTHER CONDA ENVS
we have then created a visualization script 

visualize_pdbs.py

which we then need to make executable 

```
chmod +x visualize_pdbs.py
```
now cd into the directory with your protien outputs and run something like
b

```
../visualize_pdbs.py MyProtein_relaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
``` 
assuming the script for viz is one dir down

this command line arg is the protein you want to visualize , generally this will be the relaxed one (after potential minimization)


