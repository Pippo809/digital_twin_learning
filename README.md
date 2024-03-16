# Digital Twin Learning

 The module to learn edge costs for online re-planning and learn node sampling strategies.

## Installation

1. After you cloned this repo, download the resources (clouds, graphs, meshes) [here](https://drive.google.com/drive/folders/13CbzbXhQIbneh3H8CGd4-4jRV-74kRRN?usp=sharing) and add the folder `resources` to this repo's folder.

2. Create a new Python virtual environment (e.g. using the `venv` module that can be installed with `sudo apt install python3-venv`):

        cd ~/venvs
        python3 -m venv navigation

3. Activate the environment and install PyTorch 1.11 with CUDA 11.3:

        source ~/venvs/navigation/bin/activate
        pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113


### Digital Twin Learning

1. Install the learning module:

        cd digital_twin_navigation/digital_twin_learning
        pip3 install -r requirements.txt
        pip3 install -e digital_twin_learning

2. If needed to display the Tensorboard 3D scenes correctly, [Open3D](http://www.open3d.org/) is needed and it might have to be built from source, 
in case Pypi version does not work follow these steps, firstly clone the repo:

        git clone https://github.com/isl-org/Open3D

3. Install the dependencies

        util/install_deps_ubuntu.sh

4. Config (If needed install Cmake from apt/snap)

        mkdir build
        cd build
        cmake ..

5. Build

        make -j$(nproc)

6. Install (Sudo might be needed depending on where the package is built)

        make install-pip-package


## Usage

The script to send the trainings and the evaluation is in scripts/main.py. The way to pass the various setting is by
setting them as arguments when running the script; a throught explanation of all the parameters can be obtained by running:
```
python3 scripts/main.py --help
```

For example to launch a training in for edge cost regression:
```
--experiment_name example --world rand_gen_3 --metric success --use_poses --cached_model example_model --save_pred
```
### Results
The results will be stored in the ```digital_twin_learning/results/``` folder and it is suggested to explore them with Tensorboard by running:
```
tensorboard --logdir results/predictions
```
Furthermore a csv with all the useful data from the trainings is continously updated in ```digital_twin_learning/results/predictions/results.csv``` and provides a nice overview of all the numerical results  of the trainings, an extention to view it as a table is suggested. 

Finally the script in "```digital_twin_learning/digital_twin_learning/utils/visualize.py```" can be run autonomously and the global variables ```MESH_DIRECTORY, GRAPH_DIRECTORY1, GRAPH_DIRECTORY2``` have to be setted up with the graphs that want to be visualized

### Costum Datasets
In the folders ```results/cached_data``` and ```results/trained_models``` there will be stored the cached dataset and the parameters of trained networks, when running a training or an evaluation the code will always first search a corresponding dataset in these folders and only create a new one if it does not exist.

To run with costum datasets they need to be stored in the ```resources``` folder, for the edge cost regression there needs to be the pointcloud data (saved as ```pointcloud.pcd```), the GT graph and, optionally, the mesh (to be able to see the results properly). The name of the dataset must be consistent in all the three folders and must correspond to the one passed for the ```--world``` option.

*Alternatively, if you want to name differently the PCD and the GT graph, you can manually set their names in the dictionary in ```helpers.py```, line 312+.*

### Hyperparameters
The hyperparameters for the training can be changed in the Dataclass Hyperparameters in ```/digital_twin_learning/data/config.py```

### Sampling 
To run the sampling version of the repo, the flag ```--sample``` must be activated and there are a bunch of more options to be used  (such as the voxels sizes or overlaps). The encoder model must be pre-trained (for example with the edge cost regression repo) and its corresponding name provided as a parameter (```--encoder_model```).

To run an **evaluation** on a trained sampling model; firstly it needs to be run the ```--sample``` repo with the ```--evaluate_only``` flag on to generate a map with the various sampled poses; then, to evaluate the navigation graph and generate the results, the script must be run again with the argument ```--cached_dataset```, specifying the dataset generated previously. ***Not the argument ```--sample```, since we are now again doing edge regression and not sampling***.  
It should now be saved in ```digital_twin_learning/results/cached_data``` and should be named with the same name of the passed world and contain the **"learned"** world inside it  (plus some other stem passed with the argument "--stem"), for example ```--cached_dataset ETH_HPH_learned_mix_3```.  
Of course also a model for the edge cost regression must be passed inside ```--encoder_model```.


### Multiple trains
It is possible to run multiple training or evaluation runs consecutively, just create a script in the "scripts" folder and paste the following code:

```
import subprocess

# Set the script name
script_name = "scripts/main.py"

# Set the arguments
exp1 = ["--experiment_name", "mix3_success", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
        "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test"]

exp2 = ["--experiment_name", "mix3_success_2", "--world", "rand_gen_3", "--metric", "success", "--use_poses",
        "--cached_model", "mix3_success", "--save_pred", "--evaluate_only", "--jitter_test", "--max_dropout_ratio", "0.75",
        "--dropout_height", "0.1", "--max_c", "0.3", "--max_s", "0.3"]

exp249 = ["--experiment_name", "mix3_regression_train", "--world", "mix3", "--metric", "success",
          "--cached_model", "mix3_regression_train", "--save_pred", "--voxel_size_x", "2.0", "--voxel_size_y", "2.0", "--voxel_size_z", "2.0",
          "--num_samples", "5000",  "--num_epochs", "1"]

subprocess.run(["python", script_name, *exp1])
subprocess.run(["python", script_name, *exp2])
subprocess.run(["python", script_name, *exp3])

```