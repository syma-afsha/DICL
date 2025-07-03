
This repository contains the official implementation of our methods presented in the **"DICL: Dual-Mechanism Incremental Curriculum Learning for Advancing Deep Reinforcement Learning Agents"**.  

##  Overview

This research work introduces three novel curricululum learning strategies:

- **Incremental Graded Curriculum (IGC)**: Progressively expanding task difficulty through joint state-goal distribution modulation.
- **Dual Buffer Mechanism (DB)**: Structured experience replay using positive, negative, and standard buffers to encourage targeted learning.
- **Dual-Mechanism Incremental Curriculum Learning (DICL)**: A unified framework that integrates IGC and DB into a single pipeline.

Curriculum learning strategies are evaluated on three sparse-reward robotic manipulation tasks from the Gymnasium-Robotics suite:

- `FetchPickAndPlace`  
- `FetchPush`  
- `FetchSlide`

---

##  Installation

This repository was developed and tested on **Ubuntu 20.04** with **Python 3.8** and **CUDA 12.1**.  
Nevertheless, the code should work on other configurations as well. Since the required packages are relatively common, installation should not pose major issues. The following guide assumes a fresh setup.

###  Clone the Repository

```bash
git clone https://github.com/syma-afsha/DICL
cd DICL
```

###  Virtual Environment Setup

```bash
python3.8 -m venv venv
source venv/bin/activate
```

---

##  Gymnasium Fetch Environment Setup

To run the robotic manipulation tasks, install the Gymnasium environments:

### Step 1: Clone the Gymnasium repository

```bash
git clone https://hub.fastgit.org/Farama-Foundation/Gymnasium.git
cd Gymnasium
pip install -e .
```

### Step 2: Install Gymnasium Robotics

```bash
pip install gymnasium[robotics]
```

Ensure that **MuJoCo** is properly installed and licensed. Refer to the official documentation:  
 https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/robotics

---

##  Required Dependencies

Install the required packages using:

```bash
pip install -r requirements.txt
```

If the file is missing, create it manually with:

```txt
torch==2.1.2+cu121
torchvision==0.16.2+cu121
gymnasium==0.29.1
mujoco==3.1.3
mujoco-py==2.1.2.14
numpy
matplotlib
tqdm
seaborn
scikit-learn
omegaconf
tensorboard
```

For CUDA/torch compatibility, visit: https://pytorch.org/get-started/locally/

---

##  Repository Structure

```
.
├── agents/             # SAC Agent
├── buffer/             # PER, HER, and DB implementations
├── networks/           # Policy and critic network architectures
├── config/             # YAML experiment configs
├── evaluation/         # Evaluation script
├── logger/             # Saved logs and models
├── reward/             # Penalty reward function
├── CL/                 # IGC curriculum learning strategies
├── taskenv/            # Custom wrappers or configuration for Gymnasium Fetch environments
├── results/            # Plots, evaluation tables, and trained results
└── README.md
```

---

##  Config Customization

You can manually edit the experiment, environment, and other parameters—such as enabling HER, PER, and others—by modifying the YAML configuration files located in the config/ directory.

### Common Customizations:

```yaml
environment:
  name: "FetchPickAndPlace-v4"  # Choose from FetchPush-v4, FetchSlide-v4, etc.

general:
  exp_name: "my_experiment"     # Set a custom experiment name
```

---

##  Run the Code

### To run only the Incremental Graded Curriculum (IGC):

```bash
python igc_main.py
```

### To run only the Dual Buffer (DB) mechanism:

```bash
python db.py
```

### To run the integrated DICL framework:

```bash
python dicl.py
```

---


