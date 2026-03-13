# Body Region Matching with High Density Point Clouds

## Problem Statement

Motion capture systems typically rely on markers to track the same body points across time. But in many real-world sensing systems such as LiDAR or dense 3D scans, these point identities are unknown. This makes it difficult to understand how a human body moves between frames.

Our project develops a method for matching human body point clouds across time **without relying on fixed point identities**. Using **optimal transport**, we compute correspondences between frames by treating point clouds as probability distributions.

To resolve common errors caused by body symmetry, we introduce a **4D embedding** that encodes a left–right body coordinate using automatically detected foot anchor points. This allows the algorithm to distinguish between symmetric structures such as the left and right legs.

The result is a more reliable way to track motion in **markerless 3D human data**, which could improve applications in biomechanics, robotics, animation, and motion analysis.

## File Structure:

```bash
datasets/
├── action_smplx_models/
│   ├── male2_Calibration_stageii.npz
│   ├── example_action_file1.npz
│   └── example_action_file2.npz
└── base_smplx_model/
    └── smplx/
        ├── md5sums.txt
        └── SMPLX_NEUTRAL.npz
```

## Data Acquisition:

Our data was acquired from the AMASS dataset website. First, you need to make an account to access their data. Then, from the downloads page, go to the ACCAD dataset and download the SMPLX-G data. Place the desired .npz files in the proper folder as shown above. The only required one is the male2_Calibration file. We used the following in our experiments:
* female_run_to_walk.npz
* female_run.npz
* female_walk_to_run.npz
* female_walk.npz
* male_run_to_walk.npz
* male_run.npz
* male_walk_to_run.npz
* male_walk_turn_left_90.npz
* male2_Calibration_stageii.npz

You also need the base SMPLX model, which can be obtained [here](https://smpl-x.is.tue.mpg.de/download.php) on the button labeled "Download SMPL-X with removed head bun (NPZ, 392MB) - Use this for the SMPL-X Python codebase and AMASS data". We used the SMPLX_NEUTRAL.npz file.

Experiments were ran using the experiment.py file, although we don't recommend running it yourself as it can take a while. To get a sense of how they work, it would make sense to run the experiments with less poses sampled and less hyperparameters to search. Figures for the report were generated using figures.py.

## Environment Set-up

Our environment was created using **Miniforge / Conda** with dependencies installed from `requirements.txt`.

```bash
# 1. create environment (adjust python version if needed)
conda create -n amass-env python=3.12 -y

# 2. activate environment
conda activate amass-env

# 3. install dependencies
pip install -r requirements.txt
```

These are our versions:
```
numpy==1.26.4
pandas==2.1.1
matplotlib==3.8.1
plotly==6.1.2
scipy==1.11.3
scikit-learn==1.3.1
POT==0.9.1
smplx==0.1.28
trimesh==4.11.0
requests==2.30.0
beautifulsoup4==4.12.2
nbformat==5.10.4
```

Optional verification:

```bash
python --version
pip list
```

Deactivate when finished:

```bash
conda deactivate
```

## Running Experiments
Our file for running experiments is called *experiments.py*. Simply run that file with Python in the terminal. It will print progress occasionally and should result in plots resembling the ones in our poster/report (although with the randomness it won't be the exact same). In that file are a number of functions, each of which runs a different experiment. The purpose of each is as follows:
* left_right_augmentation - **This is the main one** used for the results in the paper and on the poster. It compares a baseline of optimal transport on the raw data to our method of optimal transport on augmented 4d data.
* run_test - Archaic experiment using a method that we were investigating but didn't end up using.
* acc_by_delta_per_alpha - Experiment investigating combining graph distance and Euclidean distance. Did not end up in final report.
* naive_vs_novel_comparison - Similar to the previous one. Did not end up in final report.
* novel_component_comparison - Comparison of combinations of methods. Did not end up in final report
* fused_gromov_experiment_by_alpha - Experiment to see if fused gromov wasserstein with different alpha values would outperform the baseline. Did not seem to help; did not end up in final report.
* left_right_augmentation_relative - Same experiment as the main one but with slightly different plots. Instead of averaging each method separately, we first compare performance and then average the relative performance. Didn't reveal much new information; did not end up in final report.
* left_right_augmented_fgw - Another FGW experiment. Did not end up in final report.

The experiments take a *num_poses* and a *timedeltas* argument. The first one is the number of random initial point clouds. The timedeltas is an iterable of future values. So if *num_poses* is 2 then two initial point clouds will be sampled across all the actions. Then if *timedeltas* is [10, 20, 30] then those 2 initial frames will be matched with the frames 10, 20, and 30 frames into the future. So a total of 8 point clouds would be sampled in total. For the graphics in the poster and report, *num_poses* was set to 20 and *timedeltas* was set to *np.arange(20, 91, 10)*.