# Body Region Matching with High Density Point Clouds

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