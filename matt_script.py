import numpy as np
import smplx
import trimesh
import torch
import os
import utils

if __name__ == "__main__":
    fig = utils.plot_random_pose(n_points=5_000)
    fig.show()