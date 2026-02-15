import numpy as np
import utils
import ot
import time
import plotly.io as pio

def region_matching_example():
    run_path = "datasets/action_smplx_models/male_run.npz"
    N = 1_000
    points1, faces1 = utils.sampled_verts_from_path(run_path, idx = 0, return_faces=True, n_points=N)
    points2, faces2 = utils.sampled_verts_from_path(run_path, idx = 10, return_faces=True, n_points=N)
    a = np.ones(N)/N
    b = np.ones(N)/N
    M = ot.dist(points1, points2)
    G = ot.solve(M, a, b).plan
    return utils.plot_3d_points_and_connections_region_matched(points1, points2, faces1, faces2, G, plot_both=False)

if __name__ == "__main__":
    pio.renderers.default = "browser"
    fig1 = utils.plot_random_pose(n_points=1000)
    fig1.show()

    # fig2 = region_matching_example()
    # fig2.show()
    
    