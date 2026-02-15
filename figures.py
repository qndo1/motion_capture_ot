import numpy as np
import matplotlib.pyplot as plt
import utils
import scipy
import ot

if __name__ == "__main__":
    p = "datasets/action_smplx_models/male_run.npz"
    points = utils.sampled_verts_from_path(p, idx = 0, n_points = 1000)
    fig = utils.plot_arr(points)
    fig.show()

    p = "datasets/action_smplx_models/male2_Calibration_stageii.npz"
    points, faces = utils.sampled_verts_from_path(p, n_points = 10000, return_faces = True)
    regions, colors = utils.faces_to_regions(faces, return_colors = True)
    fig = utils.plot_arr(points, color = colors)
    fig.show()

    p = "datasets/action_smplx_models/male_run.npz"
    points1, faces1 = utils.sampled_verts_from_path(p, idx = 0, return_faces = True)
    points2, faces2 = utils.sampled_verts_from_path(p, idx = 60, return_faces = True)
    a = np.ones(1000) / np.ones(1000)
    b = np.ones(1000) / np.ones(1000)
    M = ot.dist(points1, points2)
    G = ot.solve(M, a, b).plan
    fig = utils.plot_specific_region_connections(points1, points2, faces1, faces2, G, "left_shin")
    fig.show()

    mask = points1[:,2] < np.percentile(points1[:, 2], 10)
    colors = ["green" if x else "red" for x in mask]
    fig = utils.plot_arr(points1, color = colors)
    fig.show()

    means = scipy.cluster.vq.kmeans(points1[mask][:,:2], 2)[0]
    center_index_1 = np.argmin(np.linalg.norm(points1[mask][:,:2] - means[0], axis = 1))
    center_index_2 = np.argmin(np.linalg.norm(points1[mask][:,:2] - means[1], axis = 1))
    plt.scatter(points1[mask][:,0], points1[mask][:,1], label = "Projected feet points")
    plt.scatter(means[:,0], means[:,1], label = "K-means centroids")
    plt.scatter(points1[mask][[center_index_1, center_index_2]][:,0], points1[mask][[center_index_1, center_index_2]][:,1], label = "Closest real points to centroids")
    plt.legend()
    plt.show()

    ind = np.random.choice(range(1000))
    print(ind)
    C = utils.graph_distance_within_cloud_minimally_connected(points1)
    fig = utils.plot_arr(points1, color = C[ind], colorbar = True)
    fig.show()

    aug1 = utils.left_right_augmentation(points1, points2.mean(axis = 0) - points1.mean(axis = 0))
    fig = utils.plot_arr(points1, color = aug1[:,3], colorbar = True)
    fig.show()