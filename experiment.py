import utils
import numpy as np
import ot
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def graph_distance_between_clouds_minimally_connected(P, Q, k=3):
    """
    Construct a minimally connected kNN graph on two point clouds
    and compute graph distances.

    Parameters
    ----------
    P, Q : (N, 3) numpy arrays
        Two point clouds of equal size
    k : int
        Number of nearest neighbors (default: 3)

    Returns
    -------
    D : (N, N) numpy array
        Graph distance matrix from P to Q
    """
    P = np.asarray(P)
    Q = np.asarray(Q)
    assert P.shape == Q.shape

    N = P.shape[0]
    X = np.vstack([P, Q])          # (2N, 3)
    M = 2 * N

    # --- Step 1: initial kNN graph ---
    tree = cKDTree(X)
    dists, nbrs = tree.query(X, k=k + 1)

    edges = {}  # (i, j) -> weight
    for i in range(M):
        for j, dist in zip(nbrs[i][1:], dists[i][1:]):
            a, b = sorted((i, j))
            edges[(a, b)] = dist

    # --- Step 2: minimally connect components ---
    while True:
        rows, cols, weights = [], [], []
        for (i, j), w in edges.items():
            rows += [i, j]
            cols += [j, i]
            weights += [w, w]

        A = coo_matrix((weights, (rows, cols)), shape=(M, M))
        n_components, labels = connected_components(A, directed=False)

        if n_components == 1:
            break

        # pick one component
        c0 = labels[0]
        idx_c0 = np.where(labels == c0)[0]
        idx_rest = np.where(labels != c0)[0]

        # find closest inter-component pair
        P0 = X[idx_c0]
        P1 = X[idx_rest]

        dmat = np.linalg.norm(P0[:, None, :] - P1[None, :, :], axis=2)
        i_min, j_min = np.unravel_index(np.argmin(dmat), dmat.shape)

        u = idx_c0[i_min]
        v = idx_rest[j_min]
        w = dmat[i_min, j_min]

        a, b = sorted((u, v))
        edges[(a, b)] = w

    # --- Step 3: shortest-path distances ---
    rows, cols, weights = [], [], []
    for (i, j), w in edges.items():
        rows += [i, j]
        cols += [j, i]
        weights += [w, w]

    A = coo_matrix((weights, (rows, cols)), shape=(M, M))

    # distances from P nodes (0..N-1) to all nodes
    dist_full = dijkstra(A, directed=False, indices=np.arange(N))

    # extract P -> Q distances
    D = dist_full[:, N:]

    return D


def run_test(num_poses = 10, stepsize = 0.1, timedelta = 10):
    for i in range(num_poses):
        datasets_path = "datasets/action_smplx_models/"
        num_poses = os.listdir(datasets_path)
        p = datasets_path + np.random.choice(num_poses)
        N = 1000
        num_frames = int(np.load(p)["mocap_time_length"] * 120)
        rand_idx = np.random.randint(0, num_frames - timedelta - 1)
        points1, faces1 = utils.sampled_verts_from_path(p, idx=rand_idx, n_points=N, return_faces=True)
        points2, faces2 = utils.sampled_verts_from_path(p, idx=rand_idx + timedelta, n_points=N, return_faces=True)
        print("points loaded")
        #D = graph_distance_between_clouds_connected(points1, points2)[0]
        D = graph_distance_between_clouds_minimally_connected(points1 - points1.mean(axis = 0), points2 - points2.mean(axis = 0), k = 3)
        print("graph distance calculated")
        a = np.ones(N) / N
        b = np.ones(N) / N
        M = ot.dist(points1, points2)
        print("euclidean distance calculated")
        alphas = np.arange(0, 1 + stepsize, stepsize)
        region_accs = []
        av_reg_dists = []
        for alpha in alphas:
            interp_dist = alpha * M  + (1 - alpha) * D 
            this_G = ot.solve(interp_dist, a, b).plan
            region_accs.append(utils.region_accuracy(this_G, faces1, faces2))
            av_reg_dists.append(utils.average_region_distance(this_G, faces1,faces2))
            print("alpha", alpha, "done")
        if i == 0:
            plt.plot(alphas, region_accs, label = "Region Accuracy", color = "blue")
            plt.plot(alphas, av_reg_dists, label = "Average Region Distance", color = "orange")
        else:
            plt.plot(alphas, region_accs, color = "blue")
            plt.plot(alphas, av_reg_dists, color = "orange")
        print(i, "done")
    plt.legend()
    plt.ylim(0,1)
    return plt

def acc_by_delta_per_alpha(num_poses = 10, stepsize = 0.01, timedeltas = np.arange(0, 121, 10), n_points = 1000):
    fig, ax = plt.subplots(ncols=2, figsize = (20, 10))

    datasets_path = "datasets/action_smplx_models/"
    poses = os.listdir(datasets_path)
    pose_paths = np.random.choice(poses, size = num_poses, replace = True)
    random_indices = []
    for i in range(num_poses):
        num_frames = int(np.load(datasets_path + pose_paths[i])["mocap_time_length"] * 120)
        rand_idx = np.random.randint(0, num_frames - timedeltas[-1] - 1)
        random_indices.append(rand_idx)

    print("indices chosen")

    points_dict = {}
    for i in range(num_poses):
        for td in timedeltas:
            points_dict[(i, td)] = utils.sampled_verts_from_path(datasets_path + pose_paths[i], idx = random_indices[i] + td, n_points = n_points, return_faces=True)

    print("points sampled")

    for alpha in np.arange(0, 1 + stepsize, stepsize):
        av_region_accs = []
        av_av_dists = []
        for td in timedeltas[1:]:
            these_region_accs = []
            these_av_dists = []
            for i in range(num_poses):
                points1, faces1 = points_dict[(i, 0)]
                points2, faces2 = points_dict[(i, td)]
                D = graph_distance_between_clouds_minimally_connected(points1 - points1.mean(axis = 0), points2 - points2.mean(axis = 0), k = 3)
                a = np.ones(n_points) / n_points
                b = np.ones(n_points) / n_points
                M = ot.dist(points1, points2)
                interp_dist = alpha * M  + (1 - alpha) * D 
                this_G = ot.solve(interp_dist, a, b).plan
                these_region_accs.append(utils.region_accuracy(this_G, faces1, faces2))
                these_av_dists.append(utils.average_region_distance(this_G, faces1, faces2))
            av_region_accs.append(np.mean(these_region_accs))
            av_av_dists.append(np.mean(these_av_dists))
        ax[0].plot(timedeltas[1:], av_region_accs, color = cm.get_cmap("viridis")(alpha), label = alpha)
        ax[1].plot(timedeltas[1:], av_av_dists, color = cm.get_cmap("viridis")(alpha), label = alpha)
        print(alpha, "plotted")

    ax[0].legend()
    ax[0].set_title(f"Average Region Accuracies Over {num_poses} Random Poses")
    ax[0].set_ylim(0, 1)
    ax[1].legend()
    ax[1].set_title(f"Average Average Distances Over {num_poses} Random Poses")
    ax[0].set_ylim(0, 1)
    return ax

def naive_vs_novel_comparison(num_poses = 10, timedeltas = np.arange(0, 121, 10), n_points = 1000):
    if not (0 in timedeltas):
        timedeltas = np.append(0, np.arange(0, 81, 1))
    fig, ax = plt.subplots(ncols=2, figsize = (20, 10))

    datasets_path = "datasets/action_smplx_models/"
    poses = os.listdir(datasets_path)
    pose_paths = np.random.choice(poses, size = num_poses, replace = True)
    random_indices = []
    for i in range(num_poses):
        num_frames = int(np.load(datasets_path + pose_paths[i])["mocap_time_length"] * 120)
        rand_idx = np.random.randint(0, num_frames - timedeltas[-1] - 1)
        random_indices.append(rand_idx)

    print("indices chosen")

    points_dict = {}
    normals_dict = {}
    lifted_points_dict = {}
    for i in range(num_poses):
        for td in timedeltas:
            points_dict[(i, td)] = utils.sampled_verts_from_path(datasets_path + pose_paths[i], idx = random_indices[i] + td, n_points = n_points, return_faces=True)
            normals_dict[(i, td)] = utils.pca_symmetry_plane(points_dict[(i, td)][0])
            lifted_points_dict[(i, td)] = utils.lift_with_symmetry(points_dict[(i, td)][0], normals_dict[(i, td)])

    print("points sampled")

    av_region_accs_adj_naive = []
    av_av_dists_naive = []

    av_region_accs_adj_novel = []
    av_av_dists_novel = []

    for td in timedeltas[1:]:
        these_region_accs_naive = []
        these_av_dists_naive = []

        these_region_accs_novel = []
        these_av_dists_novel = []

        for i in range(num_poses):
            points1, faces1 = points_dict[(i, 0)]
            points2, faces2 = points_dict[(i, td)]

            normals1 = normals_dict[(i, 0)]
            normals2 = normals_dict[(i, td)]

            lifted_points1 = lifted_points_dict[(i, 0)]
            lifted_points2 = lifted_points_dict[(i, td)]

            mean_centered1 = lifted_points1 - lifted_points1.mean(axis = 0)
            mean_centered2 = lifted_points2 - lifted_points2.mean(axis = 0)

            D = graph_distance_between_clouds_minimally_connected(mean_centered1[:,:-1], mean_centered2[:,:-1], k = 3)
            a = np.ones(n_points) / n_points
            b = np.ones(n_points) / n_points
            M_naive = ot.dist(points1, points2)
            M_novel = ot.dist(lifted_points1, lifted_points2)

            this_G_naive = ot.solve(M_naive, a, b).plan
            this_G_novel = ot.solve(M_novel + D, a, b).plan

            these_region_accs_naive.append(utils.region_accuracy_adjusted(this_G_naive, faces1, faces2))
            these_av_dists_naive.append(utils.average_region_distance(this_G_naive, faces1, faces2))

            these_region_accs_novel.append(utils.region_accuracy_adjusted(this_G_novel, faces1, faces2))
            these_av_dists_novel.append(utils.average_region_distance(this_G_novel, faces1, faces2))

        av_region_accs_adj_naive.append(np.mean(these_region_accs_naive))
        av_av_dists_naive.append(np.mean(these_av_dists_naive))

        av_region_accs_adj_novel.append(np.mean(these_region_accs_novel))
        av_av_dists_novel.append(np.mean(these_av_dists_novel))
        print("timedelta", td, "done")

    ax[0].plot(timedeltas[1:], av_region_accs_adj_naive, color = "red", label = "Naive Matching")
    ax[0].plot(timedeltas[1:], av_region_accs_adj_novel, color = "blue", label = "Novel Matching")

    ax[1].plot(timedeltas[1:], av_av_dists_naive, color = "red", label = "Naive Matching")
    ax[1].plot(timedeltas[1:], av_av_dists_novel, color = "blue", label = "Novel Matching")


    ax[0].legend()
    ax[0].set_title(f"Average Region Adjusted Accuracies Over {num_poses} Random Poses")
    ax[1].legend()
    ax[1].set_title(f"Average Average Distances Over {num_poses} Random Poses")
    ax[0].set_xlabel("time delta")
    ax[1].set_xlabel("time delta")
    return ax

def novel_component_comparison(num_poses = 10, timedeltas = np.arange(0, 121, 10), n_points = 1000):
    if not (0 in timedeltas):
        timedeltas = np.append(0, np.arange(0, 81, 1))
    fig, ax = plt.subplots(ncols=2, figsize = (20, 10))

    datasets_path = "datasets/action_smplx_models/"
    poses = os.listdir(datasets_path)
    pose_paths = np.random.choice(poses, size = num_poses, replace = True)
    random_indices = []
    for i in range(num_poses):
        num_frames = int(np.load(datasets_path + pose_paths[i])["mocap_time_length"] * 120)
        rand_idx = np.random.randint(0, num_frames - timedeltas[-1] - 1)
        random_indices.append(rand_idx)

    print("indices chosen")

    points_dict = {}
    normals_dict = {}
    lifted_points_dict = {}
    for i in range(num_poses):
        for td in timedeltas:
            points_dict[(i, td)] = utils.sampled_verts_from_path(datasets_path + pose_paths[i], idx = random_indices[i] + td, n_points = n_points, return_faces=True)
            normals_dict[(i, td)] = utils.pca_symmetry_plane(points_dict[(i, td)][0])
            lifted_points_dict[(i, td)] = utils.lift_with_symmetry(points_dict[(i, td)][0], normals_dict[(i, td)])

    print("points sampled")


    for label in [
        "Euclidean Only",
        "Euclidean + Graph",
        "Augmented Only",
        "Augmented + Graph",
        "Graph Only"
    ]:
        av_region_accs = []
        av_av_dists = []
        for td in timedeltas[1:]:
            these_region_accs = []
            these_av_dists = []
            for i in range(num_poses):
                points1, faces1 = points_dict[(i, 0)]
                points2, faces2 = points_dict[(i, td)]

                normals1 = normals_dict[(i, 0)]
                normals2 = normals_dict[(i, td)]

                lifted_points1 = lifted_points_dict[(i, 0)]
                lifted_points2 = lifted_points_dict[(i, td)]

                mean_centered1 = lifted_points1 - lifted_points1.mean(axis = 0)
                mean_centered2 = lifted_points2 - lifted_points2.mean(axis = 0)

                if "Graph" in label:
                    D = graph_distance_between_clouds_minimally_connected(mean_centered1[:,:-1], mean_centered2[:,:-1], k = 3)
                else:
                    D = np.zeros((n_points, n_points))

                a = np.ones(n_points) / n_points
                b = np.ones(n_points) / n_points

                if "Augmented" in label:
                    M = ot.dist(lifted_points1, lifted_points2)
                elif "Euclidean" in label:
                    M = ot.dist(points1, points2)
                else:
                    M = np.zeros((n_points, n_points))
                this_G = ot.solve(M + D, a, b).plan
                these_region_accs.append(utils.region_accuracy(this_G, faces1, faces2))
                these_av_dists.append(utils.average_region_distance(this_G, faces1, faces2))
            av_region_accs.append(np.mean(these_region_accs))
            av_av_dists.append(np.mean(these_av_dists))
        ax[0].plot(timedeltas[1:], av_region_accs, label = label)
        ax[1].plot(timedeltas[1:], av_av_dists, label = label)
        print(label, "plotted")

    ax[0].legend()
    ax[0].set_title(f"Average Region Adjusted Accuracies Over {num_poses} Random Poses")
    ax[1].legend()
    ax[1].set_title(f"Average Average Distances Over {num_poses} Random Poses")
    ax[0].set_xlabel("time delta")
    ax[1].set_xlabel("time delta")
    return ax

def fused_gromov_experiment_by_alpha(num_poses= 10, timedeltas= np.arange(0, 121, 5), n_points = 1000, beta = 1):
    if not (0 in timedeltas):
        timedeltas = np.append(0, np.arange(0, 81, 1))
    fig, ax = plt.subplots(ncols=2, figsize = (20, 10))

    datasets_path = "datasets/action_smplx_models/"
    poses = os.listdir(datasets_path)
    pose_paths = np.random.choice(poses, size = num_poses, replace = True)
    random_indices = []
    for i in range(num_poses):
        num_frames = int(np.load(datasets_path + pose_paths[i])["mocap_time_length"] * 120)
        rand_idx = np.random.randint(0, num_frames - timedeltas[-1] - 1)
        random_indices.append(rand_idx)

    print("indices chosen")

    points_dict = {}
    faces_dict = {}
    head_normals_dict = {}
    Cs_dict = {}
    pca_normals_dict = {}
    for i in range(num_poses):
        for td in timedeltas:
            points, faces = utils.sampled_verts_from_path(datasets_path + pose_paths[i], idx = random_indices[i] + td, n_points = n_points, return_faces=True)
            points_dict[(i, td)] = points
            faces_dict[(i, td)] = faces
            head_mean = points[
                (points[:,2] > np.percentile(points[:,2], 92.5) )
                & (np.linalg.norm((points - points.mean(axis = 0))[:,:-1], axis = 1) < .2)
            ].mean(axis = 0)
            diff = head_mean - np.median(points, axis = 0) / np.linalg.norm(head_mean - np.median(points, axis = 0))
            head_normal = np.array([diff[0], diff[1], 0])
            head_normals_dict[(i, td)] = head_normal / np.linalg.norm(head_normal)
            pca_normals_dict[(i, td)] = utils.pca_symmetry_plane(points)

            Cs_dict[(i, td)] = utils.graph_distance_within_cloud_minimally_connected(points)
    
    print("points sampled")

    for label in ["Unaugmented", "Head method", "Baseline"]:
        adj_region_accs = []
        av_dist_accs = []
        for alpha in np.arange(0, 1.01, 0.1):
            these_accs = []
            these_av_dists = []
            for td in timedeltas[1:]:
                for i in range(num_poses):
                    points1 = points_dict[(i, 0)]
                    faces1 = faces_dict[(i, 0)]
                    points2 = points_dict[(i, td)]
                    faces2 = faces_dict[(i, td)]

                    if label == "Unaugmented" or label == "Baseline":
                        M = ot.dist(points1, points2)
                    else:
                        if label == "Head method":
                            normals1 = head_normals_dict[(i, 0)]
                            normals2 = head_normals_dict[(i, td)]
                        elif label == "PCA method":
                            normals1 = pca_normals_dict[(i, 0)]
                            normals2 = pca_normals_dict[(i, td)]
                        elif label == "Difference method":
                            normals1 = points2.mean(axis = 0) - points1.mean(axis = 0)
                            normals1 = np.array([normals1[0], normals1[1], 0])
                            normals1 = normals1 / np.linalg.norm(normals1)

                        dists1 = (points1 - points1.mean(axis = 0)) @ (np.cross(normals1, np.array([0,0,1])))
                        dists2 = (points2 - points2.mean(axis = 0)) @ (np.cross(normals2, np.array([0,0,1])))

                        aug1 = np.append(points1, (dists1 * beta).reshape(-1, 1), axis = 1)
                        aug2 = np.append(points2, (dists2 * beta).reshape(-1, 1), axis = 1)

                        M = ot.dist(aug1, aug2)

                    C1 = Cs_dict[(i, 0)]
                    C2 = Cs_dict[(i, td)]

                    if label != "Baseline":
                        G = ot.fused_gromov_wasserstein(M, C1, C2, alpha = alpha)
                    else:
                        G = ot.solve(M, np.ones(n_points) / n_points, np.ones(n_points) / n_points).plan

                    these_accs.append(utils.region_accuracy_adjusted(G, faces1, faces2))
                    these_av_dists.append(utils.average_region_distance(G, faces1, faces2))

            adj_region_accs.append(np.mean(these_accs))
            av_dist_accs.append(np.mean(these_av_dists))
            print(alpha, "done")
            
        ax[0].plot(np.arange(0, 1.01, 0.1), adj_region_accs, label = label)
        ax[1].plot(np.arange(0, 1.01, 0.1), av_dist_accs, label = label)
        print(label, "done")

    ax[0].legend()
    ax[0].set_title(f"Average Region Adjusted Accuracies")
    ax[1].legend()
    ax[1].set_title(f"Average Average Distances")
    ax[0].set_xlabel("alpha")
    ax[1].set_xlabel("alpha")
    return ax

def left_right_augmentation(num_poses=10, timedeltas=np.arange(0, 121, 10), n_points = 1000):
    if not (0 in timedeltas):
        timedeltas = np.append(0, np.arange(0, 81, 1))
    fig, ax = plt.subplots(ncols=2, figsize = (20, 10))

    datasets_path = "datasets/action_smplx_models/"
    poses = [p for p in os.listdir(datasets_path) if p != "male2_Calibration_stagegii.npz"]
    pose_paths = np.random.choice(poses, size = num_poses, replace = True)
    random_indices = []
    for i in range(num_poses):
        num_frames = int(np.load(datasets_path + pose_paths[i])["mocap_time_length"] * 120)
        rand_idx = np.random.randint(0, num_frames - timedeltas[-1] - 1)
        random_indices.append(rand_idx)

    print("indices chosen")

    points_dict = {}
    faces_dict = {}
    for i in range(num_poses):
        for td in timedeltas:
            points, faces =  utils.sampled_verts_from_path(datasets_path + pose_paths[i], idx = random_indices[i] + td, n_points = n_points, return_faces=True)
            points_dict[(i, td)] = points
            faces_dict[(i, td)] = faces
    
    print("points sampled")

    baseline_reg_accs = []
    baseline_reg_dists = []

    aug_reg_accs = []
    aug_reg_dists = []

    a = np.ones(n_points) / n_points
    b = np.ones(n_points) / n_points

    for td in timedeltas[1:]:
        these_baseline_reg_accs = []
        these_baseline_reg_dists = []

        these_aug_reg_accs = []
        these_aug_reg_dists = []
        for i in range(num_poses):
            points1 = points_dict[(i, 0)]
            faces1 = faces_dict[(i, 0)]
            
            points2 = points_dict[(i, td)]
            faces2 = faces_dict[(i, td)]

            diff = points2.mean(axis = 0) - points1.mean(axis = 0)

            aug1 = utils.left_right_augmentation(points1, diff)
            aug2 = utils.left_right_augmentation(points2, diff)

            M_baseline = ot.dist(points1, points2)
            M_aug = ot.dist(aug1, aug2)

            G_baseline = ot.solve(M_baseline, a, b).plan
            G_aug = ot.solve(M_aug, a, b).plan

            these_baseline_reg_accs.append(utils.region_accuracy_adjusted(G_baseline, faces1, faces2))
            these_baseline_reg_dists.append(utils.average_region_distance(G_baseline, faces1, faces2))

            these_aug_reg_accs.append(utils.region_accuracy_adjusted(G_aug, faces1, faces2))
            these_aug_reg_dists.append(utils.average_region_distance(G_aug, faces1, faces2))

        baseline_reg_accs.append(np.mean(these_baseline_reg_accs))
        baseline_reg_dists.append(np.mean(these_baseline_reg_dists))

        aug_reg_accs.append(np.mean(these_aug_reg_accs))
        aug_reg_dists.append(np.mean(these_aug_reg_dists))
        print("td", td, "done")

    ax[0].plot(timedeltas[1:], baseline_reg_accs, label = "Baseline")
    ax[0].plot(timedeltas[1:], aug_reg_accs, label = "Augmented")
    ax[0].set_xlabel("Frame delta")
    ax[0].set_ylabel("Adjusted Region Accuracy")
    ax[0].set_title("Accuracy by Frame Delta\nBaseline vs Augmented")
    ax[0].legend()

    ax[1].plot(timedeltas[1:], baseline_reg_dists, label = "Baseline")
    ax[1].plot(timedeltas[1:], aug_reg_dists, label = "Augmented")
    ax[1].set_xlabel("Frame delta")
    ax[1].set_ylabel("Average Region Distance")
    ax[1].set_title("Average Region Distance by Frame Delta\nBaseline vs Augmented")
    ax[1].legend()

    return ax

    


if __name__ == "__main__":
    # plt = run_test(num_poses = 10, stepsize = 0.01, timedelta = 10)
    # plt.show()
    # ax = acc_by_delta_per_alpha(num_poses = 10, stepsize = 0.05, timedeltas = np.arange(0, 121, 10))
    # plt.show()
    # ax = novel_component_comparison(num_poses= 10, timedeltas= np.arange(0, 121, 5))
    # plt.show()
    #ax = fused_gromov_experiment_by_alpha(num_poses=10, timedeltas=np.arange(0, 121, 10))
    ax = left_right_augmentation(num_poses= 10, timedeltas = np.arange(0, 121, 10))
    plt.show()