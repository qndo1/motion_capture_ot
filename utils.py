import numpy as np
import smplx
import trimesh
import torch
import os
import plotly.graph_objects as go
import pandas as pd
import json

def smplx_vertices_from_amass(
    smplx_model_path,
    amass_npz_path,
    gender="neutral",
    device="cpu",
):
    """
    Returns:
        verts: (T, V, 3) tensor of vertex positions
        faces: (F, 3) numpy array of face indices
    """

    device = torch.device(device)

    # -------------------------
    # Load AMASS motion data
    # -------------------------
    data = np.load(amass_npz_path)

    # Required AMASS keys (most files have these)
    poses = data["poses"]          # (T, 165) for SMPL-X
    betas = data["betas"]          # (10,) or (16,)
    trans = data.get("trans", None)  # (T, 3) optional

    T = poses.shape[0]

    # Split pose vector
    # SMPL-X pose layout:
    # [global_orient (3),
    #  body_pose (63),
    #  jaw (3),
    #  leye (3),
    #  reye (3),
    #  left_hand (45),
    #  right_hand (45)]
    poses = torch.tensor(poses, dtype=torch.float32, device=device)

    global_orient = poses[:, :3]
    body_pose = poses[:, 3:66]
    jaw_pose = poses[:, 66:69]
    leye_pose = poses[:, 69:72]
    reye_pose = poses[:, 72:75]
    left_hand_pose = poses[:, 75:120]
    right_hand_pose = poses[:, 120:165]

    betas = torch.tensor(betas[:10], dtype=torch.float32, device=device)
    betas = betas.unsqueeze(0).expand(T, -1)

    if trans is not None:
        transl = torch.tensor(trans, dtype=torch.float32, device=device)
    else:
        transl = torch.zeros((T, 3), device=device)

    # -------------------------
    # Load SMPL-X model
    # -------------------------
    model = smplx.create(
        model_path=smplx_model_path,
        model_type="smplx",
        gender=gender,
        ext="npz",
        use_pca=False,
        batch_size=T,
    ).to(device)

    # -------------------------
    # Forward pass
    # -------------------------
    output = model(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        transl=transl,
    )

    verts = output.vertices  # (T, V, 3)
    faces = model.faces     # (F, 3)

    return verts, faces

def true_verts_from_path(amass_npz_path, return_faces = False):
    smplx_model_path = "datasets/base_smplx_model"

    verts, faces = smplx_vertices_from_amass(
        smplx_model_path,
        amass_npz_path,
        gender="neutral",
    )

    if return_faces:    
        return verts, faces
    else:
        return verts

def vertices_to_pointcloud(vertices, faces, n_points=1_000, return_mesh = False, return_faces = False):
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False
    )
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)

    output = [points]
    if return_mesh:
        output.append(mesh)
    if return_faces:
        output.append(face_indices)
    
    if len(output) == 1:
        return output[0]
    return tuple(output)
    
def mesh_from_path(path, idx = 0):
    verts, faces = true_verts_from_path(path, return_faces = True)

    verts = verts.detach().numpy()
    
    mesh = trimesh.Trimesh(
        vertices = verts[idx],
        faces = faces,
        process = False
    )

    return mesh

def meshes_from_path(path):
    verts, faces = true_verts_from_path(path, return_faces = True)

    verts = verts.detach().numpy()

    meshes = []

    for i in range(verts.shape[0]):
        meshes.append(trimesh.Trimesh(
            vertices = verts[i],
            faces = faces,
            process = False
        ))
    
    return meshes

def sampled_verts_from_mesh(mesh, n_points = 1_000, return_faces = False):
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    
    output = [points]
    if return_faces:
        output.append(face_indices)
    return tuple(output)

def sampled_verts_from_path(amass_npz_path, idx = None, n_points = 1_000, return_mesh = False, return_faces = False):
    
    verts, faces = true_verts_from_path(amass_npz_path, return_faces = True)
    verts = verts.detach().numpy()

    if idx is None or idx >= verts.shape[0]:
        idx = np.random.randint(0, verts.shape[0])
        print(f"Randomly chosen index: {idx}")

    return vertices_to_pointcloud(verts[idx], faces, n_points = n_points, return_mesh = return_mesh, return_faces=return_faces)

def choose_random_file(path = "datasets/action_smplx_models"):
    
    files = os.listdir(path)
    file = np.random.choice(files)

    print(f"Randomly chosen file: {file}")

    return path + "/" + file

def sampled_verts_from_random_action(n_points = 1_000, return_mesh = False, return_faces = False):
    file = choose_random_file()

    return sampled_verts_from_path(file, n_points = n_points, return_mesh = return_mesh, return_faces=return_faces)

def plot_arr(arr, color = None, colorbar = False, size = None):
    fig = go.Figure()

    this_frame = arr
    x = this_frame[:,0]
    y = this_frame[:,1]
    z = this_frame[:,2]

    if color is None:
        color = z

    if size is None:
        size = 2

    fig.add_trace(
        go.Scatter3d(
            x = x,
            y = y,
            z = z,
            marker = dict(
                color = color, 
                size = size,
                showscale = colorbar,
                colorscale = "Plotly3"),
            mode = "markers"
        )
    )

    fig.update_layout(scene = dict(aspectmode = "data"))
    fig.update_layout(
        width=600,
        height=1000,   # taller than wide works better for humans
    )
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=2.5, y=2.5, z=2.5)
        )
    )

    return fig

def plot_random_pose(n_points = 1000):
    arr = sampled_verts_from_random_action(n_points = n_points)
    
    return plot_arr(arr)

def remove_occluded_points(points, mesh, camera):
    
    # mesh: trimesh.Trimesh
    # points: (N, 3) sampled points on the mesh
    # camera: (3,) camera position

    # Direction vectors from camera to points
    dirs = points - camera
    dists = np.linalg.norm(dirs, axis=1)
    dirs = dirs / dists[:, None]

    # Small epsilon to avoid self-intersection
    epsilon = 1e-6
    origins = np.repeat(camera[None, :], len(points), axis=0) + epsilon * dirs

    # Ray-mesh intersection
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=dirs,
        multiple_hits=False
    )

    # Distance from camera to first hit
    hit_dists = np.linalg.norm(locations - camera, axis=1)

    # Initialize visibility mask
    visible = np.zeros(len(points), dtype=bool)

    # A point is visible if the first hit is at (approximately) the point
    visible[index_ray] = np.isclose(
        hit_dists,
        dists[index_ray],
        atol=1e-5
    )

    visible_points = points[visible]

    return visible_points

def get_walk_run_meshes(file_dict = None):
    dataset_path = "datasets/action_smplx_models"

    if file_dict is None:
        file_dict = {
            "walk": [
                "female_walk.npz",
                "male_walk_turn_left_90.npz"
            ],
            "run": [
                "female_run.npz",
                "male_run.npz"
            ]
        }

    walk_meshes = []
    run_meshes = []

    for file in file_dict["walk"]:
        print(file)
        this_path = dataset_path + "/" + file
        walk_meshes = walk_meshes + meshes_from_path(this_path)
    
    for file in file_dict["run"]:
        print(file)
        this_path = dataset_path + "/" + file
        run_meshes = run_meshes + meshes_from_path(this_path)

    return walk_meshes, run_meshes

def sample_from_meshes(meshes, n_points = 1_000, return_faces = False):
    output_points = []
    output_faces = []
    for mesh in meshes:
        points, faces = sampled_verts_from_mesh(mesh, n_points = n_points, return_faces=True)
        output_points.append(points)
        output_faces.append(faces)
    
    if return_faces:
        return np.array(output_points), np.array(output_faces)
    return np.array(output_points)

def construct_face_idx_to_region_json(force_override = False):
    if "face_idx_to_region.json" in os.listdir():
        if not force_override:
            print("File already exists, set force_override = True to override it")
            return False
    # Calibration model in a T pose that makes it easy to define cutoffs
    print("Loading mesh")
    mesh = mesh_from_path("datasets/action_smplx_models/male2_Calibration_stageii.npz")
    print("Mesh loaded")

    face_centers = []
    for i in range(mesh.faces.shape[0]):
        this_face = mesh.faces[i]
        these_verts = mesh.vertices[this_face]
        face_centers.append(these_verts.mean(axis = 0))

    face_centers = np.array(face_centers)

    x = face_centers[:,0]
    y = face_centers[:,1]
    z = face_centers[:,2]

    # Manually created region cutoffs (based on face centers)
    mask_dict = {}
    mask_dict["left_forearm"] = (x > -0.95) & (x < -0.7)
    mask_dict["left_hand"] = (x <= -0.95)
    mask_dict["right_forearm"] = (x < 2 * x.mean()-(-0.95)) & (x > 2 * x.mean()-(-0.7))
    mask_dict["right_hand"] = (x >= 2 * x.mean()-(-0.95))
    mask_dict["left_upper_arm"] = (x >= -0.7) & (x <= -0.5)
    mask_dict["right_upper_arm"] = (x <= 2 * x.mean()-(-0.7)) & (x >= 2 * x.mean() - (-0.5))
    mask_dict["head"] = (x >= -.5) & (x <= 2 * x.mean() - (-.5)) & ((z + y > 1.61)| (z > 1.53))
    mask_dict["upper_torso"] = (x >= -.5) & (x <= 2 * x.mean() - (-.5)) & ((z + y <= 1.61) & (z <= 1.53)) & (z > 1.2)
    mask_dict["lower_torso"] = (x >= -.5) & (x <= 2 * x.mean() - (-.5)) & (z <= 1.2) & ((z + y > 1.1) | (z > 1))
    mask_dict["pelvis"] = ((z + y <= 1.1) & (z <= 1)) & (-np.abs(x - x.mean()) + z > .67)
    mask_dict["left_thigh"] = (-np.abs(x - x.mean()) + z <= .67) & (z > .45) & (x <= x.mean())
    mask_dict["right_thigh"] = (-np.abs(x - x.mean()) + z <= .67) & (z > .45) & (x > x.mean())
    mask_dict["left_shin"] = (z <= .45) & (x <= x.mean()) & (z >= .08)
    mask_dict["right_shin"] = (z <= .45) & (x > x.mean()) & (z >= .08)
    mask_dict["left_foot"] = (x <= x.mean()) & (z < .08)
    mask_dict["right_foot"] = (x > x.mean()) & (z < .08)


    mask_df = pd.DataFrame(mask_dict)
    face_idx_to_region = {}
    for col in mask_df.columns:
        this_reg = mask_df[mask_df[col]]
        for idx in this_reg.index:
            face_idx_to_region[idx] = col
    
    with open('face_idx_to_region.json', 'w') as fp:
        json.dump(face_idx_to_region, fp)

    return True

def region_color_dict():
    dic = { 'left_forearm': '#636EFA',
            'left_hand': '#EF553B',
            'right_forearm': '#00CC96',
            'right_hand': '#AB63FA',
            'left_upper_arm': '#FFA15A',
            'right_upper_arm': '#19D3F3',
            'head': '#FF6692',
            'upper_torso': '#B6E880',
            'lower_torso': '#FF97FF',
            'pelvis': '#FECB52',
            'left_thigh': '#1f77b4',
            'right_thigh': '#ff7f0e',
            'left_shin': '#2ca02c',
            'right_shin': '#d62728',
            'left_foot': '#9467bd',
            'right_foot': '#8c564b' }
    return dic

def faces_to_regions(faces, return_colors = False):
    if "face_idx_to_region.json" in os.listdir():
        with open("face_idx_to_region.json", "r") as f:
            face_idx_to_region = json.load(f) 
        if return_colors:
            regions = [face_idx_to_region[str(face)] for face in faces]
            color_dict = region_color_dict()
            colors = [color_dict[reg] for reg in regions]
            return regions, colors
        else:
            return [face_idx_to_region[str(face)] for face in faces]
    else:
        construct_face_idx_to_region_json()
        return faces_to_regions(faces, return_colors=return_colors)

def face_idx_pairs_to_region_pairs(idx_pairs):
    start_regions = faces_to_regions(idx_pairs[:,0])
    matched_regions = faces_to_regions(idx_pairs[:,1])
    return zip(start_regions, matched_regions)

def region_accuracy(G, faces1, faces2):
    regions1 = faces_to_regions(faces1)
    regions2 = faces_to_regions(((G / G.max()) @ faces2).astype(int))
    correct = 0
    for i in range(len(regions1)):
        if regions1[i] == regions2[i]:
            correct += 1
    return correct / len(regions1)


def plot_3d_points_and_connections(points1, points2, G, switch_yz = False, color_incorrect = False, width = 600, height = 1000):
    """
    Given points1, points2, and G, plot the points and lines between matching points. If switch_xz is true then this will switch the x and z coordinates before plotting (since by default in the mocap data the x is the vertical axis).
    points1, points2: Nx3 arrays
    G: NxN array
    switch_xz: Boolean
    """
    if points1.shape[0] != points2.shape[0]:
        raise ValueError("Point clouds are not the same length")

    if G.shape[0] != G.shape[1]:
        raise ValueError("Matching matrix is not square")

    if G.shape[0] != points1.shape[0]:
        raise ValueError("Matching matrix dimensions don't match point cloud dimensions")

    if np.count_nonzero(G) > points1.shape[0]:
        raise ValueError("Matching has too many nonzero entries")

    if np.count_nonzero(G) < points1.shape[0]:
        raise ValueError("Matching has too few nonzero entries")

    x_ind = 0
    if switch_yz:
        y_ind = 2
        z_ind = 1
    else:
        y_ind = 1
        z_ind = 2

    # Ensure numpy arrays
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    G = np.asarray(G)

    fig = go.Figure()

    # Plot first set of 3D points
    fig.add_trace(go.Scatter3d(
        x=points1[:, x_ind], y=points1[:, y_ind], z=points1[:, z_ind],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Points 1'
    ))

    # Plot second set of 3D points
    fig.add_trace(go.Scatter3d(
        x=points2[:, x_ind], y=points2[:, y_ind], z=points2[:, z_ind],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Points 2'
    ))

    # Draw connections for nonzero G[i, j]
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if G[i, j] != 0:
                c = "gray"
                if color_incorrect and i != j:
                    c = "red"
                p1 = points1[i]
                p2 = points2[j]
                fig.add_trace(go.Scatter3d(
                    x=[p1[x_ind], p2[x_ind]],
                    y=[p1[y_ind], p2[y_ind]],
                    z=[p1[z_ind], p2[z_ind]],
                    mode='lines',
                    line=dict(color=c, width=2),
                    showlegend=False
                ))

    # Layout styling
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title='3D Points with Connections',
        template='plotly_white',
        width = width,
        height = height
    )
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=2.5, y=2.5, z=2.5)
        )
    )

    return fig

def plot_3d_points_and_connections_region_matched(points1, points2, faces1, faces2, G, switch_yz = False, plot_both = True, width = 600, height = 1000):
    """
    Given points1, points2, and G, plot the points and lines between matching points. If switch_xz is true then this will switch the x and z coordinates before plotting (since by default in the mocap data the x is the vertical axis).
    points1, points2: Nx3 arrays
    G: NxN array
    switch_xz: Boolean
    """
    if points1.shape[0] != points2.shape[0]:
        raise ValueError("Point clouds are not the same length")

    if G.shape[0] != G.shape[1]:
        raise ValueError("Matching matrix is not square")

    if G.shape[0] != points1.shape[0]:
        raise ValueError("Matching matrix dimensions don't match point cloud dimensions")

    if np.count_nonzero(G) > points1.shape[0]:
        raise ValueError("Matching has too many nonzero entries")

    if np.count_nonzero(G) < points1.shape[0]:
        raise ValueError("Matching has too few nonzero entries")

    print("Region accuracy:", region_accuracy(G, faces1, faces2))

    x_ind = 0
    if switch_yz:
        y_ind = 2
        z_ind = 1
    else:
        y_ind = 1
        z_ind = 2

    # Ensure numpy arrays
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    G = np.asarray(G)

    fig = go.Figure()

    regions1 = faces_to_regions(faces1)
    regions2 = faces_to_regions(((G / G.max()) @ faces2).astype(int))
    color1 = ["red" if regions1[i] != regions2[i] else "green" for i in range(len(regions1))]

    # Plot first set of 3D points
    fig.add_trace(go.Scatter3d(
        x=points1[:, x_ind], y=points1[:, y_ind], z=points1[:, z_ind],
        mode='markers',
        marker=dict(size=5, color=color1),
        name='Points 1'
    ))

    regions1 = faces_to_regions(((G.T / G.max()) @ faces1).astype(int))
    regions2 = faces_to_regions(faces2)
    color2 = ["red" if regions1[i] != regions2[i] else "green" for i in range(len(regions1))]

    if plot_both:
        # Plot second set of 3D points
        fig.add_trace(go.Scatter3d(
            x=points2[:, x_ind], y=points2[:, y_ind], z=points2[:, z_ind],
            mode='markers',
            marker=dict(size=5, color= color2),
            name='Points 2'
        ))


        # Draw connections for nonzero G[i, j]
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                if G[i, j] != 0:
                    p1 = points1[i]
                    p2 = points2[j]
                    fig.add_trace(go.Scatter3d(
                        x=[p1[x_ind], p2[x_ind]],
                        y=[p1[y_ind], p2[y_ind]],
                        z=[p1[z_ind], p2[z_ind]],
                        mode='lines',
                        line=dict(color="gray", width=2),
                        showlegend=False
                    ))

    # Layout styling
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title='3D Points with Connections',
        template='plotly_white',
        width = width,
        height = height
    )
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=2.5, y=2.5, z=2.5)
        )
    )
    return fig