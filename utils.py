import numpy as np
import smplx
import trimesh
import torch
import os
import plotly.graph_objects as go

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

def vertices_to_pointcloud(vertices, faces, n_points=1_000, return_mesh = False):
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False
    )
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    
    if return_mesh:
        return points, mesh
    else:
        return points

def sampled_verts_from_path(amass_npz_path, idx = None, n_points = 1_000, return_mesh = False):
    
    verts, faces = true_verts_from_path(amass_npz_path, return_faces = True)
    verts = verts.detach().numpy()

    if idx is None or idx >= verts.shape[0]:
        idx = np.random.randint(0, verts.shape[0])
        print(f"Randomly chosen index: {idx}")

    return vertices_to_pointcloud(verts[idx], faces, n_points = n_points, return_mesh = return_mesh)

def choose_random_file(path = "datasets/action_smplx_models"):
    
    files = os.listdir(path)
    file = np.random.choice(files)

    print(f"Randomly chosen file: {file}")

    return path + "/" + file

def sampled_verts_from_random_action(n_points = 1_000, return_mesh = False):
    file = choose_random_file()

    return sampled_verts_from_path(file, n_points = n_points, return_mesh = return_mesh)

def plot_arr(arr):
    fig = go.Figure()

    this_frame = arr
    x = this_frame[:,0]
    y = this_frame[:,1]
    z = this_frame[:,2]

    fig.add_trace(
        go.Scatter3d(
            x = x,
            y = y,
            z = z,
            marker = dict(color = z, size = 2),
            mode = "markers"
        )
    )

    fig.update_layout(scene = dict(aspectmode = "data"))
    fig.update_layout(
        width=300,
        height=500,   # taller than wide works better for humans
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
