import numpy as np
import time
from collections import defaultdict
from dev.util import logger

def interactions_to_adjacency(interactions, num_nodes):
    """
    Convert interaction list to weighted adjacency matrix.

    Args:
        interactions: List of [node1, node2] pairs (repeated pairs indicate higher weights)
        num_nodes: Number of nodes in the graph

    Returns:
        Weighted adjacency matrix as numpy array
    """
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for interaction in interactions:
        i, j = interaction[0], int(interaction[1])
        adj_matrix[i, j] += 1.0
        adj_matrix[j, i] += 1.0  # Make it symmetric (undirected graph)

    return adj_matrix


def node_correctness_asymmetric(coup, m, n):
    """
    Calculate node correctness for asymmetric coupling matrices.
    Assumes first m nodes in target correspond to m source nodes.

    Args:
        coup: Coupling matrix of shape (m, n) where n >= m
        m: Number of source nodes
        n: Number of target nodes

    Returns:
        Accuracy of node matching
    """
    coup_max = coup.argmax(1)  # For each source node, find best target match

    # Ground truth: source node i should map to target node i (for i < m)
    correct = 0
    for i in range(m):
        if coup_max[i] == i:
            correct += 1

    acc = correct / m
    return acc


def process_interaction_data(data_dict, track_accuracy=True, accuracy_interval=10):
    """
    Process the interaction data dictionary and run BAPG.

    Args:
        data_dict: Dictionary with keys 'src_index', 'tar_index',
                   'src_interactions', 'tar_interactions'
        track_accuracy: Whether to track accuracy across iterations
        accuracy_interval: How often to compute accuracy (every N iterations)

    Returns:
        Dictionary with results including coupling matrix, accuracy, and runtime
    """
    # Extract data
    src_interactions = data_dict['src_interactions']
    tar_interactions = data_dict['tar_interactions']

    # Get number of nodes
    m = len(data_dict['src_index'])  # Source nodes
    n = len(data_dict['tar_index'])  # Target nodes

    print(f"Source graph: {m} nodes")
    print(f"Target graph: {n} nodes")

    # Convert interactions to adjacency matrices
    G_adj = interactions_to_adjacency(src_interactions, m)
    G_adj_noise = interactions_to_adjacency(tar_interactions, n)

    print(f"Source edges: {np.sum(G_adj > 0) / 2}")  # Divide by 2 for undirected
    print(f"Target edges: {np.sum(G_adj_noise > 0) / 2}")

    # Initialize probability distributions
    p = np.ones([m, 1], dtype=np.float32) / m
    q = np.ones([n, 1], dtype=np.float32) / n
    Xinit = p @ q.T

    # Run BAPG with adjusted rho for stability
    start = time.time()
    rho = 1.0  # Increased from 0.1 for better numerical stability
    coup_bap, obj_list_bap, acc_list_bap = BAPG_numpy(
        A=G_adj,
        B=G_adj_noise,
        a=p,
        b=q,
        X=Xinit,
        epoch=5000,
        eps=1e-5,
        rho=rho,
        track_accuracy=track_accuracy,
        accuracy_interval=accuracy_interval,
        m=m,
        n=n
    )
    end = time.time()

    runtime = end - start
    accuracy = node_correctness_asymmetric(coup_bap, m, n)

    results = {
        'coupling_matrix': coup_bap,
        'accuracy': accuracy,
        'runtime': runtime,
        'objective_list': obj_list_bap,
        'accuracy_list': acc_list_bap,
        'source_adj': G_adj,
        'target_adj': G_adj_noise
    }

    return results


# BAPG_numpy function with numerical stabilization and accuracy tracking
def BAPG_numpy(A, B, a=None, b=None, X=None, epoch=2000, eps=1e-5, rho=1e-1,
               track_accuracy=False, accuracy_interval=10, m=None, n=None):
    """
    BAPG algorithm with optional accuracy tracking.

    Args:
        A: Source adjacency matrix
        B: Target adjacency matrix
        a: Source marginal distribution
        b: Target marginal distribution
        X: Initial coupling matrix
        epoch: Maximum number of iterations
        eps: Convergence threshold
        rho: Step size parameter
        track_accuracy: Whether to track accuracy across iterations
        accuracy_interval: How often to compute accuracy (every N iterations)
        m: Number of source nodes (required if track_accuracy=True)
        n: Number of target nodes (required if track_accuracy=True)

    Returns:
        X: Final coupling matrix
        obj_list: List of objective values
        acc_list: List of accuracy values (if track_accuracy=True)
    """
    if a is None:
        a = np.ones([A.shape[0], 1], dtype=np.float32)/A.shape[0]
    if b is None:
        b = np.ones([B.shape[0], 1], dtype=np.float32)/B.shape[0]
    if X is None:
        X = a@b.T

    obj_list, acc_list = [], []

    for ii in range(epoch):
        X = X + 1e-10

        # First update with numerical stabilization
        exponent1 = A@X@B/rho
        exponent1 = np.clip(exponent1, -500, 500)  # Prevent overflow
        X = np.exp(exponent1)*X
        X = X * (a / (X @  np.ones_like(b)))

        # Second update with numerical stabilization
        exponent2 = A@X@B/rho
        exponent2 = np.clip(exponent2, -500, 500)  # Prevent overflow
        X = np.exp(exponent2)*X
        X = X * (b.T / (X.T @ np.ones_like(a)).T)

        # Track accuracy at specified intervals
        if track_accuracy:
            if m is not None and n is not None:
                acc = node_correctness_asymmetric(X, m, n)
                acc_list.append((ii, acc))
                if ii % 50 == 0:
                    print(f"Iteration {ii}: Accuracy = {acc:.4f}")

        # Track objective
        if ii > 0 and ii % 50 == 0:
            objective = -np.trace(A @ X @ B @ X.T)
            # if len(obj_list) > 0 and np.abs((objective-obj_list[-1])/obj_list[-1]) < eps:
            #     logger.info('iter:{}, smaller than eps'.format(ii))
            #     # Get final accuracy if tracking
            #     if track_accuracy and m is not None and n is not None:
            #         final_acc = node_correctness_asymmetric(X, m, n)
            #         acc_list.append((ii, final_acc))
            #     break
            obj_list.append(objective)

    return X, obj_list, acc_list


# Example usage
if __name__ == "__main__":
    # Your example data
    data = {
        'src_index': {i: i for i in range(10)},
        'tar_index': {i: i for i in range(11)},
        'src_interactions': [[0, np.int32(7)]]*6 + [[1, np.int32(0)]]*7 + [[2, np.int32(5)]]*3 +
                           [[3, np.int32(6)]]*6 + [[4, np.int32(1)]]*3 + [[5, np.int32(7)]]*8 +
                           [[6, np.int32(8)]]*10 + [[7, np.int32(9)]]*2 + [[8, np.int32(1)]]*9 +
                           [[9, np.int32(8)]]*3,
        'tar_interactions': [[0, np.int32(7)]]*6 + [[1, np.int32(0)]]*7 + [[2, np.int32(5)]]*3 +
                           [[3, np.int32(6)]]*6 + [[4, np.int32(1)]]*3 + [[5, np.int32(7)]]*8 +
                           [[6, np.int32(8)]]*10 + [[7, np.int32(9)]]*2 + [[8, np.int32(1)]]*9 +
                           [[9, np.int32(8)]]*3 + [[0, np.int32(10)]]*10 + [[3, np.int32(10)]]*10 +
                           [[4, np.int32(7)]]*3 + [[5, np.int32(1)]]*5 + [[6, np.int32(0)]]*3 +
                           [[7, np.int32(2)]]*6 + [[9, np.int32(2)]]*5 + [[10, np.int32(1)]],
        'mutual_interactions': None
    }

    # Process the data and run BAPG with accuracy tracking
    results = process_interaction_data(data, track_accuracy=True, accuracy_interval=10)

    print(f"\nResults:")
    print(f"Runtime: {results['runtime']:.4f} seconds")
    print(f"Final node matching accuracy: {results['accuracy']:.4f}")
    print(f"Coupling matrix shape: {results['coupling_matrix'].shape}")

    print(f"\nAccuracy progression:")
    for iteration, acc in results['accuracy_list']:
        print(f"  Iteration {iteration}: {acc:.4f}")

    print(f"\nBest matches for each source node:")
    for i in range(len(data['src_index'])):
        best_match = results['coupling_matrix'][i].argmax()
        confidence = results['coupling_matrix'][i, best_match]
        print(f"  Source {i} -> Target {best_match} (confidence: {confidence:.4f})")