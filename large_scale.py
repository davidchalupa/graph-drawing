import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def compute_radial_layout(G):
    """
    A classical constructive radial layout based on BFS levels.3
    """
    # root - the node with the highest degree
    root = max(G.nodes, key=G.degree)

    # shortest path distance from the root to all other nodes
    # defines the radial levels
    levels = nx.single_source_shortest_path_length(G, root)

    # group nodes by their level
    nodes_by_level = {}
    for node, dist in levels.items():
        nodes_by_level.setdefault(dist, []).append(node)

    pos = {}
    # assign coordinates level by level
    for level, nodes in nodes_by_level.items():
        radius = level
        n_nodes = len(nodes)

        for i, node in enumerate(nodes):
            # nodes evenly distributed around the circle
            angle = 2 * np.pi * i / n_nodes
            pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

    return pos

def compute_organic_radial(G):
    """
    Constructive radial layout with a spiral jitter to
    break up the 'concentric ring' look.
    """
    root = max(G.nodes, key=G.degree)
    levels = nx.single_source_shortest_path_length(G, root)

    nodes_by_level = {}
    for node, dist in levels.items():
        nodes_by_level.setdefault(dist, []).append(node)

    pos = {}
    for level, nodes in nodes_by_level.items():
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            # The 'Secret Sauce': Add a small spiral offset to the radius
            # based on the node index to prevent hard circles.
            radius = level + (i / n_nodes) * 0.5

            angle = 2 * np.pi * i / n_nodes
            pos[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

    return pos


def compute_hde_layout(G, k=30):
    """
    Computes the High-Dimensional Embedder (HDE) layout.
    1. Selects k pivot nodes.
    2. Calculates shortest path distances from all nodes to these pivots.
    3. Uses PCA to project the k-dimensional distances into 2D coordinates.
    """
    # selects k pivot nodes
    # for simplicity, we use just a few high-degree hubs here
    nodes = list(G.nodes())
    pivots = []

    # start with the biggest hub
    first_pivot = max(G.nodes, key=G.degree)
    pivots.append(first_pivot)

    # greedily pick nodes furthest from current pivots (approximate)
    sorted_hubs = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    pivots = sorted_hubs[:k]

    # build the nodes - pivots distance matrix (V x k)
    dist_matrix = np.zeros((len(G), k))
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    for j, pivot in enumerate(pivots):
        lengths = nx.single_source_shortest_path_length(G, pivot)
        for node, dist in lengths.items():
            dist_matrix[node_to_idx[node], j] = dist

    # dimensionality reduction (PCA)
    # we project the k-dimensional distance vectors into 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(dist_matrix)

    pos = {node: coords[node_to_idx[node]] for node in nodes}
    return pos


def compute_radial_hde_layout(G, k=30):
    """
    Computes a radial variant of the HDE layout.
    1. Selects k pivot nodes (hubs).
    2. Calculates shortest path distances.
    3. Uses PCA to determine angular orientation.
    4. Uses mean distance to pivots to determine the radius.
    """
    nodes = list(G.nodes())

    # select pivots (high-degree hubs)
    sorted_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    pivots = sorted_nodes[:k]

    # build distance matrix (V x k)
    dist_matrix = np.zeros((len(G), k))
    for j, pivot in enumerate(pivots):
        lengths = nx.single_source_shortest_path_length(G, pivot)
        # Fill missing values with a large distance if graph is disconnected
        default_dist = max(lengths.values()) + 1 if lengths else 0
        for i, node in enumerate(nodes):
            dist_matrix[i, j] = lengths.get(node, default_dist)

    # PCA for angular distribution
    # we use PCA to find the 'direction' of a node relative to the hub cluster
    # instead of calculating the positions directly
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(dist_matrix)

    # pre-calculate means to find the closest possible center point
    mean_dists = np.mean(dist_matrix, axis=1)
    min_dist = np.min(mean_dists)

    # transform these to radial coordinates
    pos = {}
    for i, node in enumerate(nodes):
        theta = np.arctan2(pca_coords[i, 1], pca_coords[i, 0])

        # determine the node's "gravity":
        # how close is this node to the main hubs on average
        # the closer it is, the closer it stays to the center (low r)
        # subtraction centers the hubs at 0.
        # squaring (or cubing) creates the "core vs shell" distinction
        r = (mean_dists[i] - min_dist) ** 2

        # place the node on the map
        # we take the direction (theta) and the distance (r)
        # and "unfold" them into standard X and Y coordinates (polar to Cartesian coordinates)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        pos[node] = np.array([x, y])

    return pos

def compute_spectral_refined_layout(G, iterations_refinement=5):
    """
    Computes a spectral layout refined by a few iterations of spring layout to
    spread out the center of the graph that would otherwise appear as a small blob.
    """
    print("Computing spectral skeleton...")
    pos_init = nx.spectral_layout(G)

    print("Expanding the core...")
    pos = nx.spring_layout(
        G,
        pos=pos_init,
        iterations=iterations_refinement,
        # k controls the optimal distance between nodes
        k=2 / np.sqrt(len(G))
    )
    return pos

def visualize_graph(G, pos, title):
    """
    Renders the graph using the provided positions.
    Optimized for ~5000 nodes using small markers and transparency.
    """
    plt.figure(figsize=(12, 12))

    nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color='gray', width=0.5)

    degrees = [G.degree(n) for n in G.nodes()]
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=10,
        node_color=degrees,
        cmap=plt.cm.viridis,
        alpha=0.8
    )

    plt.colorbar(nodes, label='Node Degree')
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_graph_dark_theme(G, pos, title):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 12))

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    degree_values = [degrees[n] for n in G.nodes()]

    nx.draw_networkx_edges(
        G, pos,
        alpha=0.15,
        edge_color='#1E90FF',
        width=0.5,
        ax=ax
    )

    node_sizes = [((d / max_deg) * 55) + 8 for d in degree_values]

    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=degree_values,
        cmap=plt.cm.cool,
        alpha=0.6,
        ax=ax
    )

    ax.set_aspect('equal')
    ax.set_axis_off()

    plt.title(title, color='white', fontsize=18, pad=20, alpha=0.8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    N = 5000
    M = 2
    print(f"Generating Barabasi-Albert network (N={N}, m={M})...")
    G_demo = nx.barabasi_albert_graph(N, M)

    print("Computing graph layout...")
    # positions = compute_radial_layout(G_demo)
    # title = "BA graph: radial layout"

    # positions = compute_organic_radial(G_demo)
    # title = "BA graph: organic radial layout"

    # positions = compute_hde_layout(G_demo)
    # title = "BA graph: HDE layout"

    positions = compute_radial_hde_layout(G_demo)
    title = "BA graph: radial HDE layout"

    # positions = compute_spectral_refined_layout(G_demo)
    # title = "BA graph: refined spectral layout"

    print("Visualizing...")
    # visualize_graph(G_demo, positions, title=title)
    visualize_graph_dark_theme(G_demo, positions, title=title)
