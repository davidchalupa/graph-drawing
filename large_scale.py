import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


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


def visualize_graph(G, pos, title):
    """
    Renders the graph using the provided positions.
    Optimized for ~5000 nodes using small markers and transparency.
    """
    plt.figure(figsize=(12, 12))

    # Draw edges with low alpha to prevent a 'hairball' effect
    nx.draw_networkx_edges(G, pos, alpha=0.05, edge_color='gray', width=0.5)

    # Draw nodes; color them by degree to highlight hubs
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


def compute_organic_radial(G):
    """
    Constructive radial layout with a spiral jitter to
    break up the 'concentric ring' look.
    """
    # 1. Root selection (Highest degree hub)
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


def visualize_graph_dark_theme(G, pos, title):
    # set the dark mode
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 14))

    degrees = dict(G.degree())
    max_deg = max(degrees.values())

    nx.draw_networkx_edges(
        G, pos,
        alpha=0.03,
        edge_color='cyan',
        width=0.3,
        ax=ax
    )

    # size and color of nodes based on their degrees
    node_sizes = [((degrees[n] / max_deg) * 50) + 2 for n in G.nodes()]

    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=list(degrees.values()),
        cmap=plt.cm.magma,
        alpha=0.9,
        ax=ax
    )

    ax.set_axis_off()
    plt.title(title,
              color='white', fontsize=16, pad=20)

    plt.show()


from sklearn.decomposition import PCA


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


if __name__ == "__main__":
    N = 5000
    M = 2
    print(f"Generating Barabasi-Albert network (N={N}, m={M})...")
    G_demo = nx.barabasi_albert_graph(N, M)

    print("Computing graph layout...")
    # positions = compute_radial_layout(G_demo)
    # positions = compute_organic_radial(G_demo)
    positions = compute_hde_layout(G_demo)

    # title = "BA graph: radial layout"
    title = "BA graph: HDE layout"

    print("Visualizing...")
    visualize_graph(G_demo, positions, title=title)
    # visualize_graph_dark_theme(G_demo, positions, title=title)
