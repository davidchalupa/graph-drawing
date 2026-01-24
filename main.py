import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# convex hull (monotone chain)
def convex_hull(points):
    # points: list of (x, y, v)
    pts = sorted(points)
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2], upper[-1], p)<=0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

# automatic Tutte embedding
def tutte_embedding_auto(G):
    is_planar, _ = nx.check_planarity(G)
    if not is_planar:
        raise ValueError("Graph is not planar")

    # Step 1: get a planar straight-line drawing (may be ugly, but planar)
    pos0 = nx.planar_layout(G)

    # Step 2: extract convex hull = outer face
    pts = [(pos0[v][0], pos0[v][1], v) for v in G.nodes()]
    hull = convex_hull(pts)
    boundary = [v for _,_,v in hull]

    # Step 3: place boundary on convex polygon
    k = len(boundary)
    boundary_pos = {}
    for i,v in enumerate(boundary):
        angle = 2*np.pi*i/k
        boundary_pos[v] = (np.cos(angle), np.sin(angle))

    # Step 4: Tutte linear system
    interior = [v for v in G.nodes() if v not in boundary]
    idx = {v:i for i,v in enumerate(interior)}

    n = len(interior)
    A = np.zeros((n,n))
    bx = np.zeros(n)
    by = np.zeros(n)

    for v in interior:
        i = idx[v]
        neighbors = list(G.neighbors(v))
        A[i,i] = len(neighbors)
        for u in neighbors:
            if u in boundary:
                bx[i] += boundary_pos[u][0]
                by[i] += boundary_pos[u][1]
            else:
                A[i,idx[u]] -= 1

    x = np.linalg.solve(A, bx)
    y = np.linalg.solve(A, by)

    pos = dict(boundary_pos)
    for v,i in idx.items():
        pos[v] = (x[i], y[i])

    return pos

def get_example_graph():
    G = nx.Graph()
    G.add_edges_from([
        (0,1),(1,2),(2,3),(3,0),
        (0,4),(1,4),(2,4),(3,4)
    ])
    return G

def load_from_col_file(file_path):
    """
    Loads an undirected graph from a .col file into a networkx Graph.

    :param file_path: Path to the .col file
    :return: A networkx Graph object
    """
    G = nx.Graph()

    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and skip comments
            line = line.strip()
            if line.startswith('c'):  # Ignore comment lines
                continue
            if line.startswith('p'):  # Read metadata line
                _, _, num_vertices, num_edges = line.split()
                continue
            if line.startswith('e'):  # Read edge lines
                _, u, v = line.split()
                if u != v:
                    G.add_edge(int(u) - 1, int(v) - 1)
                    G.add_edge(int(v) - 1, int(u) - 1)

    return G

# G = get_example_graph()
G = load_from_col_file("data/example3.col")

print(f"Loaded a graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

print("Computing a planar drawing ...")

pos = tutte_embedding_auto(G)

plt.figure(figsize=(5,5))
for u,v in G.edges():
    plt.plot([pos[u][0], pos[v][0]],
             [pos[u][1], pos[v][1]], 'k-')
for v,(x,y) in pos.items():
    plt.scatter(x,y)
    plt.text(x+0.03,y+0.03,str(v))
plt.axis("equal")
plt.title("Guaranteed crossing-free Tutte embedding")
plt.show()
