import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

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
    n = len(interior)
    rows = []
    cols = []
    data = []
    bx = np.zeros(n, dtype=float)
    by = np.zeros(n, dtype=float)

    for v in interior:
        i = idx[v]
        nbrs = list(G.neighbors(v))
        degv = len(nbrs)
        rows.append(i);
        cols.append(i);
        data.append(degv)
        for u in nbrs:
            if u in boundary:
                bx[i] += boundary_pos[u][0]
                by[i] += boundary_pos[u][1]
            else:
                j = idx[u]
                rows.append(i);
                cols.append(j);
                data.append(-1.0)

    # sparse solve
    A_csr = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    x = spsolve(A_csr, bx)
    y = spsolve(A_csr, by)

    pos = dict(boundary_pos)
    for v,i in idx.items():
        pos[v] = (x[i], y[i])

    return pos

def low_crossing_layout_auto(G,
                             prefer_edge_order='degprod',
                             refine_iterations=60,
                             coarse_random_seed=None):
    """
    Constructive heuristic layout for (possibly non-planar) graph G.
    Returns a dict mapping node -> (x,y) positions (floats).

    Strategy:
      1. Greedy build a planar subgraph S by trying to add edges in a chosen order
         (default: by product of endpoint degrees).
      2. Find a simple cycle in S to use as boundary (longest cycle from cycle_basis).
      3. Compute Tutte (barycentric) embedding for S with boundary fixed on a circle.
      4. Run a short Spring layout (Fruchterman-Reingold) on the original G,
         initialized from Tutte positions, keeping boundary nodes fixed.

    Parameters
    ----------
    G : networkx.Graph
        Input graph (unchanged).
    prefer_edge_order : {'degprod','degsum','random'}
        Ordering heuristic for greedily building the planar subgraph.
        'degprod' - edges sorted by degree(u)*degree(v) descending (keeps hub-links earlier).
        'degsum'  - edges sorted by degree(u)+degree(v) descending.
        'random'  - random order.
    refine_iterations : int
        Number of iterations to run networkx.spring_layout for refinement stage.
    coarse_random_seed : int or None
        Optional RNG seed for reproducibility when random choices happen.
    """
    if coarse_random_seed is not None:
        random.seed(coarse_random_seed)
        np.random.seed(coarse_random_seed)

    nodes = list(G.nodes())

    # --- 1) greedily build a planar subgraph S ---
    S = nx.Graph()
    S.add_nodes_from(nodes)

    edges = list(G.edges())
    degs = dict(G.degree())

    if prefer_edge_order == 'degprod':
        edges.sort(key=lambda e: (degs.get(e[0], 0) * degs.get(e[1], 0)), reverse=True)
    elif prefer_edge_order == 'degsum':
        edges.sort(key=lambda e: (degs.get(e[0], 0) + degs.get(e[1], 0)), reverse=True)
    elif prefer_edge_order == 'random':
        random.shuffle(edges)
    else:
        random.shuffle(edges)

    # Greedily add edges if planarity preserved (use current networkx API)
    for (u, v) in edges:
        S.add_edge(u, v)
        is_planar, _embedding = nx.check_planarity(S)  # <--- corrected: no 'embed' kw
        if not is_planar:
            S.remove_edge(u, v)

    # fallback if S has no edges
    if S.number_of_edges() == 0 or S.number_of_nodes() == 0:
        pos = nx.spring_layout(G, iterations=refine_iterations)
        return {n: tuple(pos[n]) for n in G.nodes()}

    # --- 2) find a reasonable boundary cycle for Tutte ---
    cycles = nx.cycle_basis(S)
    boundary = None
    if cycles:
        cycles_sorted = sorted(cycles, key=lambda c: len(c), reverse=True)
        boundary = cycles_sorted[0]
        if len(boundary) < 3:
            boundary = None

    if boundary is None:
        pos = nx.spring_layout(G, iterations=refine_iterations)
        return {n: tuple(pos[n]) for n in G.nodes()}

    def make_cycle_ordered(cycle, subG):
        if len(cycle) < 3:
            return cycle
        cycset = set(cycle)
        start = cycle[0]
        ordered = [start]
        prev = None
        cur = start
        while True:
            found = None
            for nb in subG[cur]:
                if nb in cycset and nb != prev:
                    found = nb
                    break
            if found is None:
                break
            if found == start:
                break
            ordered.append(found)
            prev, cur = cur, found
            if len(ordered) > len(cycle) + 5:
                break
        if len(ordered) == len(cycle):
            return ordered
        return cycle

    boundary = make_cycle_ordered(boundary, S)
    if len(boundary) < 3:
        pos = nx.spring_layout(G, iterations=refine_iterations)
        return {n: tuple(pos[n]) for n in G.nodes()}

    # --- 3) Tutte embedding on S using chosen boundary ---
    boundary_set = set(boundary)
    interior = [v for v in S.nodes() if v not in boundary_set]
    if len(interior) == 0:
        pos = {}
        R = 1.0
        L = len(boundary)
        for i, v in enumerate(boundary):
            a = 2 * math.pi * i / L
            pos[v] = np.array([R * math.cos(a), R * math.sin(a)], dtype=float)
        for v in set(G.nodes()) - set(pos.keys()):
            pos[v] = np.array([0.0, 0.0], dtype=float)
        pos = nx.spring_layout(G, pos=pos, fixed=boundary, iterations=refine_iterations)
        return {n: tuple(pos[n]) for n in G.nodes()}

    L = len(boundary)
    R = 1.0
    boundary_pos = {}
    for i, v in enumerate(boundary):
        a = 2 * math.pi * i / L
        boundary_pos[v] = np.array([R * math.cos(a), R * math.sin(a)], dtype=float)

    idx = {v: i for i, v in enumerate(interior)}
    n_in = len(interior)

    # build sparse representation from triplets (rows, cols, data)
    rows = []
    cols = []
    data = []
    bx = np.zeros(n_in, dtype=float)
    by = np.zeros(n_in, dtype=float)

    for i, v in enumerate(interior):
        nbrs = list(S.neighbors(v))
        degv = len(nbrs)
        rows.append(i)
        cols.append(i)
        data.append(degv)
        for w in nbrs:
            if w in boundary_set:
                bx[i] += boundary_pos[w][0]
                by[i] += boundary_pos[w][1]
            else:
                j = idx[w]
                rows.append(i)
                cols.append(j)
                data.append(-1.0)

    # sparse construction of matrix A
    A_csr = sparse.csr_matrix((data, (rows, cols)), shape=(n_in, n_in))

    try:
        # sparse solve
        sol_x = spsolve(A_csr, bx)
        sol_y = spsolve(A_csr, by)
    except np.linalg.LinAlgError:
        A_dense = A_csr.toarray()
        sol_x, *_ = np.linalg.lstsq(A_dense, bx, rcond=None)
        sol_y, *_ = np.linalg.lstsq(A_dense, by, rcond=None)

    pos = {}
    for v, i in idx.items():
        pos[v] = np.array([sol_x[i], sol_y[i]], dtype=float)
    for v in boundary:
        pos[v] = boundary_pos[v].copy()
    for v in G.nodes():
        if v not in pos:
            pos[v] = np.array([0.0, 0.0], dtype=float)

    # --- 4) refinement: short spring layout keeping boundary fixed ---
    try:
        pos_refined = nx.spring_layout(G, pos=pos, fixed=list(boundary), iterations=max(10, refine_iterations))
        pos = {v: np.array(pos_refined[v], dtype=float) for v in G.nodes()}
    except Exception:
        for v in G.nodes():
            if v not in pos or np.allclose(pos[v], 0.0):
                pos[v] = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], dtype=float)

    # normalize & scale
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    span = max(maxx - minx, maxy - miny)
    if span < 1e-8:
        span = 1.0
    scale = 1.0 / span
    out = {}
    for v, p in pos.items():
        out[v] = ((p[0] - (minx + maxx) / 2.0) * scale, (p[1] - (miny + maxy) / 2.0) * scale)

    return out

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
G = load_from_col_file("data/example5.col")

print(f"Loaded a graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

is_planar, _ = nx.check_planarity(G)
if is_planar:
    print("Graph is planar. Computing a planar drawing ...")
    pos = tutte_embedding_auto(G)
else:
    print("Graph is not planar. Computing a low-crossing drawing ...")
    pos = low_crossing_layout_auto(G)

plt.figure(figsize=(5,5))
for u,v in G.edges():
    plt.plot([pos[u][0], pos[v][0]],
             [pos[u][1], pos[v][1]], 'k-')
for v,(x,y) in pos.items():
    plt.scatter(x,y)
    plt.text(x+0.03,y+0.03,str(v))
plt.axis("equal")
if is_planar:
    plt.title("Crossing-free Tutte embedding")
else:
    plt.title("Low-crossing non-planar graph embedding")
plt.show()
