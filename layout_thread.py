from PyQt6.QtCore import QThread, pyqtSignal

import math
import random
import numpy as np
import networkx as nx

from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA


class LayoutThread(QThread):
    layout_finished = pyqtSignal(dict)

    def __init__(self, G, mode="spring"):
        super().__init__()
        self.G = G
        self.mode = mode

    def run(self):
        if not self.G or not self.G.nodes:
            self.layout_finished.emit({})
            return

        if self.mode == "pca":
            pos = self.compute_hde_layout(self.G)
        elif self.mode == "lowcross":
            pos = self.low_crossing_layout_auto(self.G)
        elif self.mode == "radial":
            pos = self.compute_radial_layout(self.G)
        else:
            pos = nx.spring_layout(self.G, scale=1000)

        seen_positions = {}
        jitter_amount = 1.0

        for node in pos:
            p_tuple = (round(pos[node][0], 6), round(pos[node][1], 6))
            if p_tuple in seen_positions:
                pos[node][0] += random.uniform(-jitter_amount, jitter_amount)
                pos[node][1] += random.uniform(-jitter_amount, jitter_amount)
            else:
                seen_positions[p_tuple] = node

        scale_val = 150 if self.mode == "pca" else (2000 if self.mode == "lowcross" else 1.0)
        for node in pos:
            pos[node][0] *= (scale_val * 1.77)
            pos[node][1] *= scale_val

        self.layout_finished.emit(pos)

    def compute_hde_layout(self, G, k=30):
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        k = min(k, len(nodes))
        sorted_hubs = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
        pivots = sorted_hubs[:k]

        dist_matrix = np.full((len(nodes), k), 100.0)

        for j, pivot in enumerate(pivots):
            lengths = nx.single_source_shortest_path_length(G, pivot)
            for node, dist in lengths.items():
                if node in node_to_idx:
                    dist_matrix[node_to_idx[node], j] = dist

        pca = PCA(n_components=2)
        coords = pca.fit_transform(dist_matrix)
        return {nodes[i]: coords[i] for i in range(len(nodes))}

    def compute_radial_layout(self, G):
        if not G or not G.nodes:
            return {}

        root = max(G.nodes(), key=G.degree)
        lengths = nx.single_source_shortest_path_length(G, root)

        levels = {}
        for node, dist in lengths.items():
            levels.setdefault(dist, []).append(node)

        pos = {}
        max_dist = max(levels.keys()) if levels else 1
        radius_step = 1000.0 / max_dist if max_dist > 0 else 1000.0

        for dist, nodes_in_level in levels.items():
            radius = dist * radius_step
            count = len(nodes_in_level)

            for i, node in enumerate(nodes_in_level):
                angle = (2 * math.pi * i) / count
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                pos[node] = np.array([x, y], dtype=float)

        unplaced = set(G.nodes()) - set(pos.keys())
        if unplaced:
            outer_radius = (max_dist + 1) * radius_step
            unplaced_list = list(unplaced)
            count = len(unplaced_list)
            for i, node in enumerate(unplaced_list):
                angle = (2 * math.pi * i) / count
                x = outer_radius * math.cos(angle)
                y = outer_radius * math.sin(angle)
                pos[node] = np.array([x, y], dtype=float)

        return pos

    def low_crossing_layout_auto(self, G, prefer_edge_order='degprod', refine_iterations=60, coarse_random_seed=None):
        if coarse_random_seed is not None:
            random.seed(coarse_random_seed)
            np.random.seed(coarse_random_seed)

        nodes = list(G.nodes())
        S = nx.Graph()
        S.add_nodes_from(nodes)

        edges = list(G.edges())
        degs = dict(G.degree())

        if prefer_edge_order == 'degprod':
            edges.sort(key=lambda e: (degs.get(e[0], 0) * degs.get(e[1], 0)), reverse=True)
        elif prefer_edge_order == 'degsum':
            edges.sort(key=lambda e: (degs.get(e[0], 0) + degs.get(e[1], 0)), reverse=True)
        else:
            random.shuffle(edges)

        for (u, v) in edges:
            S.add_edge(u, v)
            is_planar, _embedding = nx.check_planarity(S)
            if not is_planar:
                S.remove_edge(u, v)

        if S.number_of_edges() == 0 or S.number_of_nodes() == 0:
            pos = nx.spring_layout(G, iterations=refine_iterations)
            return {n: tuple(pos[n]) for n in G.nodes()}

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
            if len(cycle) < 3: return cycle
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
                if found is None or found == start: break
                ordered.append(found)
                prev, cur = cur, found
                if len(ordered) > len(cycle) + 5: break
            if len(ordered) == len(cycle): return ordered
            return cycle

        boundary = make_cycle_ordered(boundary, S)
        if len(boundary) < 3:
            pos = nx.spring_layout(G, iterations=refine_iterations)
            return {n: tuple(pos[n]) for n in G.nodes()}

        boundary_set = set(boundary)
        interior = [v for v in S.nodes() if v not in boundary_set]
        if len(interior) == 0:
            pos = {}
            R, L = 1.0, len(boundary)
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

        rows, cols, data = [], [], []
        bx = np.zeros(n_in, dtype=float)
        by = np.zeros(n_in, dtype=float)

        for i, v in enumerate(interior):
            nbrs = list(S.neighbors(v))
            degv = len(nbrs)
            rows.append(i);
            cols.append(i);
            data.append(degv)
            for w in nbrs:
                if w in boundary_set:
                    bx[i] += boundary_pos[w][0]
                    by[i] += boundary_pos[w][1]
                else:
                    j = idx[w]
                    rows.append(i);
                    cols.append(j);
                    data.append(-1.0)

        A_csr = sparse.csr_matrix((data, (rows, cols)), shape=(n_in, n_in))

        try:
            sol_x = spsolve(A_csr, bx)
            sol_y = spsolve(A_csr, by)
        except np.linalg.LinAlgError:
            A_dense = A_csr.toarray()
            sol_x, *_ = np.linalg.lstsq(A_dense, bx, rcond=None)
            sol_y, *_ = np.linalg.lstsq(A_dense, by, rcond=None)

        pos = {}
        for v, i in idx.items(): pos[v] = np.array([sol_x[i], sol_y[i]], dtype=float)
        for v in boundary: pos[v] = boundary_pos[v].copy()
        for v in G.nodes():
            if v not in pos: pos[v] = np.array([0.0, 0.0], dtype=float)

        try:
            pos_refined = nx.spring_layout(G, pos=pos, fixed=list(boundary), iterations=max(10, refine_iterations))
            pos = {v: np.array(pos_refined[v], dtype=float) for v in G.nodes()}
        except Exception:
            for v in G.nodes():
                if v not in pos or np.allclose(pos[v], 0.0):
                    pos[v] = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], dtype=float)

        xs, ys = np.array([p[0] for p in pos.values()]), np.array([p[1] for p in pos.values()])
        minx, maxx, miny, maxy = xs.min(), xs.max(), ys.min(), ys.max()
        span = max(maxx - minx, maxy - miny)
        if span < 1e-8: span = 1.0
        scale = 1.0 / span
        out = {}
        for v, p in pos.items():
            out[v] = [(p[0] - (minx + maxx) / 2.0) * scale, (p[1] - (miny + maxy) / 2.0) * scale]

        return out
