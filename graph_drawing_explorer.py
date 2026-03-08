import sys
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QFileDialog, QProgressDialog, QPushButton, QHBoxLayout, QWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter, QAction
import math
import random
from scipy import sparse
from scipy.sparse.linalg import spsolve

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
            for node in pos:
                pos[node][0] *= 200
                pos[node][1] *= 200
        elif self.mode == "lowcross":
            pos = self.low_crossing_layout_auto(self.G)
            for node in pos:
                pos[node][0] *= 2000
                pos[node][1] *= 2000
        else:
            pos = nx.spring_layout(self.G, scale=1000)

        # Apply landscape stretch (16:9)
        for node in pos:
            pos[node][0] *= 1.77

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

    def low_crossing_layout_auto(self, G,
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
            out[v] = [(p[0] - (minx + maxx) / 2.0) * scale, (p[1] - (miny + maxy) / 2.0) * scale]

        return out

class GraphCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Changed to Center for better fit
        self.setBackgroundBrush(QColor("#0d0d0d"))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

    def display_graph(self, G, pos):
        self.scene.clear()
        if not G or not pos:
            return

        edge_pen = QPen(QColor(80, 80, 80, 100), 1)
        for u, v in G.edges():
            if u in pos and v in pos:
                self.scene.addLine(pos[u][0], pos[u][1], pos[v][0], pos[v][1], edge_pen)

        node_brush = QBrush(QColor("#00f2ff"))
        node_pen = QPen(Qt.GlobalColor.black, 1)
        for node in G.nodes():
            if node in pos:
                x, y = pos[node]
                ellipse = self.scene.addEllipse(x - 5, y - 5, 10, 10, node_pen, node_brush)
                ellipse.setZValue(1)

        # --- KEY CHANGE START ---
        # Get the rectangle containing all items
        rect = self.scene.itemsBoundingRect()

        # Apply a small margin so nodes aren't touching the window edges
        margin = 50
        self.setSceneRect(rect.adjusted(-margin, -margin, margin, margin))

        # Scale the view to fit the graph within the current window size
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        # --- KEY CHANGE END ---

    def wheelEvent(self, event):
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)


def load_from_col_file(file_path):
    G = nx.Graph()
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith(('c', 'p')):
                    continue
                if line.startswith('e'):
                    parts = line.split()
                    u, v = int(parts[1]), int(parts[2])
                    if u != v:
                        G.add_edge(u, v)
    except Exception as e:
        print(f"File Load Error: {e}")
    return G


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network Graph Visualizer")
        self.current_graph = None
        self.canvas = GraphCanvas()
        self.setCentralWidget(self.canvas)
        self._setup_overlay_buttons()
        self._setup_menu()
        self.showMaximized()

    def _setup_overlay_buttons(self):
        self.overlay_panel = QWidget(self)
        layout = QHBoxLayout(self.overlay_panel)
        self.overlay_panel.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a;
                color: #00f2ff;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 10px 15px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #333333; border: 1px solid #00f2ff; }
        """)
        self.btn_pca = QPushButton("📉 HDE (PCA) layout")
        self.btn_pca.clicked.connect(lambda: self.run_layout("pca"))
        self.btn_spring = QPushButton("🕸 Spring layout")
        self.btn_spring.clicked.connect(lambda: self.run_layout("spring"))
        self.btn_lowcross = QPushButton("📐 Low-crossing")
        self.btn_lowcross.clicked.connect(lambda: self.run_layout("lowcross"))
        layout.addWidget(self.btn_pca)
        layout.addWidget(self.btn_spring)
        layout.addWidget(self.btn_lowcross)
        self.overlay_panel.show()
        self.overlay_panel.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'overlay_panel'):
            h = self.overlay_panel.sizeHint().height()
            w = self.overlay_panel.sizeHint().width()
            self.overlay_panel.setGeometry(20, self.height() - h - 60, w, h)

    def _setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open Graph...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Graph", "", "Graph Files (*.col *.graphml *.gml)")
        if path:
            self.current_graph = load_from_col_file(path)
            self.run_layout("pca")

    def run_layout(self, mode):
        if not self.current_graph: return
        msg = "Calculating graph layout. This may take a moment..."
        self.progress = QProgressDialog(msg, None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setWindowTitle("Progress")
        self.progress.show()
        self.layout_thread = LayoutThread(self.current_graph, mode)
        self.layout_thread.layout_finished.connect(self.on_layout_finished)
        self.layout_thread.start()

    def on_layout_finished(self, pos):
        self.progress.accept()
        self.canvas.display_graph(self.current_graph, pos)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
