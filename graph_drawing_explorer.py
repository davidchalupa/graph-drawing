import sys
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QFileDialog, QProgressDialog, QPushButton, QHBoxLayout, QVBoxLayout, QWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter, QAction
import math
import random
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve

from analyze_graph.dominating_set import minimum_dominating_set_ilp


class DominatingSetThread(QThread):
    finished_computing = pyqtSignal(list)

    def __init__(self, G):
        super().__init__()
        self.G = G

    def run(self):
        if not self.G or not self.G.nodes:
            self.finished_computing.emit([])
            return

        # Trigger the algorithm
        dom_set = self.compute_algorithm_ilp(self.G)
        self.finished_computing.emit(dom_set if dom_set is not None else [])

    def compute_algorithm_ilp(self, G):
        """
        Placeholder function for your Dominating Set Algorithm 1.
        """
        print("Computing minimum dominating set of the graph...")

        mds = minimum_dominating_set_ilp(G, timeLimit=120)

        if mds is not None:
            print(f"Found a dominating set of size {len(mds)}.")
            return mds
        else:
            print(f"No feasible solution found.")
            return None


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
                pos[node][0] *= 150
                pos[node][1] *= 150
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

        for (u, v) in edges:
            S.add_edge(u, v)
            is_planar, _embedding = nx.check_planarity(S)
            if not is_planar:
                S.remove_edge(u, v)

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

        A_csr = sparse.csr_matrix((data, (rows, cols)), shape=(n_in, n_in))

        try:
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

        # --- 4) refinement ---
        try:
            pos_refined = nx.spring_layout(G, pos=pos, fixed=list(boundary), iterations=max(10, refine_iterations))
            pos = {v: np.array(pos_refined[v], dtype=float) for v in G.nodes()}
        except Exception:
            for v in G.nodes():
                if v not in pos or np.allclose(pos[v], 0.0):
                    pos[v] = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], dtype=float)

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

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(QColor("#0d0d0d"))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

    def display_graph(self, G, pos, highlight_nodes=None):
        self.scene.clear()
        if not G or not pos:
            return

        if highlight_nodes is None:
            highlight_nodes = []
        highlight_set = set(highlight_nodes)

        edge_pen = QPen(QColor(80, 80, 80, 100), 1)
        for u, v in G.edges():
            if u in pos and v in pos:
                self.scene.addLine(pos[u][0], pos[u][1], pos[v][0], pos[v][1], edge_pen)

        normal_brush = QBrush(QColor("#00f2ff"))
        highlight_brush = QBrush(QColor("#FF8C00"))  # Bright orange for highlighting
        node_pen = QPen(Qt.GlobalColor.black, 1)

        for node in G.nodes():
            if node in pos:
                x, y = pos[node]
                is_highlighted = node in highlight_set

                brush = highlight_brush if is_highlighted else normal_brush
                radius = 7 if is_highlighted else 5
                diameter = radius * 2

                ellipse = self.scene.addEllipse(x - radius, y - radius, diameter, diameter, node_pen, brush)
                ellipse.setZValue(2 if is_highlighted else 1)

        rect = self.scene.itemsBoundingRect()
        margin = 50
        self.setSceneRect(rect.adjusted(-margin, -margin, margin, margin))
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

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
        self.current_pos = None

        self.dominating_set = []
        self.show_dominating_set = False

        self.canvas = GraphCanvas()
        self.setCentralWidget(self.canvas)
        self._setup_overlay_buttons()
        self._setup_menu()
        self.showMaximized()

    def _setup_overlay_buttons(self):
        self.overlay_panel = QWidget(self)
        v_layout = QVBoxLayout(self.overlay_panel)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(10)

        # Toggle Button Row
        self.btn_toggle_ds = QPushButton("👁 Show Dominating Set")
        self.btn_toggle_ds.setCheckable(True)
        self.btn_toggle_ds.setEnabled(False)  # Disabled until computed
        self.btn_toggle_ds.clicked.connect(self.toggle_dominating_set)

        h_toggle_layout = QHBoxLayout()
        h_toggle_layout.addWidget(self.btn_toggle_ds)
        h_toggle_layout.addStretch()
        v_layout.addLayout(h_toggle_layout)

        # Layout Buttons Row
        self.btn_pca = QPushButton("📉 HDE (PCA) layout")
        self.btn_pca.clicked.connect(lambda: self.run_layout("pca"))

        self.btn_spring = QPushButton("🕸 Spring layout")
        self.btn_spring.clicked.connect(lambda: self.run_layout("spring"))

        self.btn_lowcross = QPushButton("📐 Low-crossing")
        self.btn_lowcross.clicked.connect(lambda: self.run_layout("lowcross"))

        h_layout_btns = QHBoxLayout()
        h_layout_btns.addWidget(self.btn_pca)
        h_layout_btns.addWidget(self.btn_spring)
        h_layout_btns.addWidget(self.btn_lowcross)
        h_layout_btns.addStretch()
        v_layout.addLayout(h_layout_btns)

        self.overlay_panel.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a;
                color: #00f2ff;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 10px 15px;
                font-weight: bold;
            }
            QPushButton:hover { 
                background-color: #333333; 
                border: 1px solid #00f2ff; 
            }
            /* Styling for active/checked state */
            QPushButton:checked, QPushButton[active="true"] {
                background-color: #00f2ff;
                color: #1a1a1a;
                border: 1px solid #00f2ff;
            }
        """)
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

        # File Menu
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open Graph...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        # Compute Menu
        compute_menu = menu_bar.addMenu("Compute")

        # Dominating set Sub-menu
        dom_set_menu = compute_menu.addMenu("Dominating set")

        # Algorithm 1 Action
        algo1_action = QAction("ILP solution", self)
        algo1_action.triggered.connect(self.run_dominating_set)
        dom_set_menu.addAction(algo1_action)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Graph", "", "Graph Files (*.col *.graphml *.gml)")
        if path:
            self.current_graph = load_from_col_file(path)

            # Reset dominating set logic on new file load
            self.dominating_set = []
            self.show_dominating_set = False
            self.btn_toggle_ds.setEnabled(False)
            self.btn_toggle_ds.setChecked(False)

            self.run_layout("pca")

    def run_layout(self, mode):
        if not self.current_graph: return

        # Update styling to highlight current layout
        buttons = {
            "pca": self.btn_pca,
            "spring": self.btn_spring,
            "lowcross": self.btn_lowcross
        }
        for key, btn in buttons.items():
            btn.setProperty("active", key == mode)
            # Re-evaluate stylesheet
            btn.style().unpolish(btn)
            btn.style().polish(btn)

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
        self.current_pos = pos
        self.redraw_graph()

    def run_dominating_set(self):
        if not self.current_graph:
            return

        msg = "Computing dominating set using ILP solution..."
        self.ds_progress = QProgressDialog(msg, None, 0, 0, self)
        self.ds_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.ds_progress.setWindowTitle("Computing")
        self.ds_progress.show()

        self.ds_thread = DominatingSetThread(self.current_graph)
        self.ds_thread.finished_computing.connect(self.on_dominating_set_finished)
        self.ds_thread.start()

    def on_dominating_set_finished(self, dom_set):
        self.ds_progress.accept()

        if dom_set:
            self.dominating_set = dom_set

            # Auto-enable and turn on highlighting
            self.btn_toggle_ds.setEnabled(True)
            self.btn_toggle_ds.setChecked(True)
            self.show_dominating_set = True

            self.redraw_graph()
            print(f"Dominating set computation completed! Dominating set size: {len(self.dominating_set)}")

    def toggle_dominating_set(self):
        self.show_dominating_set = self.btn_toggle_ds.isChecked()
        self.redraw_graph()

    def redraw_graph(self):
        """Redraws current pos using the current show_dominating_set state."""
        highlight = self.dominating_set if self.show_dominating_set else None
        self.canvas.display_graph(self.current_graph, self.current_pos, highlight_nodes=highlight)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
