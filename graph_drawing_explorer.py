import sys
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA

# --- UPDATED IMPORTS ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QFileDialog, QProgressDialog, QPushButton, QHBoxLayout, QVBoxLayout, QWidget,
    QGraphicsItem, QGraphicsPathItem, QStackedWidget, QDialog, QFormLayout,
    QSpinBox, QDialogButtonBox, QMessageBox, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QLineF, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter, QAction, QPainterPath, QImage, QPixmap
# -----------------------

import math
import random
from scipy import sparse
from scipy.sparse.linalg import spsolve

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import QPointF

from analyze_graph.dominating_set import minimum_dominating_set_ilp
from analyze_graph.max_clique import get_large_clique_greedy
from analyze_graph.max_clique import get_large_clique_bopp_hald
from analyze_graph.max_clique import get_max_clique_bnb


class DominatingSetThread(QThread):
    finished_computing = pyqtSignal(list)

    def __init__(self, G):
        super().__init__()
        self.G = G

    def run(self):
        if not self.G or not self.G.nodes:
            self.finished_computing.emit([])
            return
        dom_set = self.compute_algorithm_ilp(self.G)
        self.finished_computing.emit(dom_set if dom_set is not None else [])

    def compute_algorithm_ilp(self, G):
        print("Computing minimum dominating set of the graph...")
        mds = minimum_dominating_set_ilp(G, timeLimit=120)
        if mds is not None:
            print(f"Found a dominating set of size {len(mds)}.")
            return mds
        else:
            print(f"No feasible solution found.")
            return None


class CliqueThread(QThread):
    finished_computing = pyqtSignal(list)

    def __init__(self, G, alg):
        super().__init__()
        self.G = G
        self.alg = alg

    def run(self):
        if not self.G or not self.G.nodes:
            self.finished_computing.emit([])
            return

        if self.alg == "bopp_hald":
            clique = self.compute_algorithm_hald(self.G)
        elif self.alg == "greedy":
            clique = self.compute_algorithm_greedy(self.G)
        elif self.alg == "exact":
            clique = self.compute_algorithm_exact(self.G)
        else:
            raise ValueError("Invalid clique algorithm")
        self.finished_computing.emit(clique if clique is not None else [])

    def compute_algorithm_hald(self, G):
        print("Computing a large clique in the graph...")
        clique = get_large_clique_bopp_hald(G)
        if clique is not None:
            print(f"Found a clique of size {len(clique)}.")
            return clique
        return None

    def compute_algorithm_greedy(self, G):
        print("Computing a large clique in the graph...")
        clique = get_large_clique_greedy(G)
        if clique is not None:
            print(f"Found a clique of size {len(clique)}.")
            return clique
        return None

    def compute_algorithm_exact(self, G):
        print("Computing the maximum clique in the graph...")
        clique = get_max_clique_bnb(G)
        if clique is not None:
            print(f"Found a clique of size {len(clique)}.")
            return clique
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
        elif self.mode == "lowcross":
            pos = self.low_crossing_layout_auto(self.G)
        elif self.mode == "radial":
            pos = self.compute_radial_layout(self.G)
        else:
            pos = nx.spring_layout(self.G, scale=1000)

        seen_positions = {}
        jitter_amount = 0.1

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


class FastLineItem(QGraphicsItem):
    def __init__(self, lines, pen, bounding_rect, z_value=0):
        super().__init__()
        self.lines = lines
        self._pen = pen
        self._boundingRect = bounding_rect
        self.setZValue(z_value)

    def boundingRect(self):
        return self._boundingRect

    def paint(self, painter, option, widget):
        view = widget.parent()
        if isinstance(view, GraphCanvasOptimized) and view.is_interacting and self.zValue() == 0:
            return
        lod = option.levelOfDetailFromTransform(painter.worldTransform())
        if lod < 0.15 and self.zValue() == 0:
            return
        painter.setPen(self._pen)
        painter.drawLines(self.lines)


class FastNodeItem(QGraphicsItem):
    def __init__(self, points, color, radius, bounding_rect, z_value=0, is_highlight=False):
        super().__init__()
        self.points = points
        self.color = QColor(color)
        self.radius = radius
        self._boundingRect = bounding_rect
        self.setZValue(z_value)
        self.is_highlight = is_highlight

    def boundingRect(self):
        return self._boundingRect

    def paint(self, painter, option, widget):
        lod = option.levelOfDetailFromTransform(painter.worldTransform())
        if lod < 0.5 and not self.is_highlight:
            pen = QPen(self.color, self.radius * 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPoints(self.points)
        else:
            painter.setPen(QPen(Qt.GlobalColor.black, 0))
            painter.setBrush(QBrush(self.color))
            r = self.radius
            for p in self.points:
                painter.drawEllipse(p, r, r)


class GraphCanvasOptimized(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.is_interacting = False
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.gl_widget = QOpenGLWidget()
        self.setViewport(self.gl_widget)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(QColor("#0d0d0d"))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing)

        self.scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState)

    def mousePressEvent(self, event):
        self.is_interacting = True
        self.viewport().update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.is_interacting = False
        self.viewport().update()
        super().mouseReleaseEvent(event)

    def display_graph(self, G, pos, dom_nodes=None, clique_nodes=None):
        self.scene.clear()
        if not G or not pos:
            return

        dom_set = set(dom_nodes) if dom_nodes else set()
        clique_set = set(clique_nodes) if clique_nodes else set()

        default_edge_pen = QPen(QColor(80, 80, 80, 100), 0)
        clique_edge_pen = QPen(QColor("#00FF00"), 2)
        clique_edge_pen.setCosmetic(True)

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        if xs:
            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
            scene_rect = QRectF(min_x - 50, min_y - 50, max_x - min_x + 100, max_y - min_y + 100)
        else:
            scene_rect = QRectF(0, 0, 100, 100)

        default_lines = []
        clique_lines = []

        for u, v in G.edges():
            pu = pos.get(u)
            pv = pos.get(v)
            if pu is not None and pv is not None:
                line = QLineF(pu[0], pu[1], pv[0], pv[1])
                if u in clique_set and v in clique_set:
                    clique_lines.append(line)
                else:
                    default_lines.append(line)

        if default_lines:
            self.scene.addItem(FastLineItem(default_lines, default_edge_pen, scene_rect, z_value=0))
        if clique_lines:
            self.scene.addItem(FastLineItem(clique_lines, clique_edge_pen, scene_rect, z_value=1))

        pts_normal, pts_dom, pts_clique, pts_both = [], [], [], []

        for node in G.nodes():
            p = pos.get(node)
            if p is not None:
                qpf = QPointF(p[0], p[1])
                is_dom = node in dom_set
                is_clique = node in clique_set

                if is_dom and is_clique:
                    pts_both.append(qpf)
                elif is_clique:
                    pts_clique.append(qpf)
                elif is_dom:
                    pts_dom.append(qpf)
                else:
                    pts_normal.append(qpf)

        if pts_normal:
            self.scene.addItem(FastNodeItem(pts_normal, "#00f2ff", 5, scene_rect, z_value=2))
        if pts_dom:
            self.scene.addItem(FastNodeItem(pts_dom, "#FF8C00", 7, scene_rect, z_value=3, is_highlight=True))
        if pts_clique:
            self.scene.addItem(FastNodeItem(pts_clique, "#00FF00", 8, scene_rect, z_value=3, is_highlight=True))
        if pts_both:
            self.scene.addItem(FastNodeItem(pts_both, "#FFFFFF", 8, scene_rect, z_value=4, is_highlight=True))

        self.setSceneRect(scene_rect)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)


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

    def display_graph(self, G, pos, dom_nodes=None, clique_nodes=None):
        self.scene.clear()
        if not G or not pos:
            return

        dom_set = set(dom_nodes) if dom_nodes else set()
        clique_set = set(clique_nodes) if clique_nodes else set()

        default_edge_pen = QPen(QColor(80, 80, 80, 100), 1)
        clique_edge_pen = QPen(QColor("#00FF00"), 2)

        normal_brush = QBrush(QColor("#00f2ff"))
        dom_brush = QBrush(QColor("#FF8C00"))
        clique_brush = QBrush(QColor("#00FF00"))
        both_brush = QBrush(QColor("#FFFFFF"))
        node_pen = QPen(Qt.GlobalColor.black, 1)

        for u, v in G.edges():
            if u in pos and v in pos:
                if u in clique_set and v in clique_set:
                    pen = clique_edge_pen
                    z_val = 1
                else:
                    pen = default_edge_pen
                    z_val = 0

                line = self.scene.addLine(pos[u][0], pos[u][1], pos[v][0], pos[v][1], pen)
                line.setZValue(z_val)

        for node in G.nodes():
            if node in pos:
                x, y = pos[node]
                is_dom = node in dom_set
                is_clique = node in clique_set

                if is_dom and is_clique:
                    brush = both_brush
                    radius = 8
                elif is_clique:
                    brush = clique_brush
                    radius = 8
                elif is_dom:
                    brush = dom_brush
                    radius = 7
                else:
                    brush = normal_brush
                    radius = 5

                diameter = radius * 2
                ellipse = self.scene.addEllipse(x - radius, y - radius, diameter, diameter, node_pen, brush)
                ellipse.setZValue(3 if (is_dom or is_clique) else 2)

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

        self.clique = []
        self.show_clique = False

        self.stacked_widget = QStackedWidget()
        self.canvas_standard = GraphCanvas()
        self.canvas_optimized = GraphCanvasOptimized()
        self.stacked_widget.addWidget(self.canvas_standard)
        self.stacked_widget.addWidget(self.canvas_optimized)
        self.setCentralWidget(self.stacked_widget)
        self.canvas = self.canvas_standard

        self._setup_overlay_buttons()
        self._setup_menu()
        self.showMaximized()

    def switch_canvas(self, optimized: bool):
        if optimized:
            self.stacked_widget.setCurrentWidget(self.canvas_optimized)
            self.canvas = self.canvas_optimized
        else:
            self.stacked_widget.setCurrentWidget(self.canvas_standard)
            self.canvas = self.canvas_standard

        if hasattr(self, 'overlay_container'):
            self.overlay_container.raise_()

    def _setup_overlay_buttons(self):
        self.overlay_container = QWidget(self)

        main_layout = QVBoxLayout(self.overlay_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        main_layout.addSpacing(30)

        # 1. Expand/Collapse Button
        self.btn_collapse = QPushButton("▶ Hide")
        self.btn_collapse.setToolTip("Toggle Menu Visibility")
        self.btn_collapse.clicked.connect(self.toggle_menu_visibility)
        main_layout.addWidget(self.btn_collapse, alignment=Qt.AlignmentFlag.AlignRight)

        # 2. Main Buttons Panel
        self.button_panel = QWidget()
        grid_layout = QGridLayout(self.button_panel)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(10)

        # Toggles (Column 0)
        self.btn_toggle_ds = QPushButton("👁")
        self.btn_toggle_ds.setToolTip("Show Dominating Set")
        self.btn_toggle_ds.setCheckable(True)
        self.btn_toggle_ds.setEnabled(False)
        self.btn_toggle_ds.clicked.connect(self.toggle_dominating_set)

        self.btn_toggle_cl = QPushButton("⭐")
        self.btn_toggle_cl.setToolTip("Show Clique")
        self.btn_toggle_cl.setCheckable(True)
        self.btn_toggle_cl.setEnabled(False)
        self.btn_toggle_cl.clicked.connect(self.toggle_clique)

        # Layout Modes (Column 1)
        self.btn_radial = QPushButton("🌀")
        self.btn_radial.setToolTip("Radial layout")
        self.btn_radial.clicked.connect(lambda: self.run_layout("radial"))

        self.btn_pca = QPushButton("📉")
        self.btn_pca.setToolTip("HDE (PCA) layout")
        self.btn_pca.clicked.connect(lambda: self.run_layout("pca"))

        self.btn_spring = QPushButton("🕸")
        self.btn_spring.setToolTip("Spring layout")
        self.btn_spring.clicked.connect(lambda: self.run_layout("spring"))

        self.btn_lowcross = QPushButton("📐")
        self.btn_lowcross.setToolTip("Low-crossing layout")
        self.btn_lowcross.clicked.connect(lambda: self.run_layout("lowcross"))

        self.btn_matrix = QPushButton("▦")
        self.btn_matrix.setToolTip("Adjacency Matrix")
        self.btn_matrix.clicked.connect(lambda: self.run_layout("matrix"))

        # Add to Grid
        grid_layout.addWidget(self.btn_toggle_ds, 0, 0)
        grid_layout.addWidget(self.btn_toggle_cl, 1, 0)

        grid_layout.addWidget(self.btn_radial, 0, 1)
        grid_layout.addWidget(self.btn_pca, 1, 1)
        grid_layout.addWidget(self.btn_spring, 2, 1)
        grid_layout.addWidget(self.btn_lowcross, 3, 1)
        grid_layout.addWidget(self.btn_matrix, 4, 1)

        grid_layout.setRowStretch(5, 1)  # Push everything up
        main_layout.addWidget(self.button_panel)

        self.overlay_container.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a;
                color: #00f2ff;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 10px;
                font-weight: bold;
                font-size: 20px;
                min-width: 45px;
                min-height: 45px;
            }
            QPushButton#collapseBtn {
                font-size: 14px;
                min-height: 30px;
                min-width: 60px;
                padding: 5px 10px;
            }
            QPushButton:hover { 
                background-color: #333333; 
                border: 1px solid #00f2ff; 
            }
            QPushButton:disabled {
                color: #555555;
                border: 1px solid #333333;
                background-color: #111111;
            }
            QPushButton:checked, QPushButton[active="true"] {
                background-color: #00f2ff;
                color: #1a1a1a;
                border: 1px solid #00f2ff;
            }
        """)
        self.btn_collapse.setObjectName("collapseBtn")
        self.overlay_container.show()
        self.overlay_container.raise_()

    def toggle_menu_visibility(self):
        is_visible = self.button_panel.isVisible()
        self.button_panel.setVisible(not is_visible)
        self.btn_collapse.setText("◀ Show" if is_visible else "▶ Hide")
        self._update_overlay_position()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_overlay_position()

    def _update_overlay_position(self):
        if hasattr(self, 'overlay_container'):
            self.overlay_container.adjustSize()
            w = self.overlay_container.width()
            h = self.overlay_container.height()
            # Position at the top right, 20px padding
            self.overlay_container.setGeometry(self.width() - w - 20, 20, w, h)

    def _setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open Graph...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        gen_sf_action = QAction("Generate Scale-free network...", self)
        gen_sf_action.triggered.connect(self.generate_scale_free)
        file_menu.addAction(gen_sf_action)

        compute_menu = menu_bar.addMenu("Compute")

        dom_set_menu = compute_menu.addMenu("Dominating set")
        algo1_action = QAction("ILP solution", self)
        algo1_action.triggered.connect(self.run_dominating_set)
        dom_set_menu.addAction(algo1_action)

        clique_menu = compute_menu.addMenu("Maximum clique")
        algo_cl1_action = QAction("Greedy heuristic", self)
        algo_cl1_action.triggered.connect(self.run_clique_greedy)
        algo_cl2_action = QAction("Boppana-Halldórsson heuristic", self)
        algo_cl2_action.triggered.connect(self.run_clique_bopp_hald)
        algo_cl3_action = QAction("Branch and bound", self)
        algo_cl3_action.triggered.connect(self.run_clique_exact)
        clique_menu.addAction(algo_cl1_action)
        clique_menu.addAction(algo_cl2_action)
        clique_menu.addAction(algo_cl3_action)

    def generate_scale_free(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Generate Scale-free Network")
        layout = QFormLayout(dialog)

        n_spin = QSpinBox()
        n_spin.setRange(1, 100000)
        n_spin.setValue(500)

        m_spin = QSpinBox()
        m_spin.setRange(1, 1000)
        m_spin.setValue(2)

        layout.addRow("Number of vertices (n):", n_spin)
        layout.addRow("Edges per new vertex (m):", m_spin)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            n = n_spin.value()
            m = m_spin.value()
            if m >= n:
                QMessageBox.warning(self, "Invalid Parameters", "m must be strictly less than n.")
                return
            self.current_graph = nx.barabasi_albert_graph(n, m)
            self.setup_new_graph()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Graph", "", "Graph Files (*.col *.graphml *.gml)")
        if path:
            self.current_graph = load_from_col_file(path)
            self.setup_new_graph()

    def setup_new_graph(self):
        self.dominating_set = []
        self.show_dominating_set = False
        self.btn_toggle_ds.setEnabled(False)
        self.btn_toggle_ds.setChecked(False)

        self.clique = []
        self.show_clique = False
        self.btn_toggle_cl.setEnabled(False)
        self.btn_toggle_cl.setChecked(False)

        num_nodes = self.current_graph.number_of_nodes()
        num_edges = self.current_graph.number_of_edges()

        if num_nodes > 4000 or num_edges > 10000:
            self.switch_canvas(optimized=True)
        else:
            self.switch_canvas(optimized=False)

        if num_nodes > 2500 or num_edges > 10000:
            self.btn_spring.setEnabled(False)
            self.btn_spring.setToolTip("Spring layout disabled (Graph too large)")
        else:
            self.btn_spring.setEnabled(True)
            self.btn_spring.setToolTip("Spring layout")

        if num_nodes > 1250 or num_edges > 5000:
            self.btn_lowcross.setEnabled(False)
            self.btn_lowcross.setToolTip("Low-crossing layout disabled (Graph too large)")
        else:
            self.btn_lowcross.setEnabled(True)
            self.btn_lowcross.setToolTip("Low-crossing layout")

        if num_nodes > 10000:
            self.btn_matrix.setEnabled(False)
            self.btn_matrix.setToolTip("Adjacency Matrix disabled (Graph too large - > 10k nodes)")
        else:
            self.btn_matrix.setEnabled(True)
            self.btn_matrix.setToolTip("Adjacency Matrix")

        self.run_layout("radial")

    def run_layout(self, mode):
        if not self.current_graph: return

        buttons = {"pca": self.btn_pca, "spring": self.btn_spring, "lowcross": self.btn_lowcross,
                   "radial": self.btn_radial, "matrix": self.btn_matrix}
        for key, btn in buttons.items():
            btn.setProperty("active", key == mode)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        if mode == "matrix":
            self.show_adjacency_matrix()
            return

        msg = "Calculating graph layout. This may take a moment..."
        self.progress = QProgressDialog(msg, None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setWindowTitle("Progress")
        self.progress.show()

        self.layout_thread = LayoutThread(self.current_graph, mode)
        self.layout_thread.layout_finished.connect(self.on_layout_finished)
        self.layout_thread.start()

    def show_adjacency_matrix(self):
        G = self.current_graph
        n = G.number_of_nodes()

        if n > 10000:
            QMessageBox.warning(self, "Too Large", "Graph is too large to render as a bitmap (>10,000 nodes).")
            return

        img = QImage(n, n, QImage.Format.Format_RGB32)
        img.fill(Qt.GlobalColor.black)  # Black background

        node_list = list(G.nodes())
        node_idx = {node: i for i, node in enumerate(node_list)}

        fg_color = QColor("#00f2ff")  # Contrasting cyan foreground

        for u, v in G.edges():
            i, j = node_idx[u], node_idx[v]
            img.setPixelColor(i, j, fg_color)
            img.setPixelColor(j, i, fg_color)

        pixmap = QPixmap.fromImage(img)
        self.canvas.scene.clear()
        item = self.canvas.scene.addPixmap(pixmap)

        rect = item.boundingRect()
        self.canvas.setSceneRect(rect.adjusted(-50, -50, 50, 50))
        self.canvas.fitInView(self.canvas.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def on_layout_finished(self, pos):
        if hasattr(self, 'progress'):
            self.progress.close()
        self.current_pos = pos
        self.canvas.display_graph(
            self.current_graph,
            self.current_pos,
            self.dominating_set if self.show_dominating_set else None,
            self.clique if self.show_clique else None
        )

    def toggle_dominating_set(self):
        self.show_dominating_set = self.btn_toggle_ds.isChecked()
        if self.current_pos:
            self.on_layout_finished(self.current_pos)

    def toggle_clique(self):
        self.show_clique = self.btn_toggle_cl.isChecked()
        if self.current_pos:
            self.on_layout_finished(self.current_pos)

    def run_dominating_set(self):
        if not self.current_graph: return
        self.ds_thread = DominatingSetThread(self.current_graph)
        self.ds_thread.finished_computing.connect(self.on_ds_finished)
        self.ds_thread.start()

    def on_ds_finished(self, ds):
        self.dominating_set = ds
        if ds:
            self.btn_toggle_ds.setEnabled(True)
            self.btn_toggle_ds.setChecked(True)
            self.show_dominating_set = True
            if self.current_pos: self.on_layout_finished(self.current_pos)

    def run_clique_greedy(self):
        self._run_clique("greedy")

    def run_clique_bopp_hald(self):
        self._run_clique("bopp_hald")

    def run_clique_exact(self):
        self._run_clique("exact")

    def _run_clique(self, alg):
        if not self.current_graph: return
        self.cl_thread = CliqueThread(self.current_graph, alg)
        self.cl_thread.finished_computing.connect(self.on_clique_finished)
        self.cl_thread.start()

    def on_clique_finished(self, clique):
        self.clique = clique
        if clique:
            self.btn_toggle_cl.setEnabled(True)
            self.btn_toggle_cl.setChecked(True)
            self.show_clique = True
            if self.current_pos: self.on_layout_finished(self.current_pos)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
