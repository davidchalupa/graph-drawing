import sys
import networkx as nx

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QFileDialog, QProgressDialog, QPushButton, QVBoxLayout, QWidget,
    QGraphicsItem, QStackedWidget, QDialog, QFormLayout,
    QSpinBox, QDialogButtonBox, QMessageBox, QGridLayout
)
from PyQt6.QtCore import Qt, QLineF, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter, QAction, QImage, QPixmap, QFont


from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import QPointF

from dominating_set_thread import DominatingSetThread
from clique_thread import CliqueThread
from layout_thread import LayoutThread


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

    def display_graph(self, G, pos, dom_nodes=None, clique_nodes=None, node_labels=None, bridges=None):
        self.scene.clear()
        if not G or not pos:
            return

        dom_set = set(dom_nodes) if dom_nodes else set()
        clique_set = set(clique_nodes) if clique_nodes else set()

        # Use frozenset for undirected edge matching
        bridge_set = set()
        if bridges:
            for u, v in bridges:
                bridge_set.add(frozenset([u, v]))

        default_edge_pen = QPen(QColor(80, 80, 80, 100), 0)

        clique_edge_pen = QPen(QColor("#00FF00"), 2)
        clique_edge_pen.setCosmetic(True)

        bridge_edge_pen = QPen(QColor("#FF4444"), 2)
        bridge_edge_pen.setCosmetic(True)

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        if xs:
            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
            scene_rect = QRectF(min_x - 50, min_y - 50, max_x - min_x + 100, max_y - min_y + 100)
        else:
            scene_rect = QRectF(0, 0, 100, 100)

        default_lines = []
        clique_lines = []
        bridge_lines = []

        for u, v in G.edges():
            pu = pos.get(u)
            pv = pos.get(v)
            if pu is not None and pv is not None:
                line = QLineF(pu[0], pu[1], pv[0], pv[1])

                if frozenset([u, v]) in bridge_set:
                    bridge_lines.append(line)
                elif u in clique_set and v in clique_set:
                    clique_lines.append(line)
                else:
                    default_lines.append(line)

        if default_lines:
            self.scene.addItem(FastLineItem(default_lines, default_edge_pen, scene_rect, z_value=0))
        if clique_lines:
            self.scene.addItem(FastLineItem(clique_lines, clique_edge_pen, scene_rect, z_value=1))
        if bridge_lines:
            self.scene.addItem(FastLineItem(bridge_lines, bridge_edge_pen, scene_rect, z_value=2))

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

        # Shifted node Z-values up to account for the new edge layer
        if pts_normal:
            self.scene.addItem(FastNodeItem(pts_normal, "#00f2ff", 5, scene_rect, z_value=3))
        if pts_dom:
            self.scene.addItem(FastNodeItem(pts_dom, "#FF8C00", 7, scene_rect, z_value=4, is_highlight=True))
        if pts_clique:
            self.scene.addItem(FastNodeItem(pts_clique, "#00FF00", 8, scene_rect, z_value=4, is_highlight=True))
        if pts_both:
            self.scene.addItem(FastNodeItem(pts_both, "#FFFFFF", 8, scene_rect, z_value=5, is_highlight=True))

        # Note: Intentionally skipping `node_labels` rendering in the optimized canvas.
        # Rendering thousands of individual QGraphicsTextItems destroys OpenGL batching performance.

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

    def display_graph(self, G, pos, dom_nodes=None, clique_nodes=None, node_labels=None, bridges=None):
        self.scene.clear()
        if not G or not pos:
            return

        dom_set = set(dom_nodes) if dom_nodes else set()
        clique_set = set(clique_nodes) if clique_nodes else set()
        labels_dict = node_labels if node_labels else {}

        # Use frozenset for undirected edge matching
        bridge_set = set()
        if bridges:
            for u, v in bridges:
                bridge_set.add(frozenset([u, v]))

        default_edge_pen = QPen(QColor(80, 80, 80, 100), 1)
        clique_edge_pen = QPen(QColor("#00FF00"), 2)
        bridge_edge_pen = QPen(QColor("#FF4444"), 2)  # Distinct color for bridges

        normal_brush = QBrush(QColor("#00f2ff"))
        dom_brush = QBrush(QColor("#FF8C00"))
        clique_brush = QBrush(QColor("#00FF00"))
        both_brush = QBrush(QColor("#FFFFFF"))
        node_pen = QPen(Qt.GlobalColor.black, 1)

        for u, v in G.edges():
            if u in pos and v in pos:
                if frozenset([u, v]) in bridge_set:
                    pen = bridge_edge_pen
                    z_val = 2
                elif u in clique_set and v in clique_set:
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

                # Render computation text labels if active
                if node in labels_dict:
                    text_item = self.scene.addText(labels_dict[node])

                    font = QFont("Arial", 16)
                    font.setBold(True)
                    text_item.setFont(font)

                    text_item.setDefaultTextColor(QColor("#FFFFFF"))
                    text_item.setPos(x + radius, y - radius)
                    text_item.setZValue(4)

        rect = self.scene.itemsBoundingRect()
        margin = 50
        self.setSceneRect(rect.adjusted(-margin, -margin, margin, margin))
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)


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

        self.clustering_coeffs = {}
        self.show_clustering = False

        self.betweenness_cent = {}
        self.show_betweenness = False

        self.bridges = []
        self.show_bridges = False

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
        # NOTE: Make sure self.canvas_optimized is instantiated properly if you re-enable it.
        # if optimized:
        #     self.stacked_widget.setCurrentWidget(self.canvas_optimized)
        #     self.canvas = self.canvas_optimized
        # else:
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

        self.btn_toggle_cc = QPushButton("△")
        self.btn_toggle_cc.setToolTip("Show Clustering Coefficients")
        self.btn_toggle_cc.setCheckable(True)
        self.btn_toggle_cc.setEnabled(False)
        self.btn_toggle_cc.clicked.connect(self.toggle_clustering)

        self.btn_toggle_bc = QPushButton("⛬")
        self.btn_toggle_bc.setToolTip("Show Betweenness Centrality")
        self.btn_toggle_bc.setCheckable(True)
        self.btn_toggle_bc.setEnabled(False)
        self.btn_toggle_bc.clicked.connect(self.toggle_betweenness)

        self.btn_toggle_br = QPushButton("🔗")
        self.btn_toggle_br.setToolTip("Show Bridges")
        self.btn_toggle_br.setCheckable(True)
        self.btn_toggle_br.setEnabled(False)
        self.btn_toggle_br.clicked.connect(self.toggle_bridges)

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
        grid_layout.addWidget(self.btn_toggle_cc, 2, 0)
        grid_layout.addWidget(self.btn_toggle_bc, 3, 0)
        grid_layout.addWidget(self.btn_toggle_br, 4, 0)

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
        algo_cl3_action = QAction("Branch and bound", self)
        algo_cl3_action.triggered.connect(self.run_clique_exact)
        clique_menu.addAction(algo_cl1_action)
        clique_menu.addAction(algo_cl3_action)

        compute_menu.addSeparator()

        cc_action = QAction("Clustering coefficients", self)
        cc_action.triggered.connect(self.compute_clustering)
        compute_menu.addAction(cc_action)

        bc_action = QAction("Betweenness centrality", self)
        bc_action.triggered.connect(self.compute_betweenness)
        compute_menu.addAction(bc_action)

        compute_menu.addSeparator()

        br_action = QAction("Bridges", self)
        br_action.triggered.connect(self.compute_bridges)
        compute_menu.addAction(br_action)

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
            # Assumes load_from_col_file is imported or defined elsewhere
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

        self.clustering_coeffs = {}
        self.show_clustering = False
        self.btn_toggle_cc.setEnabled(False)
        self.btn_toggle_cc.setChecked(False)

        self.betweenness_cent = {}
        self.show_betweenness = False
        self.btn_toggle_bc.setEnabled(False)
        self.btn_toggle_bc.setChecked(False)

        self.bridges = []
        self.show_bridges = False
        self.btn_toggle_br.setEnabled(False)
        self.btn_toggle_br.setChecked(False)

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

        # Assumes LayoutThread is defined elsewhere
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
        img.fill(QColor("#0d0d0d"))  # Blends perfectly with canvas background

        node_list = list(G.nodes())
        node_idx = {node: i for i, node in enumerate(node_list)}

        fg_color = QColor("#00f2ff")

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
        if hasattr(self, 'progress') and self.progress is not None:
            self.progress.close()
            self.progress = None

        self.current_pos = pos
        node_labels = {}

        if self.show_clustering or self.show_betweenness:
            for n in self.current_graph.nodes():
                lbls = []
                if self.show_clustering and n in self.clustering_coeffs:
                    lbls.append(f"C: {self.clustering_coeffs[n]}")
                if self.show_betweenness and n in self.betweenness_cent:
                    lbls.append(f"B: {self.betweenness_cent[n]}")
                if lbls:
                    node_labels[n] = "\n".join(lbls)

        # Added bridges kwarg mapping here
        self.canvas.display_graph(
            self.current_graph,
            self.current_pos,
            self.dominating_set if self.show_dominating_set else None,
            self.clique if self.show_clique else None,
            node_labels=node_labels,
            bridges=self.bridges if self.show_bridges else None
        )

    def toggle_dominating_set(self):
        self.show_dominating_set = self.btn_toggle_ds.isChecked()
        if self.current_pos:
            self.on_layout_finished(self.current_pos)

    def toggle_clique(self):
        self.show_clique = self.btn_toggle_cl.isChecked()
        if self.current_pos:
            self.on_layout_finished(self.current_pos)

    def toggle_clustering(self):
        self.show_clustering = self.btn_toggle_cc.isChecked()
        if self.current_pos:
            self.on_layout_finished(self.current_pos)

    def toggle_betweenness(self):
        self.show_betweenness = self.btn_toggle_bc.isChecked()
        if self.current_pos:
            self.on_layout_finished(self.current_pos)

    def toggle_bridges(self):
        self.show_bridges = self.btn_toggle_br.isChecked()
        if self.current_pos:
            self.on_layout_finished(self.current_pos)

    def compute_clustering(self):
        if not self.current_graph: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            raw_cc = nx.clustering(self.current_graph)
            self.clustering_coeffs = {n: round(v, 3) for n, v in raw_cc.items()}
            self.btn_toggle_cc.setEnabled(True)
            self.btn_toggle_cc.setChecked(True)
            self.show_clustering = True
            if self.current_pos: self.on_layout_finished(self.current_pos)
        finally:
            QApplication.restoreOverrideCursor()

    def compute_betweenness(self):
        if not self.current_graph: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            raw_bc = nx.betweenness_centrality(self.current_graph)
            self.betweenness_cent = {n: round(v, 4) for n, v in raw_bc.items()}
            self.btn_toggle_bc.setEnabled(True)
            self.btn_toggle_bc.setChecked(True)
            self.show_betweenness = True
            if self.current_pos: self.on_layout_finished(self.current_pos)
        finally:
            QApplication.restoreOverrideCursor()

    def compute_bridges(self):
        if not self.current_graph: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # nx.bridges implements a Tarjan-like DFS biconnected component algorithm logic
            self.bridges = list(nx.bridges(self.current_graph))

            self.btn_toggle_br.setEnabled(True)
            self.btn_toggle_br.setChecked(True)
            self.show_bridges = True

            if self.current_pos:
                self.on_layout_finished(self.current_pos)
        finally:
            QApplication.restoreOverrideCursor()

    def run_dominating_set(self):
        if not self.current_graph: return
        # Assumes DominatingSetThread is defined elsewhere
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

    def run_clique_exact(self):
        self._run_clique("exact")

    def _run_clique(self, alg):
        if not self.current_graph: return
        # Assumes CliqueThread is defined elsewhere
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
