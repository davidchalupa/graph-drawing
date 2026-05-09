import sys
import networkx as nx

from PyQt6.QtWidgets import (
    QApplication, QMainWindow,
    QFileDialog, QProgressDialog, QPushButton, QVBoxLayout, QWidget,
    QStackedWidget, QDialog, QFormLayout,
    QSpinBox, QDialogButtonBox, QMessageBox, QGridLayout, QDoubleSpinBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QAction, QImage, QPixmap

from graph_canvas import GraphCanvas
from graph_canvas_optimized import GraphCanvasOptimized

from dominating_set_thread import DominatingSetThread
from clique_thread import CliqueThread
from layout_thread import LayoutThread


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

        # Grouped Generators into a submenu
        generate_menu = file_menu.addMenu("Generate")

        gen_sf_action = QAction("Barabási-Albert (scale-free)...", self)
        gen_sf_action.triggered.connect(self.generate_scale_free)
        generate_menu.addAction(gen_sf_action)

        gen_hk_action = QAction("Holme-Kim (powerlaw cluster)...", self)
        gen_hk_action.triggered.connect(self.generate_powerlaw_cluster)
        generate_menu.addAction(gen_hk_action)

        gen_bollobas_action = QAction("BRTS model (scale-free with bridges)...", self)
        gen_bollobas_action.triggered.connect(self.generate_scale_free_bridges)
        generate_menu.addAction(gen_bollobas_action)

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

    def generate_powerlaw_cluster(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Generate Powerlaw Cluster Network")
        layout = QFormLayout(dialog)

        n_spin = QSpinBox()
        n_spin.setRange(1, 100000)
        n_spin.setValue(500)

        m_spin = QSpinBox()
        m_spin.setRange(1, 1000)
        m_spin.setValue(2)

        p_spin = QDoubleSpinBox()
        p_spin.setRange(0.0, 1.0)
        p_spin.setSingleStep(0.1)
        p_spin.setValue(0.5)

        layout.addRow("Number of vertices (n):", n_spin)
        layout.addRow("Random edges per vertex (m):", m_spin)
        layout.addRow("Triangle formation prob (p):", p_spin)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            n = n_spin.value()
            m = m_spin.value()
            p = p_spin.value()
            if m >= n:
                QMessageBox.warning(self, "Invalid Parameters", "m must be strictly less than n.")
                return
            self.current_graph = nx.powerlaw_cluster_graph(n, m, p)
            self.setup_new_graph()

    # --- NEW GENERATOR IMPLEMENTATION HERE ---
    def generate_scale_free_bridges(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Generate Scale-free Network (with bridges)")
        layout = QFormLayout(dialog)

        n_spin = QSpinBox()
        n_spin.setRange(1, 100000)
        n_spin.setValue(500)

        alpha_spin = QDoubleSpinBox()
        alpha_spin.setRange(0.0, 1.0)
        alpha_spin.setSingleStep(0.05)
        alpha_spin.setValue(0.41)

        beta_spin = QDoubleSpinBox()
        beta_spin.setRange(0.0, 1.0)
        beta_spin.setSingleStep(0.05)
        beta_spin.setValue(0.54)

        gamma_spin = QDoubleSpinBox()
        gamma_spin.setRange(0.0, 1.0)
        gamma_spin.setSingleStep(0.05)
        gamma_spin.setValue(0.05)

        layout.addRow("Number of vertices (n):", n_spin)
        layout.addRow("Prob. add node with out-edge (α):", alpha_spin)
        layout.addRow("Prob. add edge between existing (β):", beta_spin)
        layout.addRow("Prob. add node with in-edge (γ):", gamma_spin)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            n = n_spin.value()
            a = alpha_spin.value()
            b = beta_spin.value()
            c = gamma_spin.value()

            total = a + b + c
            if total <= 0:
                QMessageBox.warning(self, "Invalid Parameters", "Sum of α, β, and γ must be > 0.")
                return

            # Normalize to ensure sum is exactly 1
            a, b, c = a / total, b / total, c / total

            # nx.scale_free_graph creates a directed Bollobás multigraph
            H = nx.scale_free_graph(n, alpha=a, beta=b, gamma=c)

            # Cast to an undirected simple graph to blend with your app natively
            G = nx.Graph(H)

            # Remove any self-loops that formed
            G.remove_edges_from(list(nx.selfloop_edges(G)))

            self.current_graph = G
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
