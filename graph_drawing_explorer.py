import sys
import networkx as nx

import numpy as np
from scipy.spatial import Delaunay

from PyQt6.QtWidgets import (
    QApplication, QMainWindow,
    QFileDialog, QProgressDialog, QPushButton, QVBoxLayout, QWidget,
    QStackedWidget, QDialog, QFormLayout,
    QSpinBox, QDialogButtonBox, QMessageBox, QGridLayout, QDoubleSpinBox, QLabel
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QAction, QImage, QPixmap

from graph_canvas import GraphCanvas
from graph_canvas_optimized import GraphCanvasOptimized

from dominating_set_thread import DominatingSetThread
from clique_thread import CliqueThread
from layout_thread import LayoutThread
from kmedoids_thread import KMedoidsThread

from graph_data import GraphData
from graph_service import GraphService
from ui_service import UIService


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Network Graph Visualizer")

        self.current_pos = None

        self.stacked_widget = QStackedWidget()
        self.canvas_standard = GraphCanvas()

        self.canvas_optimized = GraphCanvasOptimized()

        self.graph_data = GraphData(self)
        self.graph_service = GraphService(parent_window=self, graph_data=self.graph_data)

        self.stacked_widget.addWidget(self.canvas_standard)
        self.stacked_widget.addWidget(self.canvas_optimized)
        self.setCentralWidget(self.stacked_widget)
        self.canvas = self.canvas_standard

        self._setup_overlay_buttons()
        self._setup_stats_overlay()
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

    def _setup_stats_overlay(self):
        self.stats_container = QWidget(self)
        layout = QVBoxLayout(self.stats_container)
        layout.setContentsMargins(10, 10, 10, 10)

        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("""
            color: #00f2ff; 
            font-size: 16px; 
            font-weight: bold;
            background-color: rgba(13, 13, 13, 0.75);
            border-radius: 4px;
            padding: 5px;
        """)
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.stats_label)

        self.stats_container.show()
        self.stats_container.raise_()

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

        self.btn_toggle_km = QPushButton("⛭")
        self.btn_toggle_km.setToolTip("Show k-Medoids Clusters")
        self.btn_toggle_km.setCheckable(True)
        self.btn_toggle_km.setEnabled(False)
        self.btn_toggle_km.clicked.connect(self.toggle_kmedoids)

        # Layout Modes (Column 1)
        self.btn_radial = QPushButton("🌀")
        self.btn_radial.setToolTip("Radial layout")
        # FIX: Route to self.graph_data.run_layout
        self.btn_radial.clicked.connect(lambda: self.graph_data.run_layout("radial"))

        self.btn_pca = QPushButton("📉")
        self.btn_pca.setToolTip("HDE (PCA) layout")
        self.btn_pca.clicked.connect(lambda: self.graph_data.run_layout("pca"))

        self.btn_spring = QPushButton("🕸")
        self.btn_spring.setToolTip("Spring layout")
        self.btn_spring.clicked.connect(lambda: self.graph_data.run_layout("spring"))

        self.btn_lowcross = QPushButton("📐")
        self.btn_lowcross.setToolTip("Low-crossing layout")
        self.btn_lowcross.clicked.connect(lambda: self.graph_data.run_layout("lowcross"))

        self.btn_matrix = QPushButton("▦")
        self.btn_matrix.setToolTip("Adjacency Matrix")
        self.btn_matrix.clicked.connect(lambda: self.graph_data.run_layout("matrix"))

        # Add to Grid
        grid_layout.addWidget(self.btn_toggle_ds, 0, 0)
        grid_layout.addWidget(self.btn_toggle_cl, 1, 0)
        grid_layout.addWidget(self.btn_toggle_cc, 2, 0)
        grid_layout.addWidget(self.btn_toggle_bc, 3, 0)
        grid_layout.addWidget(self.btn_toggle_br, 4, 0)
        grid_layout.addWidget(self.btn_toggle_km, 5, 0)

        grid_layout.addWidget(self.btn_radial, 0, 1)
        grid_layout.addWidget(self.btn_pca, 1, 1)
        grid_layout.addWidget(self.btn_spring, 2, 1)
        grid_layout.addWidget(self.btn_lowcross, 3, 1)
        grid_layout.addWidget(self.btn_matrix, 4, 1)

        grid_layout.setRowStretch(6, 1)  # Push everything up
        main_layout.addWidget(self.button_panel)

        self.overlay_container.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a;
                color: #00f2ff;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 10px;
                font-weight: bold;
                font-size: 18px;
                min-width: 35px;
                min-height: 35px;
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
        # 1. Keep the control buttons on the right side
        if hasattr(self, 'overlay_container'):
            self.overlay_container.adjustSize()
            w = self.overlay_container.width()
            h = self.overlay_container.height()
            self.overlay_container.setGeometry(self.width() - w - 20, 20, w, h)

        # 2. Pin the stats metrics cleanly to the upper-left corner
        if hasattr(self, 'stats_container'):
            self.stats_container.adjustSize()
            sw = self.stats_container.width()
            sh = self.stats_container.height()
            self.stats_container.setGeometry(20, 50, sw, sh)

    def _setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open Graph...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.graph_data.open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        # Grouped Generators into a submenu
        generate_menu = file_menu.addMenu("Generate")

        gen_sf_action = QAction("Barabási-Albert (scale-free)...", self)
        gen_sf_action.triggered.connect(self.graph_service.generate_scale_free)
        generate_menu.addAction(gen_sf_action)

        gen_hk_action = QAction("Holme-Kim (powerlaw cluster)...", self)
        gen_hk_action.triggered.connect(self.graph_service.generate_powerlaw_cluster)
        generate_menu.addAction(gen_hk_action)

        gen_bollobas_action = QAction("BRTS model (scale-free with bridges)...", self)
        gen_bollobas_action.triggered.connect(self.graph_service.generate_scale_free_bridges)
        generate_menu.addAction(gen_bollobas_action)

        gen_planar_action = QAction("Planar graph (Delaunay / Voronoi)...", self)
        gen_planar_action.triggered.connect(self.graph_service.generate_planar)
        generate_menu.addAction(gen_planar_action)

        compute_menu = menu_bar.addMenu("Compute")

        dom_set_menu = compute_menu.addMenu("Dominating set")
        algo1_action = QAction("ILP solution", self)
        algo1_action.triggered.connect(self.graph_service.run_dominating_set)
        dom_set_menu.addAction(algo1_action)

        clique_menu = compute_menu.addMenu("Maximum clique")
        algo_cl1_action = QAction("Greedy heuristic", self)
        algo_cl1_action.triggered.connect(self.graph_service.run_clique_greedy)
        algo_cl3_action = QAction("Branch and bound", self)
        algo_cl3_action.triggered.connect(self.graph_service.run_clique_exact)
        clique_menu.addAction(algo_cl1_action)
        clique_menu.addAction(algo_cl3_action)

        compute_menu.addSeparator()

        cc_action = QAction("Clustering coefficients", self)
        cc_action.triggered.connect(self.graph_service.compute_clustering)
        compute_menu.addAction(cc_action)

        bc_action = QAction("Betweenness centrality", self)
        bc_action.triggered.connect(self.graph_service.compute_betweenness)
        compute_menu.addAction(bc_action)

        compute_menu.addSeparator()

        br_action = QAction("Bridges", self)
        br_action.triggered.connect(self.graph_service.compute_bridges)
        compute_menu.addAction(br_action)

        compute_menu.addSeparator()

        km_action = QAction("k-Medoids clustering...", self)
        km_action.triggered.connect(self.graph_service.run_kmedoids_dialog)
        compute_menu.addAction(km_action)

    def show_adjacency_matrix(self):
        G = self.graph_data.current_graph
        n = G.number_of_nodes()

        if n > 10000:
            QMessageBox.warning(self, "Too Large", "Graph is too large to render as a bitmap (>10,000 nodes).")
            return

        img = QImage(n, n, QImage.Format.Format_RGB32)
        img.fill(QColor("#0d0d0d"))

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

        if self.graph_data.show_clustering or self.graph_data.show_betweenness:
            for n in self.graph_data.current_graph.nodes():
                lbls = []
                if self.graph_data.show_clustering and n in self.graph_data.clustering_coeffs:
                    lbls.append(f"C: {self.graph_data.clustering_coeffs[n]}")
                if self.graph_data.show_betweenness and n in self.graph_data.betweenness_cent:
                    lbls.append(f"B: {self.graph_data.betweenness_cent[n]}")
                if lbls:
                    node_labels[n] = "\n".join(lbls)

        # We leave the canvas display call unmodified for k-Medoids for now,
        # waiting for your next step to update the canvas parameters.
        self.canvas.display_graph(
            self.graph_data.current_graph,
            self.current_pos,
            self.graph_data.dominating_set if self.graph_data.show_dominating_set else None,
            self.graph_data.clique if self.graph_data.show_clique else None,
            node_labels=node_labels,
            bridges=self.graph_data.bridges if self.graph_data.show_bridges else None,
            kmedoids_clusters=self.graph_data.kmedoids_clusters if self.graph_data.show_kmedoids else None,
        )

        # --- Update Main UI Overlay Text ---
        stats_lines = []
        if self.graph_data.show_dominating_set and self.graph_data.dominating_set:
            stats_lines.append(f"Dominating set size: {len(self.graph_data.dominating_set)}")
        if self.graph_data.show_clique and self.graph_data.clique:
            stats_lines.append(f"Largest clique size: {len(self.graph_data.clique)}")

        self.stats_label.setText("\n".join(stats_lines))
        self._update_overlay_position()

    def toggle_dominating_set(self):
        self.graph_data.show_dominating_set = self.btn_toggle_ds.isChecked()
        if self.current_pos:
            self.on_layout_finished(self.current_pos)

    def toggle_clique(self):
        self.graph_data.show_clique = self.btn_toggle_cl.isChecked()
        if self.current_pos:
            self.on_layout_finished(self.current_pos)

    def toggle_clustering(self):
        self.graph_data.show_clustering = self.btn_toggle_cc.isChecked()
        if self.current_pos:
            self.on_layout_finished(self.current_pos)

    def toggle_betweenness(self):
        self.graph_data.show_betweenness = self.btn_toggle_bc.isChecked()
        if self.current_pos:
            self.on_layout_finished(self.current_pos)

    def toggle_bridges(self):
        self.graph_data.show_bridges = self.btn_toggle_br.isChecked()
        if self.current_pos:
            self.on_layout_finished(self.current_pos)

    def toggle_kmedoids(self):
        self.graph_data.show_kmedoids = self.btn_toggle_km.isChecked()
        if self.current_pos:
            # Re-runs layout/rendering step; UI visualization logic can be built here next
            self.on_layout_finished(self.current_pos)

    def on_layout_finished(self, pos):
        self.current_pos = pos
        node_labels = {}

        # FIX: Point to the actual graph object
        graph = self.graph_data.current_graph

        # Guard clause just in case
        if not graph:
            return

        if self.graph_data.show_clustering or self.graph_data.show_betweenness:
            # FIX: Iterate over the real graph's nodes
            for n in graph.nodes():
                lbls = []
                if self.graph_data.show_clustering and n in self.graph_data.clustering_coeffs:
                    lbls.append(f"C: {self.graph_data.clustering_coeffs[n]}")
                if self.graph_data.show_betweenness and n in self.graph_data.betweenness_cent:
                    lbls.append(f"B: {self.graph_data.betweenness_cent[n]}")
                if lbls:
                    node_labels[n] = "\n".join(lbls)

        self.canvas.display_graph(
            graph,  # FIX: Pass the real graph here
            self.current_pos,
            self.graph_data.dominating_set if self.graph_data.show_dominating_set else None,
            self.graph_data.clique if self.graph_data.show_clique else None,
            node_labels=node_labels,
            bridges=self.graph_data.bridges if self.graph_data.show_bridges else None,
            kmedoids_clusters=self.graph_data.kmedoids_clusters if self.graph_data.show_kmedoids else None,
        )

        # --- Update Main UI Overlay Text ---
        stats_lines = []
        if self.graph_data.show_dominating_set and self.graph_data.dominating_set:
            stats_lines.append(f"Dominating set size: {len(self.graph_data.dominating_set)}")
        if self.graph_data.show_clique and self.graph_data.clique:
            stats_lines.append(f"Largest clique size: {len(self.graph_data.clique)}")

        self.stats_label.setText("\n".join(stats_lines))
        self._update_overlay_position()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
