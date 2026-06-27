import networkx as nx

import numpy as np
from scipy.spatial import Delaunay

from PyQt6.QtWidgets import (
    QApplication, QDialog, QFormLayout,
    QSpinBox, QDialogButtonBox, QMessageBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt

from dominating_set_thread import DominatingSetThread
from clique_thread import CliqueThread
from kmedoids_thread import KMedoidsThread


def generate_random_planar_graph(num_nodes):
    # random 2D points
    points = np.random.rand(num_nodes, 2)

    # compute Delaunay triangulation (always planar)
    tri = Delaunay(points)

    # create networkx graph from triangulation edges
    G = nx.Graph()
    for path in tri.simplices:
        G.add_edge(path[0], path[1])
        G.add_edge(path[1], path[2])
        G.add_edge(path[2], path[0])

    return G, points


class GraphService:
    def __init__(self, parent_window, graph_data):
        self.parent = parent_window
        self.graph_data = graph_data

    def generate_scale_free(self):
        # FIX: Pass self.parent to Dialog
        dialog = QDialog(self.parent)
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
                # FIX: Pass self.parent to MessageBox
                QMessageBox.warning(self.parent, "Invalid Parameters", "m must be strictly less than n.")
                return

            # FIX: Store new graph in graph_data, not self!
            self.graph_data.current_graph = nx.barabasi_albert_graph(n, m)
            self.graph_data.setup_new_graph()

    def generate_powerlaw_cluster(self):
        dialog = QDialog(self.parent)
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
                QMessageBox.warning(self.parent, "Invalid Parameters", "m must be strictly less than n.")
                return
            self.graph_data.current_graph = nx.powerlaw_cluster_graph(n, m, p)
            self.graph_data.setup_new_graph()

    def generate_scale_free_bridges(self):
        dialog = QDialog(self.parent)
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
                QMessageBox.warning(self.parent, "Invalid Parameters", "Sum of α, β, and γ must be > 0.")
                return

            a, b, c = a / total, b / total, c / total

            H = nx.scale_free_graph(n, alpha=a, beta=b, gamma=c)
            G = nx.Graph(H)
            G.remove_edges_from(list(nx.selfloop_edges(G)))

            self.graph_data.current_graph = G
            self.graph_data.setup_new_graph()

    def generate_planar(self):
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Generate Planar Graph (Delaunay / Voronoi)")
        layout = QFormLayout(dialog)

        n_spin = QSpinBox()
        n_spin.setRange(1, 100000)
        n_spin.setValue(500)

        layout.addRow("Number of vertices (n):", n_spin)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            n = n_spin.value()
            G, points = generate_random_planar_graph(n)

            self.graph_data.current_graph = G
            self.graph_data.setup_new_graph()

    def compute_clustering(self):
        # FIX: Check graph_data.current_graph
        if not self.graph_data.current_graph: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            raw_cc = nx.clustering(self.graph_data.current_graph)
            self.graph_data.clustering_coeffs = {n: round(v, 3) for n, v in raw_cc.items()}
            self.parent.btn_toggle_cc.setEnabled(True)
            self.parent.btn_toggle_cc.setChecked(True)
            self.graph_data.show_clustering = True
            if self.graph_data.current_pos:
                self.graph_data.on_layout_finished(self.graph_data.current_pos)
        finally:
            QApplication.restoreOverrideCursor()

    def compute_betweenness(self):
        if not self.graph_data.current_graph: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            raw_bc = nx.betweenness_centrality(self.graph_data.current_graph)
            self.graph_data.betweenness_cent = {n: round(v, 4) for n, v in raw_bc.items()}
            self.parent.btn_toggle_bc.setEnabled(True)
            self.parent.btn_toggle_bc.setChecked(True)
            self.graph_data.show_betweenness = True
            if self.graph_data.current_pos:
                self.graph_data.on_layout_finished(self.graph_data.current_pos)
        finally:
            QApplication.restoreOverrideCursor()

    def compute_bridges(self):
        if not self.graph_data.current_graph: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.graph_data.bridges = list(nx.bridges(self.graph_data.current_graph))
            self.parent.btn_toggle_br.setEnabled(True)
            self.parent.btn_toggle_br.setChecked(True)
            self.graph_data.show_bridges = True
            if self.graph_data.current_pos:
                self.graph_data.on_layout_finished(self.graph_data.current_pos)
        finally:
            QApplication.restoreOverrideCursor()

    def run_dominating_set(self):
        if not self.graph_data.current_graph: return
        self.ds_thread = DominatingSetThread(self.graph_data.current_graph)
        self.ds_thread.finished_computing.connect(self.graph_data.on_ds_finished)
        self.ds_thread.start()

    def run_clique_greedy(self):
        self._run_clique("greedy")

    def run_clique_exact(self):
        self._run_clique("exact")

    def _run_clique(self, alg):
        if not self.graph_data.current_graph: return
        self.cl_thread = CliqueThread(self.graph_data.current_graph, alg)
        self.cl_thread.finished_computing.connect(self.graph_data.on_clique_finished)
        self.cl_thread.start()

    def run_kmedoids_dialog(self):
        if not self.graph_data.current_graph: return

        dialog = QDialog(self.parent)
        dialog.setWindowTitle("k-Medoids Clustering")
        layout = QFormLayout(dialog)

        k_spin = QSpinBox()
        max_k = max(2, self.graph_data.current_graph.number_of_nodes() - 1)
        k_spin.setRange(2, max_k)
        k_spin.setValue(3)

        layout.addRow("Number of clusters (k):", k_spin)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            k = k_spin.value()
            self.run_kmedoids(k)

    def run_kmedoids(self, k):
        if not self.graph_data.current_graph: return
        self.km_thread = KMedoidsThread(self.graph_data.current_graph, k)
        self.km_thread.finished_computing.connect(self.graph_data.on_kmedoids_finished)
        self.km_thread.start()
