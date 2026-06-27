from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPixmap


class UIService:
    def __init__(self, parent_window, graph_data):
        self.parent = parent_window
        self.graph_data = graph_data

    def toggle_menu_visibility(self):
        is_visible = self.parent.button_panel.isVisible()
        self.parent.button_panel.setVisible(not is_visible)
        self.parent.btn_collapse.setText("◀ Show" if is_visible else "▶ Hide")
        self.parent.update_overlay_position()

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
        self.parent.canvas.scene.clear()
        item = self.parent.canvas.scene.addPixmap(pixmap)

        rect = item.boundingRect()
        self.parent.canvas.setSceneRect(rect.adjusted(-50, -50, 50, 50))
        self.parent.canvas.fitInView(self.parent.canvas.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def toggle_dominating_set(self):
        self.graph_data.show_dominating_set = self.parent.btn_toggle_ds.isChecked()
        if self.parent.current_pos:
            self.parent.on_layout_finished(self.parent.current_pos)

    def toggle_clique(self):
        self.graph_data.show_clique = self.parent.btn_toggle_cl.isChecked()
        if self.parent.current_pos:
            self.parent.on_layout_finished(self.parent.current_pos)

    def toggle_clustering(self):
        self.graph_data.show_clustering = self.parent.btn_toggle_cc.isChecked()
        if self.parent.current_pos:
            self.parent.on_layout_finished(self.parent.current_pos)

    def toggle_betweenness(self):
        self.graph_data.show_betweenness = self.parent.btn_toggle_bc.isChecked()
        if self.parent.current_pos:
            self.parent.on_layout_finished(self.parent.current_pos)

    def toggle_bridges(self):
        self.graph_data.show_bridges = self.parent.btn_toggle_br.isChecked()
        if self.parent.current_pos:
            self.parent.on_layout_finished(self.parent.current_pos)

    def toggle_kmedoids(self):
        self.graph_data.show_kmedoids = self.parent.btn_toggle_km.isChecked()
        if self.parent.current_pos:
            self.parent.on_layout_finished(self.parent.current_pos)
