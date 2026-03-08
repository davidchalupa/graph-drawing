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
            # Scaling here is less important now because fitInView handles it
            for node in pos:
                pos[node][0] *= 100
                pos[node][1] *= 100
        else:
            pos = nx.spring_layout(self.G, scale=2000)

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
        layout.addWidget(self.btn_pca)
        layout.addWidget(self.btn_spring)
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
