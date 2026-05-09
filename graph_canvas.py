from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter, QFont


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
