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

    def display_graph(self, G, pos, dom_nodes=None, clique_nodes=None, node_labels=None, bridges=None,
                      kmedoids_clusters=None):
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

        # Prepare k-medoids color mapping
        node_cluster_map = {}
        cluster_colors = {}
        if kmedoids_clusters:
            items = list(kmedoids_clusters.items())
            for i, (medoid, members) in enumerate(items):
                # Distribute hues evenly across the color wheel
                hue = int((i / len(items)) * 359)
                cluster_colors[i] = QColor.fromHsv(hue, 220, 255)
                for member in members:
                    node_cluster_map[member] = (i, member == medoid)

        default_edge_pen = QPen(QColor(80, 80, 80, 100), 1)
        clique_edge_pen = QPen(QColor("#00FF00"), 2)
        bridge_edge_pen = QPen(QColor("#FF4444"), 2)

        normal_brush = QBrush(QColor("#00f2ff"))
        dom_brush = QBrush(QColor("#FF8C00"))
        clique_brush = QBrush(QColor("#00FF00"))
        both_brush = QBrush(QColor("#FFFFFF"))

        default_node_pen = QPen(Qt.GlobalColor.black, 1)
        medoid_node_pen = QPen(Qt.GlobalColor.white, 3)

        # Draw Edges
        for u, v in G.edges():
            if u in pos and v in pos:
                pen = default_edge_pen
                z_val = 0

                if frozenset([u, v]) in bridge_set:
                    pen = bridge_edge_pen
                    z_val = 2
                elif u in clique_set and v in clique_set:
                    pen = clique_edge_pen
                    z_val = 1
                # this would also color the edges, in the current version we will keep just nodes labelled
                # elif kmedoids_clusters and u in node_cluster_map and v in node_cluster_map:
                #     # Color the edge if it strictly belongs to a single cluster
                #     if node_cluster_map[u][0] == node_cluster_map[v][0]:
                #         cluster_color = cluster_colors[node_cluster_map[u][0]]
                #         pen = QPen(QColor(cluster_color.red(), cluster_color.green(), cluster_color.blue(), 160), 1.5)
                #         z_val = 0

                line = self.scene.addLine(pos[u][0], pos[u][1], pos[v][0], pos[v][1], pen)
                line.setZValue(z_val)

        # Draw Nodes
        for node in G.nodes():
            if node in pos:
                x, y = pos[node]
                is_dom = node in dom_set
                is_clique = node in clique_set
                is_medoid = kmedoids_clusters and node in node_cluster_map and node_cluster_map[node][1]

                current_node_pen = medoid_node_pen if is_medoid else default_node_pen

                # Overrides for brushing priority (Clique/Dom vs KMedoids color base)
                if is_dom and is_clique:
                    brush = both_brush
                    radius = 8
                elif is_clique:
                    brush = clique_brush
                    radius = 8
                elif is_dom:
                    brush = dom_brush
                    radius = 7
                elif kmedoids_clusters and node in node_cluster_map:
                    brush = QBrush(cluster_colors[node_cluster_map[node][0]])
                    radius = 9 if is_medoid else 5
                else:
                    brush = normal_brush
                    radius = 5

                diameter = radius * 2
                ellipse = self.scene.addEllipse(x - radius, y - radius, diameter, diameter, current_node_pen, brush)
                ellipse.setZValue(3 if (is_dom or is_clique or is_medoid) else 2)

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
