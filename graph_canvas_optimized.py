import networkx as nx

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem
)
from PyQt6.QtCore import Qt, QLineF, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush, QPainter


from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import QPointF


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

        # ToDo: Intentionally skipping `node_labels` rendering in the optimized canvas for now
        # Rendering thousands of individual QGraphicsTextItems destroys OpenGL batching performance.

        self.setSceneRect(scene_rect)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)
