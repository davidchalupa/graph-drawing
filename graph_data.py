import networkx as nx

from PyQt6.QtWidgets import  QFileDialog, QProgressDialog
from PyQt6.QtCore import Qt

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


class GraphData:
    def __init__(self, parent_window):
        self.parent = parent_window
        self.current_graph = None

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self.parent, "Open Graph", "", "Graph Files (*.col *.graphml *.gml)"
        )
        if path:
            self.current_graph = load_from_col_file(path)
            self.setup_new_graph()

    def setup_new_graph(self):
        self.dominating_set = []
        self.show_dominating_set = False

        self.parent.btn_toggle_ds.setEnabled(False)
        self.parent.btn_toggle_ds.setChecked(False)

        self.clique = []
        self.show_clique = False
        self.parent.btn_toggle_cl.setEnabled(False)
        self.parent.btn_toggle_cl.setChecked(False)

        self.clustering_coeffs = {}
        self.show_clustering = False
        self.parent.btn_toggle_cc.setEnabled(False)
        self.parent.btn_toggle_cc.setChecked(False)

        self.betweenness_cent = {}
        self.show_betweenness = False
        self.parent.btn_toggle_bc.setEnabled(False)
        self.parent.btn_toggle_bc.setChecked(False)

        self.bridges = []
        self.show_bridges = False
        self.parent.btn_toggle_br.setEnabled(False)
        self.parent.btn_toggle_br.setChecked(False)

        # Reset k-medoids state
        self.kmedoids_clusters = {}
        self.show_kmedoids = False
        self.parent.btn_toggle_km.setEnabled(False)
        self.parent.btn_toggle_km.setChecked(False)

        num_nodes = self.current_graph.number_of_nodes()
        num_edges = self.current_graph.number_of_edges()

        if num_nodes > 4000 or num_edges > 10000:
            self.parent.switch_canvas(optimized=True)
        else:
            self.parent.switch_canvas(optimized=False)

        if num_nodes > 2500 or num_edges > 10000:
            self.parent.btn_spring.setEnabled(False)
            self.parent.btn_spring.setToolTip("Spring layout disabled (Graph too large)")
        else:
            self.parent.btn_spring.setEnabled(True)
            self.parent.btn_spring.setToolTip("Spring layout")

        if num_nodes > 1250 or num_edges > 5000:
            self.parent.btn_lowcross.setEnabled(False)
            self.parent.btn_lowcross.setToolTip("Low-crossing layout disabled (Graph too large)")
        else:
            self.parent.btn_lowcross.setEnabled(True)
            self.parent.btn_lowcross.setToolTip("Low-crossing layout")

        if num_nodes > 10000:
            self.parent.btn_matrix.setEnabled(False)
            self.parent.btn_matrix.setToolTip("Adjacency Matrix disabled (Graph too large - > 10k nodes)")
        else:
            self.parent.btn_matrix.setEnabled(True)
            self.parent.btn_matrix.setToolTip("Adjacency Matrix")

        self.run_layout("radial")

    def run_layout(self, mode):
        if getattr(self, 'current_graph', None) is None: return

        buttons = {
            "pca": self.parent.btn_pca,
            "spring": self.parent.btn_spring,
            "lowcross": self.parent.btn_lowcross,
            "radial": self.parent.btn_radial,
            "matrix": self.parent.btn_matrix
        }
        for key, btn in buttons.items():
            btn.setProperty("active", key == mode)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        if mode == "matrix":
            self.parent.show_adjacency_matrix()
            return

        msg = "Calculating graph layout. This may take a moment..."

        self.progress = QProgressDialog(msg, None, 0, 0, self.parent)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setWindowTitle("Progress")
        self.progress.show()

        self.layout_thread = LayoutThread(self.current_graph, mode)
        self.layout_thread.layout_finished.connect(self.on_layout_finished)
        self.layout_thread.start()

    def on_ds_finished(self, ds):
        self.dominating_set = ds
        if ds:
            self.parent.btn_toggle_ds.setEnabled(True)
            self.parent.btn_toggle_ds.setChecked(True)
            self.show_dominating_set = True
            if hasattr(self, 'current_pos') and self.current_pos:
                self.on_layout_finished(self.current_pos)

    def on_clique_finished(self, clique):
        self.clique = clique
        if clique:
            self.parent.btn_toggle_cl.setEnabled(True)
            self.parent.btn_toggle_cl.setChecked(True)
            self.show_clique = True
            if hasattr(self, 'current_pos') and self.current_pos:
                self.on_layout_finished(self.current_pos)

    def on_kmedoids_finished(self, clusters):
        self.kmedoids_clusters = clusters
        if clusters:
            self.parent.btn_toggle_km.setEnabled(True)
            self.parent.btn_toggle_km.setChecked(True)
            self.show_kmedoids = True
            if hasattr(self, 'current_pos') and self.current_pos:
                self.on_layout_finished(self.current_pos)

    def on_layout_finished(self, pos):
        self.current_pos = pos

        # Make sure the progress dialog is actually closed!
        if hasattr(self, 'progress') and self.progress is not None:
            self.progress.close()

        if hasattr(self.parent, 'on_layout_finished'):
            self.parent.on_layout_finished(pos)

