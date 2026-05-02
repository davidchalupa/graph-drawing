from PyQt6.QtCore import QThread, pyqtSignal

from analyze_graph.max_clique import get_large_clique_greedy
from analyze_graph.max_clique import get_large_clique_bopp_hald
from analyze_graph.max_clique import get_max_clique_bnb


class CliqueThread(QThread):
    finished_computing = pyqtSignal(list)

    def __init__(self, G, alg):
        super().__init__()
        self.G = G
        self.alg = alg

    def run(self):
        if not self.G or not self.G.nodes:
            self.finished_computing.emit([])
            return

        if self.alg == "bopp_hald":
            clique = self.compute_algorithm_hald(self.G)
        elif self.alg == "greedy":
            clique = self.compute_algorithm_greedy(self.G)
        elif self.alg == "exact":
            clique = self.compute_algorithm_exact(self.G)
        else:
            raise ValueError("Invalid clique algorithm")
        self.finished_computing.emit(clique if clique is not None else [])

    def compute_algorithm_hald(self, G):
        print("Computing a large clique in the graph...")
        clique = get_large_clique_bopp_hald(G)
        if clique is not None:
            print(f"Found a clique of size {len(clique)}.")
            return clique
        return None

    def compute_algorithm_greedy(self, G):
        print("Computing a large clique in the graph...")
        clique = get_large_clique_greedy(G)
        if clique is not None:
            print(f"Found a clique of size {len(clique)}.")
            return clique
        return None

    def compute_algorithm_exact(self, G):
        print("Computing the maximum clique in the graph...")
        clique = get_max_clique_bnb(G)
        if clique is not None:
            print(f"Found a clique of size {len(clique)}.")
            return clique
        return None
