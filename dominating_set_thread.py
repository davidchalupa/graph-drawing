from PyQt6.QtCore import QThread, pyqtSignal

from analyze_graph.dominating_set import minimum_dominating_set_ilp


class DominatingSetThread(QThread):
    finished_computing = pyqtSignal(list)

    def __init__(self, G):
        super().__init__()
        self.G = G

    def run(self):
        if not self.G or not self.G.nodes:
            self.finished_computing.emit([])
            return
        dom_set = self.compute_algorithm_ilp(self.G)
        self.finished_computing.emit(dom_set if dom_set is not None else [])

    def compute_algorithm_ilp(self, G):
        print("Computing minimum dominating set of the graph...")
        mds = minimum_dominating_set_ilp(G, timeLimit=120)
        if mds is not None:
            print(f"Found a dominating set of size {len(mds)}.")
            return mds
        else:
            print(f"No feasible solution found.")
            return None
