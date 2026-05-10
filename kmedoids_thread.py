import random
import networkx as nx
from PyQt6.QtCore import QThread, pyqtSignal


class KMedoidsThread(QThread):
    # Emits a dictionary mapping {medoid_node: [list_of_cluster_member_nodes]}
    finished_computing = pyqtSignal(dict)

    def __init__(self, graph, k):
        super().__init__()
        self.graph = graph
        self.k = k

    def run(self):
        nodes = list(self.graph.nodes())
        if not nodes:
            self.finished_computing.emit({})
            return

        if self.k >= len(nodes):
            # If k is >= nodes, every node is its own medoid
            self.finished_computing.emit({n: [n] for n in nodes})
            return

        # Precompute all shortest paths for accurate graph distance mapping.
        # Fallback to a large number if the graph is disconnected.
        try:
            dist_dict = dict(nx.all_pairs_shortest_path_length(self.graph))
        except Exception:
            dist_dict = {}

        def get_dist(u, v):
            try:
                return dist_dict[u][v]
            except KeyError:
                return 999999  # Disconnected components

        # Initialize random medoids
        medoids = random.sample(nodes, self.k)
        clusters = {}

        # Iteratively optimize medoids (Voronoi iteration)
        max_iterations = 10
        for _ in range(max_iterations):
            # Step A: Assign nodes to nearest medoid
            new_clusters = {m: [] for m in medoids}
            for n in nodes:
                closest = min(medoids, key=lambda m: get_dist(n, m))
                new_clusters[closest].append(n)

            # Step B: Update medoids
            new_medoids = []
            for m, members in new_clusters.items():
                if not members:
                    new_medoids.append(m)
                    continue

                # The new medoid is the node in the cluster that minimizes the sum of distances to all other nodes in the cluster
                best_m = min(members, key=lambda candidate: sum(get_dist(candidate, other) for other in members))
                new_medoids.append(best_m)

            # Check for convergence
            if set(new_medoids) == set(medoids):
                clusters = new_clusters
                break

            medoids = new_medoids
            clusters = new_clusters

        self.finished_computing.emit(clusters)
