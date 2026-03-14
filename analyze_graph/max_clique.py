from networkx.algorithms import approximation


def get_large_clique_bopp_hald(G):
    """
    A heuristic for solving the maximum clique problem.
    Provides a guarantee of $O(|V| / (\log |V|)^2)$ approximation ratio.
    It works by recursively finding large independent sets in the complement graph to
    prune the search space.

    Boppana, R., & Halldórsson, M. M. (1992). Approximating maximum independent sets by excluding subgraphs.
    BIT Numerical Mathematics, 32(2), 180-196. https://doi.org/10.1007/BF01994876
    """

    clique_set = approximation.max_clique(G)

    return list(clique_set)

def get_large_clique_greedy(G, top_k_starts=10):
    """
    A fast greedy heuristic for finding a large clique in sparse graphs.
    Tests `top_k_starts` high-degree nodes as starting points.
    """
    if len(G) == 0:
        return []

    # Sort nodes by degree in descending order
    nodes_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
    best_clique = []

    # Try starting from a few of the highest degree nodes
    for start_node, _ in nodes_by_degree[:top_k_starts]:
        clique = [start_node]
        candidates = set(G.neighbors(start_node))

        while candidates:
            # Greedily pick the candidate with the highest degree
            next_node = max(candidates, key=lambda n: G.degree(n))
            clique.append(next_node)
            # Filter candidates to only those connected to the new node
            candidates.intersection_update(G.neighbors(next_node))

        if len(clique) > len(best_clique):
            best_clique = clique

    return best_clique


def get_max_clique_bnb(G):
    """
    An exact maximum clique algorithm using Branch and Bound.
    Highly memory efficient. Prunes branches that cannot possibly
    exceed the size of the largest clique found so far.
    """
    if not G or G.number_of_edges() == 0:
        return []

    best_clique = []
    max_size = 0

    # Sort nodes by degree (descending).
    # In scale-free networks, evaluating hubs first establishes a large
    # 'max_size' early, which allows us to prune the rest of the graph instantly.
    nodes = [n for n, d in sorted(G.degree(), key=lambda x: x[1], reverse=True)]

    def expand(current_clique, candidates):
        nonlocal best_clique, max_size

        # Update the best clique if we found a larger one
        if len(current_clique) > max_size:
            best_clique = current_clique[:]
            max_size = len(current_clique)

        # Bounding condition: If the current clique plus all remaining
        # candidates cannot beat the best clique, stop exploring this branch.
        if len(current_clique) + len(candidates) <= max_size:
            return

        for i, v in enumerate(candidates):
            remaining_candidates = candidates[i + 1:]

            # Secondary pruning: Check again before doing intersection math
            if len(current_clique) + 1 + len(remaining_candidates) <= max_size:
                break

            # To expand the clique, new candidates must be neighbors of the current node 'v'
            neighbors_v = set(G.neighbors(v))
            new_candidates = [n for n in remaining_candidates if n in neighbors_v]

            # Recurse depth-first
            current_clique.append(v)
            expand(current_clique, new_candidates)
            current_clique.pop()  # Backtrack

    expand([], nodes)
    return best_clique