from networkx.algorithms import approximation


def get_large_clique(G):
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
