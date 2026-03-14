import pulp
import networkx as nx


def get_graph_adjacency_list(graph):
    # Convert graph to adjacency list representation
    adjacency_list = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        adjacency_list[node] = neighbors

    return adjacency_list


def minimum_dominating_set_ilp(graph, timeLimit=None):
    adjacency_list = get_graph_adjacency_list(graph)

    # Define ILP problem
    prob = pulp.LpProblem("Minimum Dominating Set", pulp.LpMinimize)

    # Define variables
    nodes = list(adjacency_list.keys())
    x = pulp.LpVariable.dicts("x", nodes, cat=pulp.LpBinary)

    # Objective function: minimize the number of selected nodes
    prob += pulp.lpSum(x[node] for node in nodes)

    # Constraints
    for node in nodes:
        prob += pulp.lpSum(x[n] for n in adjacency_list[node]) + x[node] >= 1

    # Solve the problem with a time limit
    if timeLimit is not None:
        solver = pulp.PULP_CBC_CMD(timeLimit=timeLimit, msg=True, keepFiles=True)
    else:
        solver = pulp.PULP_CBC_CMD(msg=True, keepFiles=True)
    print("Calling the solver...")
    prob.solve(solver)

    # Check the solution status
    status = pulp.LpStatus[prob.status]
    print("Solver status:", status)

    if status != 'Optimal':
        print("Problem not solved to optimality within time limit.")

        # Access upper and lower bounds
        objective_value_found = pulp.value(prob.objective)
        print("Last objective value found:", objective_value_found)

        """
        # The solverModel attribute might now be accessible
        if hasattr(prob, 'solverModel'):
            lower_bound = prob.solverModel.bestBound
            print("Lower Bound (Best Bound):", lower_bound)
        else:
            print("Solver model details are not accessible.")
        """

        return None

    # Extract solution
    min_dominating_set = [node for node in nodes if pulp.value(x[node]) == 1]

    return min_dominating_set
