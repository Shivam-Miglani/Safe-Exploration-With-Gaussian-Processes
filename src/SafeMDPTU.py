import numpy as np

from src.utilities import max_out_degree

class SafeMDPTU(object):
    """Base class for safe exploration in MDPs, but with transitional uncertainty added.

    This class only provides basic options to compute the safely reachable
    and returnable sets. The actual update of the safety feature must be done
    in a class that inherits from `SafeMDPTU`. See `src.grid_world` for an
    example.

    Parameters
    ----------
    graph: networkx.DiGraph
        The graph that models the MDP. Each edge has an attribute `safe` in its
        metadata, which determines the safety of the transition.
    gp: GPy.core.GPRegression
        A Gaussian process model that can be used to determine the safety of
        transitions. Exact structure depends heavily on the use-case.
    S_hat0: boolean array
        An array that has True on the ith position if the ith node in the graph
        is part of the safe set.
    h: float
        The safety threshold.
    L: float
        The lipschitz constant
    beta: float, optional
        The confidence interval used by the GP model.
    """
    def __init__(self, graph, gp, S_hat0, h, L, beta=2):
        super(SafeMDPTU, self).__init__()
        # Scalar for gp confidence intervals
        self.beta = beta

        # Threshold
        self.h = h

        # Lipschitz constant
        self.L = L

        # GP model
        self.gp = gp

        self.graph = graph
        self.graph_reverse = self.graph.reverse()

        num_nodes = self.graph.number_of_nodes()
        num_edges = max_out_degree(graph)
        safe_set_size = (num_nodes, num_edges + 1)

        self.reach = np.empty(safe_set_size, dtype=np.bool)

        self.G = np.empty(safe_set_size, dtype=np.bool)


        self.S_hat = S_hat0.copy()

        self.S_hat0 = self.S_hat.copy()

        self.initial_nodes = self.S_hat0[:, 0].nonzero()[0].tolist()
        print("intial nodes: ")
        print(self.initial_nodes)

    def compute_S_hat(self):
        """Compute the safely reachable set given the current safe_set."""
        self.reach[:] = False
        reachable_set(self.graph, self.initial_nodes, out=self.reach)
        # print("Reachable Set:")
        # print(self.reach)
        self.S_hat[:] = False
        returnable_set(self.graph, self.graph_reverse, self.initial_nodes, out=self.S_hat)
        # print("Returnable Set:")
        # print(self.S_hat)

        self.S_hat &= self.reach
        print("S_hat:")
        print(self.S_hat)

    def add_gp_observations(self, x_new, y_new):
        """Add observations to the gp mode."""
        # Update GP with observations
        self.gp.set_XY(np.vstack((self.gp.X,x_new)),np.vstack((self.gp.Y,y_new)))


def link_graph_and_safe_set(graph, safe_set):
    """Link the safe set to the graph model.

    Parameters
    ----------
    graph: nx.DiGraph()
    safe_set: np.array
        Safe set. For each node the edge (i, j) under action (a) is linked to
        safe_set[i, a]
    """
    for node, next_node in graph.edges_iter():
        edge = graph[node][next_node]
        edge['safe'] = safe_set[node:node + 1, edge['action']]


def reachable_set(graph, initial_nodes, out=None):
    """
    Compute the safe, reachable set of a graph

    Parameters
    ----------
    graph: nx.DiGraph
        Directed graph. Each edge must have associated action metadata,
        which specifies the action that this edge corresponds to, and a
        probability value which corresponds to the chance of that edge
        being followed given that action.
        Each edge has an attribute ['safe'], which is a boolean that
        indicates safety
    initial_nodes: list
        List of the initial, safe nodes that are used as a starting point to
        compute the reachable set.
    out: np.array
        The array to write the results to. Is assumed to be False everywhere
        except at the initial nodes

    Returns
    -------
    reachable_set: np.array
        Boolean array that indicates whether a node belongs to the reachable
        set.
    """

    if not initial_nodes:
        raise AttributeError('Set of initial nodes needs to be non-empty.')

    if out is None:
        visited = np.zeros((graph.number_of_nodes(),
                            max_out_degree(graph) + 1),
                           dtype=np.bool)
    else:
        visited = out

    # All nodes in the initial set are visited
    visited[initial_nodes, 0] = True

    stack = list(initial_nodes)

    # TODO: rather than checking if things are safe, specify a safe subgraph?
    while stack:
        node = stack.pop(0)
        scary_actions = list()
        # examine all edges from node, see which actions are unsafe
        for _, next_node, data in graph.edges_iter(node, data=True):
            action = data['action']
            probability = data['probability']
            safe = data['safe']
            if not safe and probability:
                scary_actions.append(action)
        for _, next_node, data in graph.edges_iter(node, data=True):
            action = data['action']
            safe = data['safe']
            if not visited[node, action] and safe and action not in scary_actions:
                visited[node, action] = True
                if not visited[next_node, 0]:
                    stack.append(next_node)
                    visited[next_node, 0] = True

    if out is None:
        return visited

def returnable_set(graph, reverse_graph, initial_nodes, out=None):
    """
    Compute the safe, returnable set of a graph

    Parameters
    ----------
    graph: nx.DiGraph
        Directed graph. Each edge must have associated action metadata,
        which specifies the action that this edge corresponds to.
        Each edge has an attribute ['safe'], which is a boolean that
        indicates safety
    reverse_graph: nx.DiGraph
        The reversed directed graph, `graph.reverse()`
    initial_nodes: list
        List of the initial, safe nodes that are used as a starting point to
        compute the returnable set.
    out: np.array
        The array to write the results to. Is assumed to be False everywhere
        except at the initial nodes

    Returns
    -------
    returnable_set: np.array
        Boolean array that indicates whether a node belongs to the returnable
        set.
    """

    if not initial_nodes:
        raise AttributeError('Set of initial nodes needs to be non-empty.')

    if out is None:
        visited = np.zeros((graph.number_of_nodes(),max_out_degree(graph) + 1), dtype=np.bool)
    else:
        visited = out

    # Get reachable set
    reachable = reachable_set(graph, initial_nodes)

    # All nodes in the initial set are visited
    visited[initial_nodes, 0] = True

    stack = list(initial_nodes)
    popped = list()

    while stack:
        # for each node in the stack of safe nodes
        node = stack.pop(0)
        # save node for dead-end check if not in the initial safe set
        if not node in initial_nodes:
            popped.append(node)
        # iterate over edges going into node
        for _, prev_node in reverse_graph.edges_iter(node):
            data = graph.get_edge_data(prev_node, node)
            semi_safe_action = True
            # see if that action can also lead to unsafe places
            for _, other_node in graph.edges_iter(prev_node):
                other_data = graph.get_edge_data(prev_node, other_node)
                if other_data['probability'] and \
                        data['action'] == other_data['action'] and \
                        (not other_data['safe'] or reachable[other_node, 0]):
                    safe_action = False
            # if not yet visited and cannot lead to unsafe places, add to returnable states
            if not visited[prev_node, data['action']] and semi_safe_action:
                visited[prev_node, data['action']] = True
                if not visited[prev_node, 0]:
                    stack.append(prev_node)
                    visited[prev_node, 0] = True

    while popped:
        # for each node to check for dead-endiness
        node = popped.pop(0)
        # see if it leads somewhere
        leadsSomewhere = False
        for action in range(1,5):
            if visited[node, action]:
                leadsSomewhere = True
        if not leadsSomewhere:
            # mark node as non-returnable
            visited[node, 0] = False
            # check all nodes leading here
            for _, prev_node in reverse_graph.edges_iter(node):
                data = graph.get_edge_data(prev_node, node)
                # if that action leading here was considered safe
                if data['probability'] and visited[prev_node, data['action']]:
                    # consider it unsafe now
                    visited[prev_node, data['action']] = False
                    # and check if this previous node might be a dead end, if not defined as safe
                    if not prev_node in popped and not prev_node in initial_nodes:
                        popped.append(prev_node)

    if out is None:
        return visited
