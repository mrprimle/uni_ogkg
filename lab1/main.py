import math
from copy import deepcopy
from unittest import TestCase
from collections import OrderedDict
from models import (
    Point,
    Vertex,
    Edge,
    OrientedEdge,
    Graph,
    OrientedGraph,
    Node,
    NodeWithParent,
    BinTree,
    ChainsBinTree,
)

from chain_method import chain_method



def test_chain_method():
    graph = OrientedGraph()
    point = Point(4, 5)
    v1 = Vertex(Point(4, 2))
    v2 = Vertex(Point(2, 4))
    v3 = Vertex(Point(6, 5))
    v4 = Vertex(Point(5, 7))

    e1 = OrientedEdge(v1, v2, 1)
    e2 = OrientedEdge(v1, v3, 1)
    e3 = OrientedEdge(v2, v3, 1)
    e4 = OrientedEdge(v2, v4, 1)
    e5 = OrientedEdge(v3, v4, 1)

    graph.add_vertex(v1)
    graph.add_vertex(v2)
    graph.add_vertex(v3)
    graph.add_vertex(v4)

    graph.add_edge(v1, v2, 1)
    graph.add_edge(v1, v3, 1)
    graph.add_edge(v2, v3, 1)
    graph.add_edge(v2, v4, 1)
    graph.add_edge(v3, v4, 1)

    ordered = [v1, v2, v3, v4]

    weight_table = OrderedDict(
        {
            v1: {"vin": [], "vout": [e1, e2], "win": 0, "wout": 2},
            v2: {"vin": [e1], "vout": [e4, e3], "win": 1, "wout": 2},
            v3: {"vin": [e3, e2], "vout": [e5], "win": 2, "wout": 1},
            v4: {"vin": [e4, e5], "vout": [], "win": 2, "wout": 0},
        }
    )

    e1_balanced = deepcopy(e1)
    e1_balanced.weight = 2
    e5_balanced = deepcopy(e5)
    e5_balanced.weight = 2
    weight_table_balanced = {
        v1: {"vin": [], "vout": [e1_balanced, e2], "win": 0, "wout": 3},
        v2: {"vin": [e1_balanced], "vout": [e4, e3], "win": 2, "wout": 2},
        v3: {"vin": [e3, e2], "vout": [e5_balanced], "win": 2, "wout": 2},
        v4: {"vin": [e4, e5_balanced], "vout": [], "win": 3, "wout": 0},
    }

    e1_new = deepcopy(e1)
    e1_new.weight = 0
    e2_new = deepcopy(e2)
    e2_new.weight = 0
    e3_new = deepcopy(e3)
    e3_new.weight = 0
    e4_new = deepcopy(e4)
    e4_new.weight = 0
    e5_new = deepcopy(e5)
    e5_new.weight = 0

    chains = [[e1_new, e4_new], [e1_new, e3_new, e5_new], [e2_new, e5_new]]

    root = NodeWithParent(data=chains[1])
    tree = ChainsBinTree(root)
    tree.root.left = NodeWithParent(data=chains[0], parent=root)
    tree.root.right = NodeWithParent(data=chains[2], parent=root)

    point_between = (chains[0], chains[1])

    ans = chain_method(graph, point)
    TestCase().assertEqual(ordered, next(ans))
    TestCase().assertEqual(weight_table, next(ans))
    TestCase().assertEqual(weight_table_balanced, next(ans))
    TestCase().assertEqual(chains, next(ans))
    TestCase().assertEqual(tree, next(ans))
    TestCase().assertEqual(point_between, next(ans))



# there is no assertionError, so the code is running correctly
test_chain_method()




