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

from region_tree import region_tree_method

def test_region_tree_method():
    pts = [Point(1, 9), Point(7, 13), Point(3, 3), Point(1.5, 3), Point(5, 7),
           Point(9, 8), Point(6, 9), Point(7, 5), Point(7, 12), Point(4, 11), Point(1, 5)]
    x_range, y_range = [2.2, 7.7], [6.6, 11.11]

    pre = (sorted(pts), sorted(sorted(pts), key=lambda u: u.y))
    projections = [
        [Point(1, 5), Point(1, 9)],
        [Point(1.5, 3)],
        [Point(3, 3)],
        [Point(4, 11)],
        [Point(5, 7)],
        [Point(6, 9)],
        [Point(7, 5), Point(7, 12), Point(7, 13)],
        [Point(9, 8)]
    ]

    tree = BinTree(Node([[1, 8], [Point(1.5, 3),
                                  Point(3, 3),
                                  Point(1, 5),
                                  Point(7, 5),
                                  Point(5, 7),
                                  Point(9, 8),
                                  Point(1, 9),
                                  Point(6, 9),
                                  Point(4, 11),
                                  Point(7, 12),
                                  Point(7, 13)]]))
    tree.root.left = Node([[1, 4], [Point(1.5, 3),
                                    Point(3, 3),
                                    Point(1, 5),
                                    Point(1, 9),
                                    Point(4, 11)]])
    tree.root.left.left = Node([[1, 2], [Point(1.5, 3), Point(1, 5), Point(1, 9)]])
    tree.root.left.right = Node([[2, 4], [Point(1.5, 3), Point(3, 3), Point(4, 11)]])
    tree.root.left.right.left = Node([[2, 3], [Point(1.5, 3), Point(3, 3)]])
    tree.root.left.right.right = Node([[3, 4], [Point(3, 3), Point(4, 11)]])

    tree.root.right = Node([[4, 8], [Point(7, 5),
                                     Point(5, 7),
                                     Point(9, 8),
                                     Point(6, 9),
                                     Point(4, 11),
                                     Point(7, 12),
                                     Point(7, 13)]])
    tree.root.right.left = Node([[4, 6], [Point(5, 7), Point(6, 9), Point(4, 11)]])
    tree.root.right.left.left = Node([[4, 5], [Point(5, 7), Point(4, 11)]])
    tree.root.right.left.right = Node([[5, 6], [Point(5, 7), Point(6, 9)]])
    tree.root.right.right = Node([[6, 8], [Point(7, 5),
                                           Point(9, 8),
                                           Point(6, 9),
                                           Point(7, 12),
                                           Point(7, 13)]])
    tree.root.right.right.left = Node([[6, 7], [Point(7, 5),
                                                Point(6, 9),
                                                Point(7, 12),
                                                Point(7, 13)]])
    tree.root.right.right.right = Node([[7, 8], [Point(7, 5),
                                                 Point(9, 8),
                                                 Point(7, 12),
                                                 Point(7, 13)]])

    ps = [tree.root.left.right.right, tree.root.right.left, tree.root.right.right.left]
    ss = [[Point(4, 11)], [Point(5, 7), Point(6, 9), Point(4, 11)], [Point(6, 9)]]

    ans = region_tree_method(pts, x_range, y_range)
    TestCase().assertEqual(pre, next(ans))
    TestCase().assertEqual(projections, next(ans))
    TestCase().assertEqual(tree, next(ans))
    TestCase().assertEqual([3, 7], next(ans))
    TestCase().assertEqual(ps, next(ans))
    TestCase().assertEqual(ss, next(ans))





# All assertion tests are passing
test_region_tree_method()

