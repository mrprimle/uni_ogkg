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
    Polygon,
    Hull,
)

def test_divide_and_conquer_hull1():
    p1 = Point(2, 2)
    p2 = Point(2, -2)
    p3 = Point(-2, -2)
    p4 = Point(-2, 2)
    r1 = Polygon((p1, p2, p3, p4))
    p1 = Point(3, 0)
    p2 = Point(0, -3)
    p3 = Point(-3, 0)
    p4 = Point(0, 3)
    r2 = Polygon((p1, p2, p3, p4))

    h = divide_and_conquer_hull(Hull(r1), Hull(r2))
    TestCase().assertEqual(
        h,
        [
            Point(0, -3),
            Point(2, -2),
            Point(3, 0),
            Point(2, 2),
            Point(0, 3),
            Point(-2, 2),
            Point(-3, 0),
            Point(-2, -2)
        ]
    )


def test_divide_and_conquer_hull2():
    p1 = Point(2, 2)
    p2 = Point(2, 0)
    p3 = Point(0, 0)
    p4 = Point(0, 2)
    r1 = Polygon((p1, p2, p3, p4))
    p1 = Point(-2, -2)
    p2 = Point(-2, 0)
    p3 = Point(0, -1)
    p4 = Point(0, -2)
    r2 = Polygon((p1, p2, p3, p4))

    h = divide_and_conquer_hull(Hull(r1), Hull(r2))
    TestCase().assertEqual(h, [Point(0, -2), Point(2, 0), Point(2, 2), Point(0, 2), Point(-2, 0), Point(-2, -2)])
