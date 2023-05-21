from collections import OrderedDict
import math
from operator import add, sub
from itertools import chain, cycle, dropwhile, groupby, takewhile


class Point:
    def __init__(self, *args):
        """Make tuple of args."""
        self.coords = tuple(map(float, args))

    def dominating(self, other):
        '''True if each self coordinate is bigger than other'''
        return reduce(
            lambda a, b: a and b[0] >= b[1], zip(self.coords, other.coords), True
        )

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @property
    def z(self):
        return self.coords[2]

    @property
    def dim(self):
        return len(self.coords)

    def __getitem__(self, key):
        """Wrap tuple-like behavior."""
        return self.coords[key]

    def __str__(self):
        """Coords string representation."""
        return "(%s, %s)" % (self.x, self.y)

    def __eq__(self, other):
        """Compares point coords."""
        return self.coords == other.coords

    def __lt__(self, other):
        """Compares point coords."""
        return self.coords < other.coords

    def __add__(self, other):
        """To delete."""
        return Point(*list(map(add, self.coords, other.coords)))

    def __sub__(self, other):
        """To delete."""
        return Point(*list(map(sub, self.coords, other.coords)))

    def dist_to_point(self, other):
        '''Euclidean distance to point'''
        s = sum([(a - b) ** 2 for a, b in zip(self.coords, other.coords)])
        return math.sqrt(s)

    def dist_to_line(self, line):
        '''Euclidean distance to the 2D line'''
        return (
            abs(line.A * self.x + line.B * self.y + line.C) /
            math.sqrt(line.A ** 2 + line.B ** 2)
        )

    def angle_with(self, point1, point2):
        '''Angle point1-self-point2 in [-pi, pi]'''
        v1 = Vector.from_two_points(self, point1)
        v2 = Vector.from_two_points(self, point2)
        v1.normalize()
        v2.normalize()

        return math.acos(v1 * v2 / (v1.euclidean_norm * v2.euclidean_norm))

    def polar_angle_with(self, other):
        '''Polar angle between self and other with other as origin'''
        return math.atan2(self.y - other.y, self.x - other.x)

    def ccw_polar_angle_with(self, other):
        '''Non-negative polar angle between self and other with other as origin'''
        angle = self.polar_angle_with(other)
        return angle if angle >= 0 else 2 * math.pi + angle


    def __hash__(self):
        '''Hash all the point representation'''
        return hash(self.coords)

    def __repr__(self):
        '''Representation for debugging.'''
        return str(self)

    @staticmethod
    def direction(point1, point2, point3):
        '''Numeric description of point positions.

        < 0 if point3 is at the left of vector point1->point2;
        > 0 if point3 is at the right of vector point1->point2;
        = 0 if point3 is at the vector point1->point2.
        '''
        v1 = Vector.from_two_points(point1, point3)
        v2 = Vector.from_two_points(point1, point2)
        return v1.cross_product_with(v2)

    @staticmethod
    def centroid(point_iter):
        '''Coordinate-wise mean of points iterable'''
        return Point(*(sum(coord) / len(coord) for coord in zip(*point_iter)))

class Vector:
    def __init__(self, coords):
        """Make vector from coords iterable."""
        self.coords = coords

    def __len__(self):
        """Dimension of vector instance."""
        return len(self.coords)

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @property
    def z(self):
        return self.coords[2]

    @property
    def euclidean_norm(self):
        return math.sqrt(sum((i ** 2 for i in self.coords)))

    def __mul__(self, other):
        """Scalar vector multiplication."""
        return sum((i * j for i, j in zip(self.coords, other.coords)))

    def __getitem__(self, key):
        return self.coords[key]

    def angle(self, other):
        if len(self) == len(other):
            return math.acos(
                (self * other) / (self.euclidean_norm * other.euclidean_norm)
            )

    def signed_angle(self, other):
        def abs_vect_mul_2d(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]

        return math.asin(
            abs_vect_mul_2d(self, other)
            / (self.euclidean_norm * other.euclidean_norm)
        )

    @staticmethod
    def from_two_points(p1, p2):
        return Vector((p2 - p1).coords)

    def normalize(self):
        self.coords = tuple(x / self.euclidean_norm for x in self.coords)

    def cross_product_with(self, other):
        return self.x * other.y - other.x * self.y


class Vertex:
    def __init__(self, point):
        self.point = point

    def __hash__(self):
        return hash(self.point)

    def __eq__(self, other):
        return self.point == other.point

    def __getitem__(self, x):
        return self.point.coords[x]

    def __str__(self):
        return str(self.point)

    def __repr__(self):
        return str(self)

class Edge:
    def __init__(self, v1, v2, weight=0):
        self.v1, self.v2, self.weight = v1, v2, weight

    def __hash__(self):
        return hash(self.v1) + hash(self.v2)

    def __eq__(self, other):
        return set((self.v1, self.v2)) == set((other.v1, other.v2))

    def __str__(self):
        return '({}, {})'.format(self.v1, self.v2) + f"weight = {repr(self.weight)}"


class OrientedEdge(Edge):
    def __hash__(self):
        return super().__hash__() + hash(self.weight)

    def __eq__(self, other: Edge):
        return (self.v1, self.v2) == (other.v1, other.v2)

    def __repr__(self):
        return f"{repr(self.v1)}->{repr(self.v2)}, weight = {repr(self.weight)}"

class Node:
    def __init__(self, data):
        """By default Node has no children."""
        self.data = data
        self.left = None
        self.right = None

    def __eq__(self, other):
        """Recursive equality."""
        return (
            self.data == other.data
            and self.left == other.left
            and self.right == other.right
        )


class Graph:
    edge_class = Edge

    def __init__(self):
        self.vertices, self.edges = set(), set()

    def __str__(self):
        """Return str for edges of graph."""
        return str(self.edges)

    def sorted_vertices(self, sort_key):
        return sorted(self.vertices, key=sort_key)

    def add_vertex(self, v: Vertex):
        self.vertices.add(v)

    def add_edge(self, v1: Vertex, v2: Vertex, weight=0):
        e1 = self.edge_class(v1, v2, weight)
        e2 = self.edge_class(v2, v1)
        if (v1 in self.vertices and v2 in self.vertices
            and e1 not in self.edges and e2 not in self.edges):
            self.edges.add(e1)



class OrientedGraph(Graph):
    edge_class = OrientedEdge

    def add_edge(self, v1: Vertex, v2: Vertex, weight=0):
        if (v1 in self.vertices and v2 in self.vertices):
            self.edges.add(self.edge_class(v1, v2, weight))

    def is_regular(self):
        '''
            Checks whether a graph is regular, i.e. each of its vertices
            has both incoming and outcoming edge(s),
            except for the starting (no incoming) and ending (no outcoming).
        '''
        sorted_vertices = self.sorted_vertices(sort_key=lambda v: v.point.y)[1:-1]
        regular_vertices = [e.v1 for e in self.edges] + [e.v2 for e in self.edges]

        return all(v in regular_vertices for v in sorted_vertices)

class OrientedEdge(Edge):
    def __hash__(self):
        return super().__hash__() + hash(self.weight)

    def __eq__(self, other: Edge):
        return (self.v1, self.v2) == (other.v1, other.v2)

    def __repr__(self):
        return f"{repr(self.v1)}->{repr(self.v2)}, weight = {repr(self.weight)}"


class NodeWithParent(Node):
    def __init__(self, data, parent = None):
        self.parent = parent
        super().__init__(data)


class BinTree:
    node_class = Node

    def __init__(self, root: Node):
        self.root = root

    def __eq__(self, other):
        return self.root == other.root
    
    @classmethod
    def from_iterable(cls, iterable):
        return cls(cls._from_iterable(iterable))
    
    @classmethod
    def _from_iterable(cls, iterable, left=0, right=None):
        if right is None:
            right = len(iterable)-1
        if left > right:
            return None
        
        mid = (left + right) // 2
        node = cls.node_class(iterable[mid])
        node.left = cls._from_iterable(iterable, left, mid-1)
        node.right = cls._from_iterable(iterable, mid+1, right)

        return node

    def traverse_inorder(self, node=None, nodes=None):
        if node is None:
            node = self.root
        if nodes is None:
            nodes = []
        
        if node.left:
            self.traverse_inorder(node.left, nodes)
        
        nodes.append(node)

        if node.right:
            self.traverse_inorder(node.right, nodes)
        
        return nodes

    @property
    def nodes(self):
        """
            Returns the tree represented as left-to-right list of tuples
            with nodes' data, and the data of their left and right children.
        """
        return self._nodes(self.root)
    
    def _nodes(self, node, result=None):
        if result is None:
            result = []
        
        left_data = node.left.data if node.left else None
        right_data = node.right.data if node.right else None

        if node:
            result.append((node.data, left_data, right_data))
        if node.left:
            self._nodes(node.left, result)
        if node.right:
            self._nodes(node.right, result)
        
        return result





class ChainsBinTree(BinTree):
    def make_tree(self, list, node):
        mid = len(list) // 2
        if mid == 0:
            return

        list_l = list[:mid]
        list_r = list[-mid:]
        left, right = list_l[mid // 2], list_r[mid // 2]

        node.left = NodeWithParent(left, node)
        if node.data != right:
            node.right = NodeWithParent(right, node)

        self.make_tree(list_l, node.left)
        self.make_tree(list_r, node.right)

    @staticmethod
    def _point_in_edge(edge, point):
        return edge.v1[1] <= point.y and edge.v2[1] >= point.y

    @staticmethod
    def _location_against_edge(point, edge):
        return Point.direction(edge.v1.point, edge.v2.point, point)

    def search_point(self, point):
        '''Returns a pair of chains the point is between'''
        current_node = self.root
        left_parent, right_parent = None, None

        while current_node:
            edge = list(
                filter(lambda e: ChainsBinTree._point_in_edge(e, point), current_node.data)
            )[0]
            location = ChainsBinTree._location_against_edge(point, edge)

            if location > 0:
                if current_node.right is not None:
                    current_node = current_node.right
                    left_parent = current_node.parent
                else:
                    return (current_node.data, right_parent.data)

            elif location < 0:
                if current_node.left is not None:
                    current_node = current_node.left
                    right_parent = current_node.parent
                else:
                    return (left_parent.data, current_node.data)
            else:
                return (current_node.data, None)

class RegionTree(BinTree):
    def __init__(self, points):
        self.__y_sort_flat = lambda l: sorted(chain.from_iterable(l), key=lambda u: u.y)
        self.x_ordered = sorted(points)
        self.y_ordered = sorted(self.x_ordered, key=lambda u: u.y)
        self.projections = [list(g) for _, g in groupby(self.x_ordered, key=lambda u: u.x)]
        interval = [1, len(self.projections)]
        super().__init__(Node([interval, self.__y_sort_flat(self.projections)]))
        self.__build(self.root)

    def __build(self, node: Node):
        start, end = node.data[0]
        mid = (end + start) // 2
        if (end - start) == 1:
            return
        l_int, r_int = [start, mid], [mid, end]
        l_list = self.__y_sort_flat(self.projections[start - 1:mid])
        r_list = self.__y_sort_flat(self.projections[mid - 1:end])
        if l_list:
            node.left = Node([l_int, l_list])
            self.__build(node.left)
        if r_list:
            node.right = Node([r_int, r_list])
            self.__build(node.right)

    def region_search(self, x_range, y_range):
        x_norm = self.region_normalization(self.projections, x_range)
        if not x_norm:
            return []
        nodes = RegionTree.primary_search(self.root, x_norm)
        res = set(chain.from_iterable(self.secondary_search(node.data[1], y_range) for node in nodes))
        return res

    @staticmethod
    def region_normalization(projections, x_range):
        enum = list(enumerate([ls[0] for ls in projections]))
        l_bound = next((x for x, val in enum if val.x >= x_range[0]), None)
        r_bound = next((x for x, val in reversed(enum) if val.x <= x_range[1]), None)
        return [l_bound + 1, r_bound + 1] if l_bound is not None and r_bound is not None else []

    @staticmethod
    def primary_search(node: Node, interval):
        begin, end = node.data[0]
        if begin == interval[0] and end == interval[1]:
            return [node]
        mid = (end + begin) // 2
        nodes = []
        if node.left and interval[0] < mid:
            new_interval = [interval[0], mid if interval[1] > mid else interval[1]]
            nodes.extend(RegionTree.primary_search(node.left, new_interval))
        if node.right and interval[1] > mid:
            new_interval = [mid if interval[0] < mid else interval[0], interval[1]]
            nodes.extend(RegionTree.primary_search(node.right, new_interval))
        return nodes

    @staticmethod
    def secondary_search(ls, interval):
        enum = list(enumerate(ls))
        l_bound = next((x for x, val in enum if val.y >= interval[0]), None)
        r_bound = next((x for x, val in reversed(enum) if val.y <= interval[1]), None)
        return ls[l_bound:r_bound + 1] if l_bound is not None and r_bound is not None else []



class Polygon:
    def __init__(self, points):
        """Make polygon by point list."""
        self.points = points

    def __getitem__(self, key):
        """Wraps tuple-like behavior."""
        return self.points[key]

    @property
    def point_pairs(self):
        def cyclic_offset(li, n):
            return li[-n:] + li[:-n]

        return zip(self.points, cyclic_offset(self.points, 1))

    def contains_point(self, point):
        pairs = self.point_pairs

        def angle(center, p1, p2):
            v1 = Vector.from_two_points(center, p1)
            v2 = Vector.from_two_points(center, p2)
            return v1.signed_angle(v2)

        total_angle = reduce(
            lambda accum, a: accum + angle(point, a[0], a[1]), pairs, 0
        )
        return total_angle > math.pi

    @property
    def area(self):
        a, b, c = self.points[:3]
        p = Point.centroid((a, b, c))
        pairs = self.point_pairs

        def accumulate_triangle_area(area_sum, pair):
            return area_sum + Triangle(p, pair[0], pair[1]).area

        return reduce(accumulate_triangle_area, pairs, 0)


class Hull(Polygon):
    def reference_points(self, point):
        v = Vector((0, 1))
        def key (end_point):
            return v.signed_angle(Vector.from_two_points(point, end_point))
        return (min(self, key=key), max(self, key=key))

    def get_arc(self, point):
        point_cycle = cycle(self)
        u, v = self.reference_points(point)

        def arc(start, end):
            """Return arc of point_cycle with start and end exclusively."""
            return list(
                chain(
                    takewhile(
                        lambda x: x != end, dropwhile(lambda x: x != start, point_cycle)
                    ),
                    (end,),
                )
            )

        arc1, arc2 = arc(u, v), arc(v, u)

        def key(arc):
            return Polygon(list(arc)+[point]).area

        return max((arc1, arc2), key=key)
