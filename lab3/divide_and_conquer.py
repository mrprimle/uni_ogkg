from models import Point
from graham import graham


def divide_and_conquer_hull(hull1, hull2):
    p1, p2, p3 = hull1.points[:3]
    centroid = Point.centroid((p1, p2, p3))

    if hull2.contains_point(centroid):
        points = list(hull1) + list(hull2)
    else:
        points = list(hull1) + list(hull2.get_arc(centroid))
    
    return list(graham(points))[-1]
