
def dbscan(metrics, eps, minimum_points_per_cluster):
    clusters = []
    not_visited = [name for name in metrics]
    noise = []
    while not_visited:
        not_visited_point = not_visited[0]
        not_visited = not_visited[1:]
        neighbours = _determine_neighbours(metrics, not_visited_point, eps)
        if len(neighbours) < minimum_points_per_cluster:
            noise += [not_visited_point]
        else:
            clusters += [Cluster()]
            clusters[-1].add(not_visited_point)
            while neighbours:
                neighbour = neighbours[0]
                neighbours = neighbours[1:]
                if neighbour in not_visited:
                    not_visited.remove(neighbour)
                    neighbours_2nd = _determine_neighbours(metrics,
                                                           neighbour,
                                                           eps)
                    if len(neighbours_2nd) >= minimum_points_per_cluster:
                        neighbours = _join_sets(neighbours,
                                                neighbours_2nd)
                if not _point_is_in_any_cluster(neighbour, clusters):
                    clusters[-1].add(neighbour)

    print("/////////////////////////////\ndbscan result:\n")
    print("clusters are:")
    for cluster in clusters:
        print("cluster contains: {}".format(cluster.members))
    print("considered noise:")
    print(noise)
    print("\n")


class Cluster:
    def __init__(self):
        self.members = []

    def add(self, point):
        self.members += [point]


def _determine_neighbours(metrics, center, distance):
    return [neighbour for neighbour in metrics[center]
            if metrics[center][neighbour] < distance] + [center]


def _point_is_in_any_cluster(point, clusters):
    for cluster in clusters:
        if point in cluster.members:
            return True
    return False


def _join_sets(set1, set2):
    return set1.extend([p for p in set2 if p not in set1])
