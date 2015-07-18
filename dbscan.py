
def dbscan(metrics, eps, minimum_points_per_cluster):
    clusters = []
    visited = set()
    noise = set()
    for node_name in metrics:
        if node_name in visited:
            continue
        visited.add(node_name)
        neighbours = _determine_neighbours(metrics, node_name, eps)
        if len(neighbours) < minimum_points_per_cluster:
            noise.add(node_name)
        else:
            new_cluster = Cluster()
            new_cluster.add(node_name)
            while neighbours:
                neighbour = neighbours.pop()
                if neighbour not in visited:
                    visited.add(neighbour)
                    neighbours_2nd = _determine_neighbours(metrics,
                                                           neighbour,
                                                           eps)
                    if len(neighbours_2nd) >= minimum_points_per_cluster:
                        neighbours.union(neighbours_2nd)
                if not _point_is_in_any_cluster(neighbour, clusters):
                    new_cluster.add(neighbour)
            clusters += [new_cluster]

    print("/////////////////////////////\ndbscan result:\n")
    print("clusters are:")
    for cluster in clusters:
        print("cluster contains: {}".format(cluster.members))
    print("considered noise:")
    print(noise)
    print("\n")
    return clusters


class Cluster:
    def __init__(self):
        self.members = set()

    def add(self, point):
        self.members.add(point)

    def union(self, other_set):
        self.members = self.members.union(other_set)


def _determine_neighbours(metrics, center, distance):
    return set([neighbour for neighbour in metrics[center]
                if metrics[center][neighbour] < distance])


def _point_is_in_any_cluster(point, clusters):
    for cluster in clusters:
        if point in cluster.members:
            return True
    return False
