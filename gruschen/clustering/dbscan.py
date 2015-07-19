
class Cluster:
    def __init__(self):
        self.members = set()

    def add(self, point):
        self.members.add(point)

    def union(self, other_set):
        self.members = self.members.union(other_set)


def cluster_data(metrics, eps, minimum_points_per_cluster):
    clusters = []
    visited = set()
    noise = set()
    for node_name in metrics:
        if node_name in visited:
            continue
        _handle_unvisited_node(node_name, visited, clusters, metrics, eps,
                               noise, minimum_points_per_cluster)
    _print_clustering_results(clusters, noise)
    return clusters


def _handle_unvisited_node(node_name, visited, clusters, metrics, eps, noise,
                           minimum_points_per_cluster):
    visited.add(node_name)
    neighbours = _determine_neighbours(metrics, node_name, eps)
    if len(neighbours) < minimum_points_per_cluster:
        noise.add(node_name)
    else:
        clusters += [_expand_to_new_cluster(node_name, neighbours,
                                            visited, clusters, metrics, eps,
                                            minimum_points_per_cluster)]


def _expand_to_new_cluster(node_name, neighbours, visited, clusters, metrics,
                           eps, minimum_points_per_cluster):
    new_cluster = Cluster()
    new_cluster.add(node_name)
    while neighbours:
        neighbours = _handle_new_cluster_neighbour(new_cluster, neighbours,
                                                   visited, clusters, metrics,
                                                   eps,
                                                   minimum_points_per_cluster)
    return new_cluster


def _handle_new_cluster_neighbour(new_cluster, neighbours, visited, clusters,
                                  metrics, eps, minimum_points_per_cluster):
    neighbour = neighbours.pop()
    if neighbour not in visited:
        visited.add(neighbour)
        neighbours.union(
            _handle_second_degree_neighbourhood(neighbour, metrics, eps,
                                                minimum_points_per_cluster)
            )
    if not _point_is_in_any_cluster(neighbour, clusters):
        new_cluster.add(neighbour)


def _handle_second_degree_neighbourhood(neighbour, metrics, eps,
                                        minimum_points_per_cluster):
    neighbours_2nd = _determine_neighbours(metrics,
                                           neighbour,
                                           eps)
    if len(neighbours_2nd) >= minimum_points_per_cluster:
        return neighbours_2nd
    else:
        return set()


def _determine_neighbours(metrics, center, distance):
    return set([neighbour for neighbour in metrics[center]
                if metrics[center][neighbour] < distance])


def _point_is_in_any_cluster(point, clusters):
    for cluster in clusters:
        if point in cluster.members:
            return True
    return False


def _print_clustering_results(clusters, noise):
    print("/////////////////////////////\ndbscan result:\n")
    print("clusters are:")
    for cluster in clusters:
        print("cluster contains: {}".format([r.name for r in cluster.members]))
    print("considered noise:")
    print([r.name for r in noise])
    print("\n")
