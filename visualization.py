import random

import matplotlib.pyplot as plt
import numpy as np


class Edge:
    def __init__(self, length, node1, node2):
        self._target_length = length
        self._node1 = node1
        self._node2 = node2

    def length(self):
        return np.linalg.norm(self._node1.pos - self._node2.pos)

    def energy(self):
        return (self._target_length - self.length())**2


class Node:
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos
        self.edges = []

    def add_edge(self, edge):
        self.edges.append(edge)

    def energy(self):
        return sum([edge.energy() for edge in self.edges])


# randomly selects graph and then uses linear axis search to improve the
# solution
def build_graph(metrics):
    n = len(metrics)
    node_names = [_ for _ in metrics]
    nodes = [Node(node_names[i], np.array([0., 0.])) for i in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append(_connect_nodes(metrics, nodes[i], nodes[j]))

    start_energy = max([max([metrics[n1][n2] for n2 in metrics[n1]]) for n1
                        in metrics])
    energy = start_energy
    while energy > 1e-5:
        change_in_energy = 0
        for i in range(len(nodes)):
            for j in range(2):
                direction = 2*np.array([random.random(), random.random()])-1
                old_pos = nodes[i].pos
                old_energy = sum([node.energy() for node in nodes])
                nodes[i].pos += energy * direction
                if sum([node.energy() for node in nodes]) < old_energy:
                    change_in_energy += abs(old_energy - nodes[i].energy())
                else:
                    nodes[i].pos = old_pos
        if change_in_energy <= 1e-7:
            energy /= 1.5
            energy_span = energy - start_energy
            percent_done = 100 / (1e-5 - start_energy) * energy_span
            print("\rOptimizing graph: {}%".format(percent_done), end="")
    print("\r")
    return nodes, edges


def _connect_nodes(metrics, node1, node2):
    edge = Edge(metrics[node1.name][node2.name], node1, node2)
    node1.add_edge(edge)
    node2.add_edge(edge)
    return edge


def print_graph(nodes, edges):
    max_energy = max([edge.energy() for edge in edges])
    min_distance = min([edge.length() for edge in edges])
    print("max observed edge energy: {}".format(max_energy))
    print("min edge length: {}".format(min_distance))
    for node in nodes:
        print("{}: ({}, {})".format(node.name, node.pos[0], node.pos[1]))

    fig = plt.gcf()
    for node in nodes:
        _draw_circle(fig, node.pos, min_distance / 20)
        _draw_caption(fig, node.pos, node.name)
    for edge in edges:
        _draw_edge(edge._node1.pos,
                   edge._node2.pos,
                   edge.energy() / max_energy)
    min_x_coord = min([node.pos[0] for node in nodes])
    max_x_coord = max([node.pos[0] for node in nodes])
    min_y_coord = min([node.pos[1] for node in nodes])
    max_y_coord = max([node.pos[1] for node in nodes])
    plt.axis([min_x_coord, max_x_coord, min_y_coord, max_y_coord])
    plt.show()


def _draw_circle(fig, pos, radius):
    fig.gca().add_artist(plt.Circle((pos[0], pos[1]), radius))


def _draw_caption(fig, pos, s):
    fig.text(pos[0], pos[1], s)


def _draw_edge(start, end, color_intensity):
    pass
