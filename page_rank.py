import os
import time
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from progress import Progress

WEB_DATA = os.path.join(os.path.dirname(__file__), 'school_web.txt')


def load_graph(fd):
    """Load graph from text file

    Parameters:
    fd -- a file like object that contains lines of URL pairs

    Returns:
    A representation of the graph.

    Called for example with

    >>> graph = load_graph(open("school_web.txt"))

    the function parses the input file and returns a graph representation.
    Each line in the file contains two white space seperated URLs and
    denotes a directed edge (link) from the first URL to the second.
    """
    # Initiate graph dictionary
    graph = {}
    relate = {}
    # Iterate through the file line by line
    for line in fd:
        # And split each line into two URLs
        node, target = line.split()
        # Put nodes into the 'from' list
        graph.setdefault('from', [])
        # Put targets into the 'to' list
        graph.setdefault('to', [])
        graph['from'].append(node)
        graph['to'].append(target)

    # Create directional graph
    data_frame = pd.DataFrame(graph)
    G = nx.from_pandas_edgelist(data_frame, 'from', 'to', create_using=nx.DiGraph())

    nx.draw(G, arrows=True)

    # Display directional graph
    plt.show()
    return graph


def print_stats(graph):
    node_amount = len(set(graph['from']).union(set(graph['to'])))
    edge_amount = len(graph['from'])

    print(f'There are {node_amount} nodes and {edge_amount} edges in this graph.')


def stochastic_page_rank(graph, n_iter=1000000, n_steps=100):

    hit_count = {}
    for node in graph['from']:
        hit_count.setdefault(node, 0)
    for node in graph['to']:
        hit_count.setdefault(node, 0)

    for i in range(n_iter):
        current_node = random.choice(graph['from'])
        current_node_index = graph['from'].index(current_node)
        for j in range(n_steps):
            current_node = random.choice([n for n in graph['from'] if n == graph['from'][current_node_index]])
        hit_count[current_node] += (1 / n_iter)

    return hit_count


def distribution_page_rank(graph, n_iter=100):

    hit_count = {}
    temp_hit_count = {}
    base_hit = 1 / len(set(graph['from']).union(set(graph['to'])))
    for node in graph['from']:
        hit_count.setdefault(node, base_hit)
        temp_hit_count.setdefault(node, 0)
    for node in graph['to']:
        hit_count.setdefault(node, base_hit)
        temp_hit_count.setdefault(node, 0)

    for i in range(n_iter):
        for node in graph['from']:
            out_degree = graph['from'].count(node)
            if out_degree == 0:
                pass
            else:
                p = 1 / out_degree
                target_indices = [graph['from'].index(n) for n in graph['from'] if n == node]
                for index in target_indices:
                    temp_hit_count[graph['to'][index]] += p

    return hit_count


def main():
    web = load_graph(open(WEB_DATA))

    print_stats(web)

    diameter = 3

    print("Estimate PageRank through random walks:")
    n_iter = len(web) ** 2
    n_steps = 2 * diameter
    start = time.time()
    ranking = stochastic_page_rank(web, n_iter, n_steps)
    stop = time.time()
    time_stochastic = stop - start

    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    print('\n'.join(f'{100 * v:.2f}\t{k}' for k, v in top[:20]))
    print(f'Calculation took {time_stochastic:.2f} seconds.\n')

    print("Estimate PageRank through probability distributions:")
    n_iter = 2 * diameter
    start = time.time()
    ranking = distribution_page_rank(web, n_iter)
    stop = time.time()
    time_probabilistic = stop - start

    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    print('\n'.join(f'{100 * v:.2f}\t{k}' for k, v in top[:20]))
    print(f'Calculation took {time_probabilistic:.2f} seconds.\n')

    speedup = time_stochastic / time_probabilistic
    print(f'The probabilitic method was {speedup:.0f} times faster.')


if __name__ == '__main__':
    main()