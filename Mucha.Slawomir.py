#!/usr/bin/env python
# Author: Slawomir Mucha
# Date: 2019-01-14
import argparse
import copy
import os
import random
import time
import numpy as np
from collections import namedtuple, defaultdict
from itertools import tee
from os import listdir
from os.path import join, isfile


Solution = namedtuple('Solution', ['path', 'path_cost'])


class NearestNeighbourHeuristic:
    """
    Easiest heuristic that quickly yields feasible solution
    """
    def find_hamiltonian_cycle(self, matrix, precedences):
        cities = list(range(len(matrix)))
        ready_cities = list(set(cities) - set(precedences.keys()))
        start = random.choice(ready_cities)
        return self._perform_greedy_search(start, matrix, precedences)

    def _perform_greedy_search(self, start, matrix, precedences):
        n = len(matrix)
        path = [start]
        stack = [start]
        cost = 0
        precedences = copy.deepcopy(precedences)
        ready = set(range(n)) - set(precedences.keys())
        visited = set()

        while stack:
            current = stack.pop()
            visited.add(current)

            for city, city_precedences in precedences.items():
                if current not in city_precedences:
                    continue

                city_precedences.remove(current)
                if not city_precedences:
                    ready.add(city)

            neighbours = [(matrix[current][j], j) for j in range(n)
                          if j not in visited
                          and j in ready
                          and matrix[current][j] != -1]

            if not neighbours:
                break

            edge_cost, next_city = min(neighbours)

            cost += edge_cost
            stack.append(next_city)
            path.append(next_city)

        assert len(path) == n

        return Solution(path, cost)


class AntColonySystemHeuristic:

    def __init__(self,
                 iterations_number=8000,
                 evaporation_rate=0.1,
                 alpha=1,
                 beta=2,
                 d_0=0.8):
        self._iterations_number = iterations_number
        self._evaporation_rate = evaporation_rate
        self._alpha = alpha
        self._beta = beta
        self._d_0 = d_0

    def find_hamiltonian_cycle(self, matrix, precedences):
        matrix_array = np.array(matrix)
        visibility = 1 / (matrix_array + 0.00000001)
        initial_pheromones = visibility.copy()
        pheromones = visibility.copy()
        solutions = []

        for _ in range(self._iterations_number):
            solution = self._perform_ant_traversal(0, matrix_array, precedences, pheromones,
                                                   initial_pheromones, visibility)

            pheromones = (1 - self._evaporation_rate) * pheromones + (self._evaporation_rate / solution.path_cost)
            solutions.append(solution)

        return min(solutions, key=lambda s: s.path_cost)

    def _perform_ant_traversal(self, node, matrix, precedences, pheromones, initial_pheromones, visibility):
        n = len(matrix)
        precedences = copy.deepcopy(precedences)
        ready = set(range(n)) - set(precedences.keys())
        path = [node]
        visited = set()
        cost = 0

        for _ in range(n-1):
            visited.add(node)

            for city, city_precedences in precedences.items():
                if node not in city_precedences:
                    continue

                city_precedences.remove(node)
                if not city_precedences:
                    ready.add(city)

            neighbours = [j for j in range(n) if j in ready and j not in visited and matrix[node][j] != -1]
            d = random.random()

            if d > self._d_0:
                probs = np.array([self._pheromone_power(node, j, pheromones, visibility) for j in neighbours])
                probs /= probs.sum()
                next_node = np.random.choice(neighbours, p=probs)
            else:
                _, next_node = max((self._pheromone_power(node, j, pheromones, visibility), j) for j in neighbours)

            pheromones[node][next_node] = ((1 - self._evaporation_rate) * pheromones[node][next_node] +
                                           self._evaporation_rate * initial_pheromones[node][next_node])
            cost += matrix[node][next_node]
            path.append(next_node)
            node = next_node

        return Solution(path, cost)

    def _pheromone_power(self, i, j, pheromones, visibility):
        return (pheromones[i][j] ** self._alpha) * (visibility[i][j] ** self._beta)


class LPP3OptHeuristic:
    """
    Lexicographic Path Preserving 3-opt

    According to:
    https://www.chalmers.se/en/departments/math/research/research-groups/optimization/OptimizationMasterTheses/MScThesis-RaadSalman-final.pdf

    this approach outperforms other local-search heuristics
    """
    def improve_hamiltonian_cycle(self, matrix, precedences, initial_solution):
        # Step 1
        improvement = True
        solution = copy.deepcopy(initial_solution)
        path = solution.path
        m = len(path)

        # Step 2
        while improvement:
            # Step 3
            improvement = False
            stack = list(range(m - 2, 0, -1))

            # Step 4
            h = stack.pop()

            # Step 5
            count_h = 1
            f_mark = [0 for _ in range(m)]
            b_mark = [0 for _ in range(m)]

            while True:
                # Step 6
                best_exchange = None
                cost = None  # infinite
                backward = False
                forward = False

                # Step 7
                # iterate forward
                i_f = h + 1
                j_f = i_f + 1

                # Step 8
                while not best_exchange and i_f < m-1 and i_f < j_f:
                    for city in path[i_f+1:]:
                        if path[i_f] in precedences[city]:
                            f_mark[city] = count_h

                    # Step 9
                    while j_f < m and f_mark[path[j_f]] != count_h:
                        cost_old = matrix[path[h]][path[h+1]] + matrix[path[i_f]][path[i_f+1]] + matrix[path[j_f]][path[j_f+1]]
                        cost_new = matrix[path[h]][path[i_f+1]] + matrix[path[j_f]][path[h+1]] + matrix[path[i_f]][path[j_f+1]]

                        # Step 10
                        if cost_new < cost_old and (cost is None or cost_new < cost):
                            best_exchange = (h, i_f, j_f)
                            forward = True
                            backward = False
                            cost = cost_new

                        # Step 11
                        j_f += 1
                    i_f += 1

                # Step 12
                # iterate backward
                i_b = h - 1
                j_b = i_b - 1

                # Step 13
                while not best_exchange and i_b > 0 and i_b > j_b:
                    for city in path[0:i_b+1]:
                        if city in precedences[path[i_b+1]]:
                            b_mark[city] = count_h

                    # Step 14
                    while j_b >= -1 and b_mark[path[j_b+1]] != count_h:
                        cost_old = matrix[path[j_b]][path[j_b + 1]] + matrix[path[i_b]][path[i_b + 1]] + matrix[path[h]][path[h+1]]
                        cost_new = matrix[path[j_b]][path[i_b + 1]] + matrix[path[h]][path[j_b+1]] + matrix[path[i_b]][path[h+1]]

                        # Step 15
                        if cost_new < cost_old and (cost is None or cost_new < cost):
                            best_exchange = (h, i_b, j_b)
                            forward = False
                            backward = True
                            cost = cost_new

                        # Step 16
                        j_b -= 1
                    i_b -= 1

                if best_exchange:
                    # Step 17
                    if forward:
                        h_, i_, j_ = best_exchange
                        path = path[0:h_+1] + path[i_+1:j_+1] + path[h_+1:i_+1] + path[j_+1:]
                        improvement = True
                        stack += [x for x in (h_, i_, j_, h_ + 1, i_ + 1, j_ + 1) if x not in stack and x < m - 1]

                    # Step 18
                    elif backward:
                        h_, i_, j_ = best_exchange
                        path = path[0:j_+1] + path[i_+1:h_+1] + path[j_+1:i_+1] + path[h_+1:]
                        improvement = True
                        stack += [x for x in (h_, i_, j_, h_ + 1, i_ + 1, j_ + 1) if x not in stack and x < m - 1]

                # Step 19
                if not stack:
                    break

                h = stack.pop()
                count_h += 1

        return Solution(path, self._compute_cost(matrix, path))

    def _compute_cost(self, matrix, path):
        return sum(matrix[i][j] for i, j in pairwise(path))


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def read_data(path):
    with open(path, 'r') as file:
        _ = [file.readline() for _ in range(7)]
        number = int(file.readline().strip())
        matrix = []

        for _ in range(number):
            line = file.readline().strip()
            row_numbers = [int(x.strip()) for x in line.split()]
            matrix.append(row_numbers)

        return matrix


def read_directory(path):
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        yield file, read_data(file)


def write_result(input_path, target_directory, graph_path, path_cost, execution_time):
    filename = os.path.basename(input_path)

    with open('{}.out'.format(join(target_directory, filename)), 'w') as f:
        f.write(' '.join(str(x) for x in graph_path))
        f.write('\n{} {:.3f}s'.format(path_cost, execution_time))


def get_precedences(matrix):
    precedences = defaultdict(set)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == -1:
                precedences[i].add(j)
    return precedences


def find_cheap_hamiltonian_cycle(matrix):
    start = time.time()
    initial_heuristics = [
        NearestNeighbourHeuristic(),
        AntColonySystemHeuristic(),
    ]
    improvement_heuristics = [
        LPP3OptHeuristic(),
    ]

    precedences = get_precedences(matrix)
    best_solution = Solution(None, None)

    for initial_heuristic in initial_heuristics:
        solution = initial_heuristic.fnd_hamiltonian_cycle(matrix, precedences)
        print('Initial solution cost: {}'.format(solution.path_cost))
        for improvement_heuristic in improvement_heuristics:
            solution = improvement_heuristic.improve_hamiltonian_cycle(matrix, precedences, solution)
            print('Improved solution cost: {}'.format(solution.path_cost))

        if best_solution.path_cost is None or solution.path_cost < best_solution.path_cost:
            best_solution = solution

    return best_solution.path, best_solution.path_cost, time.time() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        type=str,
                        help='sciezka do katalogu z danymi')
    args = parser.parse_args()

    current_path = os.path.dirname(os.path.abspath(__file__))

    for input_file_path, matrix in read_directory(args.path):
        result = find_cheap_hamiltonian_cycle(matrix)
        write_result(input_file_path, current_path, *result)
        print('{} finished'.format(input_file_path))


if __name__ == '__main__':
    main()
