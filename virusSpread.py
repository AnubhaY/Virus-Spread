from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
import pandas as pd
import itertools
import time
import matplotlib.pyplot as plt

class MatrixGenerator():
    """
    Create object for Graph/Matrix
    """
    @classmethod
    def generate(self, matrix_list: list, algo='matrix'):
        """
        Take input:
        - matrix_list:  2d list, such as [[2, 1, 0], [1, 0, 1], [1, 1, 1]]
        - algo: str. possible values: matrix, grid
        
        Output: object to solve matrix.
        """
        try:
            matrix = np.array(matrix_list, np.int32)
        except ValueError as e:
            print(f" [!] Matrix not correct. {e}")
            raise e
        if algo == 'matrix':
            product = InfectionSpread()
        elif algo == 'grid':
            product = InfectionGraph()
        else:
            raise NotImplementedError(f"Algo {algo} is not implemented")
            
        product.input_layout(matrix)
        
        return product

class InfectionMatrix(ABC):
    @abstractmethod
    def input_layout(self, initial_matrix) -> None:
        raise NotImplementedError
        
    @abstractmethod
    def timer_till_all_infected(self) -> int:
        raise NotImplementedError


class InfectionSpread(InfectionMatrix):
    """
    Matrix algo to determine the spread of infection
    """
    def input_layout(self, initial_matrix) -> None:
        """
        Input is input layout.
        - initial_matrix: np.ndarray. 2d array where 2=infected, 1=uninfected, 0=none.
        """
        x,y = initial_matrix.shape
        self._infected_matrix_padded = np.zeros((x+2)*(y+2))
        self._infected_matrix_padded.shape = x+2, y+2
        self._infected_matrix_padded[1:-1, 1:-1] = initial_matrix.copy()
        self._infected_matrix_padded[self._infected_matrix_padded == 1] = 0
        self._infected_matrix_padded = np.array(
                self._infected_matrix_padded,
                dtype=bool)                              # False = 0, 1. True = 2
        self.infected_matrix = self._infected_matrix_padded[1:-1, 1:-1]
        
        self.occupancy_matrix = np.array(initial_matrix.copy(), dtype=bool) # False = 0. True = 1,2
        self.occupied_count = self._count_non_empty(self.occupancy_matrix)

    @classmethod
    def _up(cls, a):
        return a[2:, 1:-1]

    @classmethod
    def _down(cls, a):
        return a[:-2, 1:-1]

    @classmethod
    def _left(cls, a):
        return a[1:-1, 2:]

    @classmethod
    def _right(cls, a):
        return a[1:-1, :-2]

    def spread(self):
        """
        Perform 1 round of spread.
        Values are updated in infected_mtrix
        """
        a1 = self._up(self._infected_matrix_padded)
        a2 = self._down(self._infected_matrix_padded)
        a3 = self._left(self._infected_matrix_padded)
        a4 = self._right(self._infected_matrix_padded)
        
        self._infected_matrix_padded[1:-1, 1:-1] = a1 + a2 + a3 + a4 + self.infected_matrix
        self._infected_matrix_padded[1:-1, 1:-1] = self.infected_matrix * self.occupancy_matrix # make unoccupied as uninfected

    @classmethod
    def _count_non_empty(self, matrix):
        return np.count_nonzero(matrix)

    def infected_count(self) -> int:
        """
        returns total infected in layout.
        """
        return self._count_non_empty(self.infected_matrix)

    def occupancy_count(self) -> int:
        """
        returns total occupied in layout. Count is done only once in input_layout() method.
        """
        return self.occupied_count

    def are_all_infected(self) -> bool:
        """
        Check if everyone in hospital are infected.
        """
        return self.infected_count() == self.occupancy_count()
        
    def timer_till_all_infected(self) -> int:
        """
        It plays the infection spread till all in hospital are infected.
        Returns integer for total iterations required.
        """
        counter = 0
        total_infected = self.infected_count()
        if total_infected == 0:
            return -1 # Since no one is infected, no virus
        total_infected_previous = 0
        total_occupied = self.occupancy_count()
        while True:
            
            if total_infected == total_occupied: # virus is spread to all
                return counter # return counter
                
            elif total_infected == total_infected_previous: # virus is no longer spreading
                return -1 # cannot be done
            
            total_infected_previous = total_infected # update previous counter
            counter += 1
            self.spread() 
            total_infected = self.infected_count()


class InfectionGraph(InfectionMatrix):
    """
    Algorithm allows using graphs for solving.
    """
    def input_layout(self, initial_matrix) -> None:
        """
        Input is input layout.
        - initial_matrix: np.ndarray. 2d array where 2=infected, 1=uninfected, 0=none.
        """
        self.virus = []
        self.labels = {}
        self.color_map = []
        x,y = initial_matrix.shape
        self.G = nx.grid_2d_graph(x,y) # Create grid graph

        # update the grid based on data
        for a,b in itertools.product(range(x), range(y)):
            data = initial_matrix[a,b]
            coordinates = (a,b)
            
            if data == 0: # Empty node
                self.G.remove_node(coordinates)
                
            elif data == 1: # Uninfected
                self.labels[coordinates] = f'{coordinates}'
                self.color_map.append('blue')
                
            else: # Virus node
                self.virus.append(coordinates)
                self.labels[coordinates] = f'{coordinates}'
                self.color_map.append('red')

    def draw_graph(self) -> None:
        """
        Plot graph using matplotlib. Only for initial graph.
        """
        nx.draw(self.G, labels=self.labels, node_color=self.color_map, pos=nx.spring_layout(self.G, iterations=20))
        plt.draw()
        plt.show()
        
    def timer_till_all_infected(self) -> int:
        """
        Calculate using digkstra algorithm for each node the distance from all virus nodes.
        - Output time till everyone is infected.
        """
        
        # create a 2d list of all shortest paths from each virus node using dijkstra
        data = []
        for item in self.virus:
            data.append(nx.shortest_path_length(self.G, item))
            
        df = pd.DataFrame(data) # convert to dataframe
        if df.empty:
            return -1
        if len(df.columns) < len(self.G.nodes()): # Some nodes not reachable
            return -1 
        return int(df.min().max()) # max of (min reachable of each node)

if __name__ == "__main__":
    algo = 'grid'
    data = [[2, 1, 0, 2, 1], [1, 1, 1, 1, 1], [1, 0, 0, 2, 1]]
    matrix = MatrixGenerator.generate(data, algo=algo)
    print(matrix.timer_till_all_infected())
    if algo == 'grid':
        matrix.draw_graph()
