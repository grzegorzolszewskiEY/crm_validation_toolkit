
import numpy as np
import pandas as pd 
from scipy.stats import uniform, binom, norm

class gAUC():

    def __init__(self) -> None:
        pass

    def r_i(self, row: int, cont_table: np.array):
        value = sum(cont_table[row][:])
        return value

    def r_i_2(self,row: int, cont_table: np.array):
        value = sum(cont_table[row][:])**2
        return value

    def w_r(self,cont_table: np.array):
        summation_list = []

        for row in range(cont_table.shape[0]):
            summation_list.append(self.r_i_2(row, cont_table))
        
        value = self.F_ij(cont_table)**2 - sum(summation_list)
        return value
    
    def c_j(self,col: int, cont_table: np.array):
        value = sum(cont_table[:][col])
        return value

    def F_ij(self, cont_table: np.array):
        empty_list = []

        for i in range(cont_table.shape[0]):
            empty_list.append(self.c_j(col = i, cont_table = cont_table))

        return sum(empty_list)

    def A_ij(self, row: int, col: int, cont_table: np.array):
        lower_observations = self.F_ij(cont_table=cont_table[:row,:col])
        higher_observations = self.F_ij(cont_table=cont_table[row+1:,col+1:])
        value = lower_observations + higher_observations
        return value

    def D_ij(self, row: int, col: int, cont_table: np.array):
        first_component = self.F_ij(cont_table=cont_table[row+1:,:col])
        second_component = self.F_ij(cont_table=cont_table[:row,col+1:])
        value = first_component + second_component
        return value

    def P_component(self, cont_table: np.array):
        summation_list = []

        for row in range(cont_table.shape[0]):
            for col in range(cont_table.shape[1]):
                calc = cont_table[row][col]*self.A_ij(row, col, cont_table)
                summation_list.append(calc)

        return sum(summation_list)

    def Q_component(self, cont_table: np.array):
        summation_list = []

        for row in range(cont_table.shape[0]):
            for col in range(cont_table.shape[1]):
                calc = cont_table[row][col]*self.D_ij(row, col, cont_table)
                summation_list.append(calc)

        return sum(summation_list)

    def calc_gAUC(self, cont_table: np.array):
        tt = (self.P_component(cont_table) - self.Q_component(cont_table))/self.w_r(cont_table)
        gAUC = (tt+1)/2
        return gAUC

    def d_ij(self, row: int, col: int, cont_table: np.array):
        return self.A_ij(row, col, cont_table) - self.D_ij(row, col, cont_table)

    def standard_dev(self, cont_table: np.array):
        summation_list = []
        P = self.P_component(cont_table)
        Q = self.Q_component(cont_table)
        F = self.F_ij(cont_table)
        wr = self.w_r(cont_table)
        for row in range(cont_table.shape[0]):
            for col in range(cont_table.shape[1]):
                component = cont_table[row][col] * (wr*self.d_ij(row,col,cont_table) - (P-Q)*(F-self.r_i(row, cont_table))) ** 2
                summation_list.append(component)

        return sum(summation_list) ** (1/2) * 1/(wr**2)