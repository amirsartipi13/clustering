from Sql import SqlManager
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, chain
import itertools

def get_information(column1, column2):
    """this function get information of column and wealth from database and return up_count and low_count and labels"""
    sql_manager = SqlManager("information.sqlite")

    data = sql_manager.crs. \
        execute('select {},{} from information'
                .format(column1, column2)).fetchall()
    col_1 = []
    col_2 = []
    for item in data:
        col_1.append(item[0])
        col_2.append(item[1])

    return col_1, col_2

def make_plot(column1, column2, address, col1_name, col2_name):
    colors = (0, 1)

    # Plot
    plt.scatter(column1, column2, alpha=0.5)
    plt.title('Scatter plot ofd density')
    plt.xlabel(str(col1_name))
    plt.ylabel(str(col2_name))
    plt.savefig(address)
    plt.close()

def run_plots():
    names = ['num_comments', 'num_reactions',
               'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
               'num_sads', 'num_angrys']
    # names = ['num_comments']
    result = make_sub_set(names, 2)
    for item in result:
        col_1, col_2 = get_information(item[0][0], item[0][1])
        make_plot(col_1, col_2, 'outs\\density_plots\\' + str(item[0][0])+' and '+str(item[0][1]), item[0][0], item[0][1])

def make_sub_set(l, number):
    l1 = list(map(set, itertools.combinations(l, number)))
    result = []
    for i in range(len(l1)):
        result.append((list(l1[i]), [x for x in l if x not in list(l1[i])]))
    return result

if __name__ == '__main__':
    run_plots()
