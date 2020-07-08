from Sql import SqlManager
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot


def get_information(column):
    """this function get information of column and wealth from database and return up_count and low_count and labels"""
    sql_manager = SqlManager("information.sqlite")

    data = sql_manager.crs. \
        execute('select {}, count({}) from information  GROUP by {} ORDER BY {}'
                .format(column, column, column, column)).fetchall()
    print(data)
    return data

def make_plot(data, address):
    """
        input : a column name
        outputs:
                    save plots depend on low and high income
                    and number of them
        Description:
                    x axi is for all distinct values of column
                    and y axi is number of value who have low or high income
        """
    labels = []
    counts = []
    for item in data:
        labels.append(item[0])
        counts.append(item[1])
    print(labels)
    print(counts)
    x = np.arange(len(labels)) * 100  # the label locations
    width = 30  # the width of the bars
    try:
        fig, ax = plt.subplots(figsize=(200, 100))
        plt.xticks(rotation=50, fontsize=60)
        # plt.xticks(rotation=50)
        plt.yticks(fontsize=60)
        # plt.yticks()
        rects1 = ax.bar(x - width / 2, counts, width, label='Low Income')
        ax.set_ylabel('Counts')
        # ax.set_title('Scores by income and {}'.format(column))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        # plt.show()
        plt.savefig(address)
        plt.close()
        fig.tight_layout()
        print(address, "  finished")
    except Exception as e:
        print("EXCEPT", e)


def run_plots():
    names =['status_type', 'num_comments', 'num_reactions',
            'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
            'num_sads', 'num_angrys']

    # names = ['num_comments']
    for col in names:
        data = get_information(col)
        make_plot(data, 'outs\\histogram_plots\\' + col)

if __name__ == '__main__':
    run_plots()
