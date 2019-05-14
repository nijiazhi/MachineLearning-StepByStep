import pandas as pd
import matplotlib.pyplot as plt



def plot_rmse(file_path):
    fig = plt.figure()
    df1 = pd.read_csv(file_path, index_col=0)
    plt.title(file_path)
    plt.plot(df1.index, df1.loc[:, '0'], '-')
    return


def main():
    rmse_result_path_1 = './rmse_result/full-batch-rmse-result.csv'
    rmse_result_path_2 = './rmse_result/stochastic-rmse-result.csv'
    rmse_result_path_3 = './rmse_result/mini-batch-stochastic-rmse-result.csv'

    plot_rmse(rmse_result_path_1)
    plot_rmse(rmse_result_path_2)
    plot_rmse(rmse_result_path_3)

    plt.show()
    return


if __name__ == '__main__':
    main()