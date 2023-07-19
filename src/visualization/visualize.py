import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


class Visualize:
    def __init__(self):
        pass

    @staticmethod
    def graph_settings(arg):
        # Customizable Set-ups
        if arg == 1:
            plt.figure(figsize=(13, 15))

        font = FontProperties()
        font.set_family('serif bold')
        font.set_style('oblique')
        font.set_weight('bold')
        ax = plt.axes()
        ax.set_facecolor("#e6eef1")

    def plot_line(self, df, label, title, x_label, y_label):
        self.graph_settings(0)

        plt.figure(figsize=(20, 10))
        plt.plot(df, label=label, color='indigo')
        plt.title(title, fontsize=15, fontweight='bold')
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_graph(self, actual, predicted, title, x_label, y_label):
        self.graph_settings(1)
        plt.plot(actual, color='red', label='Actual Google Stock Price')
        plt.plot(predicted, color="blue", label="Predicted Google Stock Price")
        plt.title(title, fontsize=15, fontweight='bold')
        plt.xlabel(x_label, fontsize=12, fontweight='bold')
        plt.ylabel(y_label, fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid()
        plt.show()


