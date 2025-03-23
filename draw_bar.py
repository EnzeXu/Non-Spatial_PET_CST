import numpy as np
import matplotlib.pyplot as plt

def plot_bars_with_bounds(lower_bounds, upper_bounds, data):
    # Take the logarithm of the data and bounds
    log_data = np.log10(data)
    log_lower_bounds = np.log10(lower_bounds)
    log_upper_bounds = np.log10(upper_bounds)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(32, 9))

    # Plot the input data as short horizontal dashes in red
    ax.plot(np.arange(len(data)), log_data, 'r_', markersize=20, markeredgewidth=2)

    # Add the lower and upper bounds as error bars
    ax.errorbar(np.arange(len(data)), log_data, yerr=[log_data-log_lower_bounds, log_upper_bounds-log_data],
                fmt='none', ecolor='black', capsize=3)

    # Add text labels for the values
    for i, (x, y, yerr_low, yerr_high) in enumerate(zip(np.arange(len(data)), log_data, log_data-log_lower_bounds, log_upper_bounds-log_data)):
        ax.text(x, y+0.1, f'{data[i]:.2g}', fontsize=8, ha='center', color='red')
        ax.text(x, y-0.3, f'{lower_bounds[i]:.2g}', fontsize=8, ha='center')
        ax.text(x, yerr_high+0.1, f'{upper_bounds[i]:.2g}', fontsize=8, ha='center')

    # Set the tick labels to the original values
    ax.set_xticks(np.arange(0, len(data), 5))
    ax.set_xticklabels(range(1, len(data)+1, 5))

    # Set the y-axis label
    ax.set_ylabel('Value (log10)')

    # Show the plot
    plt.savefig("test/parameters.png", dpi=300)
    # plt.show()
    plt.close()

if __name__ == "__main__":
    from const import PARAMS, STARTS_WEIGHTS
    lb = [item["lb"] for item in PARAMS] + [item["lb"] for item in STARTS_WEIGHTS]
    ub = [item["ub"] for item in PARAMS] + [item["ub"] for item in STARTS_WEIGHTS]
    x = np.load("saves/params_20230314_185400_648329.npy")
    plot_bars_with_bounds(lb, ub, x)
