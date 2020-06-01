import numpy as np


def get2d_hist(df, bins=20, x="DEATH", y="WORKING", z="DURATION", reduce_op=np.mean):
    x_bins = bins
    y_bins = bins

    x_steps = np.linspace(df[x].values.min(), df[x].values.max(), x_bins)

    y_steps = np.linspace(df[y].values.min(), df[y].values.max(), y_bins)

    hist = np.ones((x_bins-1, y_bins-1))
    for i in range(x_bins-1):
        for k in range(y_bins-1):
            sub_inds = np.logical_and(np.logical_and(y_steps[k] <= df[y], df[y] < y_steps[k+1]), 
                                      np.logical_and(x_steps[i] <= df[x], df[x] < x_steps[i+1]))

            if np.any(sub_inds):
                hist[i, k] = reduce_op(df.iloc[df.index[sub_inds]][z].values)
            else:
                hist[i, k] = np.nan
    return hist, x_steps, y_steps
