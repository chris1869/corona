import pandas
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.mixture import GaussianMixture


import matplotlib.pyplot as plt

df = pandas.read_pickle("stats.xz")

def print_tree(estimator, feat_names):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    values = estimator.tree_.value
#    samples = estimator.tree_.sample
#    print(samples.shape)
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            #if len(values) > 1:
            #    print("%snode=%s leaf node value %.3f." % (node_depth[i] * "\t", i, values[i]))
            #else:
            print("%snode=%s leaf node" % (node_depth[i] * "\t", i))
            print(values[i])
        else:
            print("%snode=%s test node: go to node %s if %s <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                    i,
                 children_left[i],
                 feat_names[feature[i]],
                 threshold[i],
                 children_right[i],
                 ))
    print()


def add_cross(a, cols):
    base_cols = list(cols)
    """
    squares = a**2
    cols.extend(map(lambda x: x+"_SQR", base_cols))
    logs = np.log(a+1)
    cols.extend(map(lambda x: x+"_LOG", base_cols))
    sqrts = np.sqrt(a)
    cols.extend(map(lambda x: x+"_SQRT", base_cols))
    """

    b = [a] #, squares, logs, sqrts]
    for i in range(len(base_cols)):
        for k in range(i+1, len(base_cols)):
            b.append((a[:, i] * a[:, k])[:,None])
            b.append(((a[:, i]+1) / (a[:, k]+1))[:,None])
            b.append((a[:, i] - a[:, k])[:,None])
            b.append((a[:, i] + a[:, k])[:,None])
            cols.append(base_cols[i] + "_MULT_" + base_cols[k])
            cols.append(base_cols[i] + "_DIV_" + base_cols[k])
            cols.append(base_cols[i] + "_DIFF_" + base_cols[k])
            cols.append(base_cols[i] + "_ADD_" + base_cols[k])
    return np.hstack(b), cols

targets = ["WORKING", "INFECTED", "SICK_PEAK", "DEATH", "DURATION"]
drops = ["WIDTH", "HEIGHT", "NUM_AGENTS", "FPS", "INITIAL_SICK", "AGENT_RADIUS", "DESEASE_DURATION", "RUN", "START_SPEED"]

#df.drop(df.index[np.logical_or(df["NUM_AGENTS"]==200.0, df["DURATION"]<150)], inplace=True)
df.drop(df.index[df["NUM_AGENTS"]==200.0], inplace=True)
#df["WORKING"] = df["WORKING"]/df["DURATION"]
x_cols = list(df.columns)
for col in (targets + drops):
    del x_cols[x_cols.index(col)]

X = np.array(df[x_cols].values, dtype=np.float64)
print(df["SD_RECOVERED"].unique())
print(df["NUM_AGENTS"].unique())
rng = np.random.RandomState(10)

#for upper_S in np.arange(0.1, 0.5, 0.025):
#    for upper_W in np.arange(5.5, 7.5, 0.1)

y = df["DURATION"] > 150
#        print(upper_W, upper_S, np.unique(y, return_counts=True)[1])


df["SICK_PEAK"] = df["SICK_PEAK"] < 0.2
#df["DURATION"] = np.log(df["DURATION"])
#df["INFECTED"] = np.log(df["INFECTED"])
#X, x_cols = add_square(X, x_cols) 
X, x_cols = add_cross(X, x_cols) 
targets = ["WORKING"] #, "SICK_PEAK"]
print(X.shape)
for target in targets:
    #y = np.array(df[targets].values)
    #if target in ["WORKING", "DURATION", "INFECTED"]:
    #    y = np.log(y) < 6
    # Fit regression model
    imps = []
    regr_1 = DecisionTreeClassifier(max_depth=4).fit(X,y)
    print("Accuracy to Tree predict target: %s: %.3f" % (target, regr_1.score(X, y)))
    for i in np.argsort(regr_1.feature_importances_)[::-1]:
        imp = regr_1.feature_importances_[i]
        if imp > 0:
            imps.append(i)
            print("Coeff: %s : %.5f" % (x_cols[i], imp))
    #plt.figure()
    print_tree(regr_1, x_cols)
    #break
    #plt.show()

    imps = np.array(imps)
    reg = LogisticRegression().fit(X[:, imps], y)
    print("Accuracy to Linear predict same_set target: %s: %.3f" % (target, reg.score(X[:, imps], y)))
    for i in np.argsort(np.abs(reg.coef_[0,:]))[::-1]:
        print(i)
        print("Coeff: %s : %.5f" % (x_cols[imps[i]], reg.coef_[0, i]))

    """
    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4, min_samples_leaf=10000),
                               n_estimators=10, random_state=rng).fit(X, y)
    print("Accuracy to Boost Tree predict target: %s: %.3f" % (target, regr_2.score(X, y)))

    for i in np.argsort(regr_2.feature_importances_)[::-1]:
        imp = regr_2.feature_importances_[i]
        if imp > 0:
            print("Coeff: %s : %.5f" % (x_cols[i], imp))

    print_tree(regr_2, x_cols)
    """
    break
