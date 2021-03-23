import math

import numpy as np
import pandas as pd
from collections import Counter

og_data = pd.DataFrame


class Node:
    def __init__(self):
        self.attribute = ""
        self.children = {}
        self.label = None


def main():
    global og_data
    df = pd.read_csv("data")
    print(df)
    og_data = df

    tree = generate_tree(df, 'gini')  # put 'ig', 'gr' or 'gini'

    test = pd.DataFrame([['overcast', 'A', 'A', False]], columns=['outlook', 'temperature', 'humidity', 'wind'])
    print(predict(tree, test))


def predict(tree, example):
    node = tree
    while node.label is None:
        node = node.children[example.at[0, node.attribute]]
    return node.label


def gain_ratios(data):
    columns = list(data.columns)
    e = []

    y = data[columns[-1]].tolist()

    for i in columns[:-1]:
        e.append(split_info(data[i].tolist(), y))
    info = entropies(data)

    return np.divide(info, e, out=np.zeros_like(info), where=(e != 0.0 and e != -0.0))


def gini_feature(feature, y):
    counts = Counter(feature)
    class_counts = Counter(y)
    pair = [(feature[i], y[i]) for i in range(len(feature))]
    pair_counts = Counter(pair)
    i_res = 0
    total = len(y)
    for i in counts:
        p_v = counts[i] / total
        term = 0
        for j in class_counts:
            term = math.pow(pair_counts[(i, j)] / counts[i], 2)
        i_res += p_v * term

    return i_res


def gini(data):
    columns = list(data.columns)
    e = []
    y = data[columns[-1]].tolist()
    for i in columns[:-1]:
        e.append(gini_feature(data[i].tolist(), y))
    return e


def entropies(data):
    columns = list(data.columns)
    e = []

    y = data[columns[-1]].tolist()
    c_e = class_entropy(y)

    for i in columns[:-1]:
        e.append(entropy_feature(data[i].tolist(), y))
    e = c_e - np.array(e)
    return e


def class_entropy(column):
    counts = Counter(column)
    size = len(column)
    entropy = sum([-1*(counts[i]/size)*math.log2(counts[i]/size) for i in counts])
    return entropy


def entropy_feature(feature, y):
    counts = Counter(feature)
    class_counts = Counter(y)
    pair = [(feature[i], y[i]) for i in range(len(feature))]
    pair_counts = Counter(pair)
    i_res = 0
    total = len(y)
    for i in counts:
        p_v = counts[i]/total
        term = 0
        for j in class_counts:
            num = pair_counts[(i, j)] / counts[i]
            term += num * math.log2(num) if num != 0 else 0
        i_res += p_v * term

    i_res *= -1
    return i_res


def split_info(feature, y):
    counts = Counter(feature)
    i_res = 0
    total = len(y)
    for i in counts:
        p_v = counts[i]/total
        i_res += p_v * math.log2(p_v)
    i_res *= -1
    return i_res


def generate_tree(data, crit):
    N = Node()
    columns = list(data.columns)
    y = data[columns[-1]].tolist()
    y_count = Counter(y)
    if len(y_count) == 1:
        N.label = y[0]
        return N
    elif len(columns) == 1:
        N.label = y_count.most_common(1)[0][0]
        return N
    else:
        if crit == 'ig':
            e = np.array(entropies(data))
        elif crit == 'gr':
            e = np.array(gain_ratios(data))
        else:
            e = np.array(gini(data))
        name = columns[np.argmax(e)]

        N.attribute = name

        og_col = og_data[name].tolist()
        names = set(og_col)
        for i in names:
            branch = Node()
            new_data = data.copy()
            new_data = new_data.loc[data[name] == i]
            new_data = new_data.drop(N.attribute, axis=1)
            if new_data.empty:
                branch.label = y_count.most_common(1)[0][0]
            else:
                branch = generate_tree(new_data, crit)
            N.children[i] = branch
    return N


main()
