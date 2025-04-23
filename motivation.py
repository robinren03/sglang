import numpy as np
import copy
from collections import Counter, defaultdict

def lower_bound(l, x):
    """
    在递增序列l中查找最小的索引i，使得l[i] >= x。
    如果不存在这样的i，则返回len(l)。
    """
    left, right = 0, len(l)
    while left < right:
        mid = (left + right) // 2
        if l[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left

metric = '80th Percentile'
def plot_lorenz_curve(ca):
    # 将输入的列表转换为numpy数组，并排序（降序）
    l = copy.deepcopy(ca)
    l_sorted = np.sort(l)  # 升序排序
    cumulative_counts = np.cumsum(l_sorted)
    total = cumulative_counts[-1]
    x_values = np.linspace(0, 1, len(l_sorted) + 1)
    y_values = np.insert(cumulative_counts / total, 0, 0)
    # gini_coefficient = 1 - np.trapz(y_values, x_values) * 2
    # return gini_coefficient
    fif_index = lower_bound(y_values, 0.2)
    return 1-x_values[fif_index]

def analyze_expert_distribution(l, num_threshold, freq_threshold, ana_threshold):
    """
    l: list of shape [num_layer][num_token][8], 每个值为expert编号
    num_threshold: 判断反常token的阈值
    freq_threshold: 不常用expert的最大选择次数（不超过的都算不常用）
    """
    num_layer = len(l)
    num_token = len(l[0])
    gini_coefficients = []
    token_abnormal_count = defaultdict(int)

    for layer_idx in range(num_layer):
        layer = l[layer_idx]
        flat_experts = [e for token in layer for e in token]
        
        # 统计每个expert被选择的次数
        expert_counts = Counter(flat_experts)
        
        # 用于画洛伦兹曲线
        expert_freqs = [expert_counts[e] for e in sorted(expert_counts)]
        gini_coefficients.append(float(plot_lorenz_curve(expert_freqs)))

        # 找出不常用专家
        infrequent_experts = {e for e, count in expert_counts.items() if count <= freq_threshold * len(flat_experts)}

        for token_idx, token in enumerate(layer):
            num_infrequent = sum(1 for e in token if e in infrequent_experts)
            if num_infrequent > num_threshold:
                token_abnormal_count[token_idx] += 1

    # 判断哪些 token 是“常态异常 token”
    normal_anomalous_tokens = {idx for idx, count in token_abnormal_count.items() if count > ana_threshold}

    print(f"常态异常 token 数量: {len(normal_anomalous_tokens)}")
    print(f"常态异常 token 索引集合: {sorted(normal_anomalous_tokens)}")
    print(f"Gini coefficients for each layer: {gini_coefficients}")
    plot_gini_curve(gini_coefficients)

def plot_gini_curve(gini_coefficients):
    import matplotlib.pyplot as plt
    plt.plot(range(len(gini_coefficients)), gini_coefficients, marker='o')
    plt.xlabel('Layer Index')
    plt.ylabel(f'{metric}')
    plt.title(f'{metric} for Each Layer')
    plt.grid()
    plt.show()

import json
l = json.load(open("motivation_data.txt", "r"))
print(len(l[0]))
# analyze_expert_distribution(l, 2, 1/512, 20)