from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

# 定义贝叶斯网络结构
model = BayesianNetwork([('A_strength', 'A_vs_B'), ('B_strength', 'A_vs_B'),
                         ('A_strength', 'A_vs_C'), ('C_strength', 'A_vs_C'),
                         ('B_strength', 'B_vs_C'), ('C_strength', 'B_vs_C')])

# 队伍实力的CPT
cpd_A_strength = TabularCPD(variable='A_strength', variable_card=4, values=[[0.3], [0.3], [0.2], [0.2]])
cpd_B_strength = TabularCPD(variable='B_strength', variable_card=4, values=[[0.4], [0.4], [0.1], [0.1]])
cpd_C_strength = TabularCPD(variable='C_strength', variable_card=4, values=[[0.2], [0.2], [0.3], [0.3]])

# 比赛结果的CPT
# 这里假设获胜概率与队伍实力成比例
def match_outcome_cpd(team1_strength, team2_strength):
    outcome_prob = np.zeros((3, 4, 4))  # 3个结果（胜、平、负） x 4个实力等级 x 4个实力等级
    for i in range(4):
        for j in range(4):
            if i > j:  # 队伍1更强
                outcome_prob[:, i, j] = [0.6, 0.3, 0.1]  # 更可能获胜
            elif i < j:  # 队伍2更强
                outcome_prob[:, i, j] = [0.1, 0.3, 0.6]  # 更可能失败
            else:  # 实力相当
                outcome_prob[:, i, j] = [0.3, 0.4, 0.3]  # 概率均等

    # 将3D数组转换为2D数组
    outcome_prob_2d = outcome_prob.reshape(3, -1)

    return TabularCPD(variable=team1_strength + '_vs_' + team2_strength, variable_card=3,
                          values=outcome_prob_2d, evidence=[team1_strength + '_strength', team2_strength + '_strength'],
                          evidence_card=[4, 4])

cpd_A_vs_B = match_outcome_cpd('A', 'B')
cpd_A_vs_C = match_outcome_cpd('A', 'C')
cpd_B_vs_C = match_outcome_cpd('B', 'C')

# 将CPDs添加到模型中
model.add_cpds(cpd_A_strength, cpd_B_strength, cpd_C_strength, cpd_A_vs_B, cpd_A_vs_C, cpd_B_vs_C)

# 检查模型是否有效
model.check_model()

# 初始化推理方法
inference = VariableElimination(model)

# 精确求解的函数
def exact_inference(evidence):
    result = inference.query(variables=['B_vs_C'], evidence=evidence)
    num_states = len(result.values)
    probabilities = {state: round(result.values[state], 3) for state in range(num_states)}
    return probabilities


# 拒绝采样的函数
def rejection_sampling(evidence, num_samples=1000):
    samples = []
    for _ in range(num_samples):
        sample = {}
        for cpd in model.cpds:
            var = cpd.variable
            if var not in evidence:
                # 生成一个随机样本
                sample[var] = np.random.choice(range(cpd.variable_card))
            else:
                # 对于证据节点，使用证据值
                sample[var] = evidence[var]

        # 检查生成的样本是否符合证据
        consistent_with_evidence = all(sample[var] == val for var, val in evidence.items())
        if consistent_with_evidence:
            # 只有当样本与证据一致时，才将其添加到结果中
            samples.append(sample['B_vs_C'])

    # 计算每种结果的频率
    counts = np.bincount(samples, minlength=3)
    total = counts.sum()
    probabilities = {state: counts[state] / total for state in range(len(counts))}

    return probabilities


# 似然加权采样和Gibbs采样的占位函数
def likelihood_weighting(evidence, num_samples=1000):
    weights = {0: 0, 1: 0, 2: 0}  # 初始化权重

    for _ in range(num_samples):
        sample, weight = {}, 1

        # 为证据变量设置值，并计算权重
        for ev, ev_value in evidence.items():
            cpd = model.get_cpds(ev)
            parent_states = [sample[parent] if parent in sample else np.random.choice(range(model.get_cpds(parent).variable_card)) for parent in cpd.variables[1:]]
            state_index = tuple([ev_value] + parent_states)
            weight *= cpd.values[state_index]

        # 为非证据变量生成样本
        for cpd in model.cpds:
            var = cpd.variable
            if var not in evidence:
                # 为 reduce 函数创建正确格式的输入
                evidence_for_var = [(var, sample[var]) for var in cpd.variables if var != cpd.variable and var in sample]
                reduced_cpd = cpd.reduce(evidence_for_var, inplace=False)

                # 生成随机样本
                var_sample = np.random.choice(cpd.cardinality[0], p=reduced_cpd.values.flatten())
                sample[var] = var_sample

        # 更新权重
        weights[sample['B_vs_C']] += weight

    # 归一化权重以得到概率分布
    total_weight = sum(weights.values())
    probabilities = {k: round(v / total_weight, 3) for k, v in weights.items()}

    return probabilities


def gibbs_sampling(evidence, num_samples):
    # 初始化样本，可以随机初始化或使用证据值初始化
    sample = {var: np.random.choice(range(cpd.variable_card)) for cpd in model.cpds for var in cpd.variables}
    sample.update(evidence)

    # 更新查询变量的计数器
    counts = {0: 0, 1: 0, 2: 0}

    for _ in range(num_samples):
        for var in model.nodes():
            if var not in evidence:
                # 获取该变量的条件概率分布
                cpd = model.get_cpds(var)

                # 计算条件概率分布，给定其他变量的当前值
                var_evidence = [(ev, sample[ev]) for ev in cpd.variables if ev != var]
                reduced_cpd = cpd.reduce(var_evidence, inplace=False)
                probabilities = reduced_cpd.values.flatten()

                # 从条件概率分布中采样新值
                sample[var] = np.random.choice(cpd.cardinality[0], p=probabilities)

        # 更新查询变量的计数
        counts[sample['B_vs_C']] += 1

    # 计算每种结果的频率作为概率估计
    total_samples = sum(counts.values())
    probabilities = {k: v / total_samples for k, v in counts.items()}

    return probabilities


# 使用证据来测试精确推理
evidence = {'A_vs_B': 0, 'A_vs_C': 2}  # 0: 胜, 1: 平, 2: 负

exact_result0 = exact_inference(evidence)
print(exact_result0)

exact_result1 = rejection_sampling(evidence)
print(exact_result1)

exact_result2 = likelihood_weighting(evidence)
print(exact_result2)

gibbs_result = gibbs_sampling(evidence, 1000)
print(gibbs_result)
# 显示精确推理的结果


