import matplotlib.pyplot as plt

# 输出的概率分布
exact_inference_probs = {0: 0.152, 1: 0.312, 2: 0.537}
rejection_sampling_probs = {0: 0.327, 1: 0.331, 2: 0.342}
likelihood_weighting_probs = {0: 0.231, 1: 0.327, 2: 0.442}
gibbs_sampling_probs = {0: 0.238, 1: 0.307, 2: 0.455}

# 将概率分布转换为列表
methods = ['Exact Inference', 'Rejection Sampling', 'Likelihood Weighting', 'Gibbs Sampling']
prob_0 = [exact_inference_probs[0], rejection_sampling_probs[0], likelihood_weighting_probs[0], gibbs_sampling_probs[0]]
prob_1 = [exact_inference_probs[1], rejection_sampling_probs[1], likelihood_weighting_probs[1], gibbs_sampling_probs[1]]
prob_2 = [exact_inference_probs[2], rejection_sampling_probs[2], likelihood_weighting_probs[2], gibbs_sampling_probs[2]]

# 绘制条形图
x = range(len(methods))
plt.bar(x, prob_0, width=0.2, label='B wins (0)')
plt.bar(x, prob_1, width=0.2, label='Draw (1)', bottom=prob_0)
for i in range(len(prob_0)):
    prob_0[i] += prob_1[i]
plt.bar(x, prob_2, width=0.2, label='C wins (2)', bottom=prob_0)

plt.xlabel('Methods')
plt.ylabel('Probabilities')
plt.title('Comparison of Different Inference Methods')
plt.xticks(x, methods)
plt.legend()

# 显示图表
plt.show()

