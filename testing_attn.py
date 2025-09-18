import torch


def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

# Your journey starts with one step
# 3d embeddings for phrase
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]
    ]
)

# Step 1 - Compute intermediate values omega (attention scores)
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0]) # 2 appears after every var bc it's the second word we calc for
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
#print("Weights: S", attn_weights_2_tmp)
#print("Sum: ", attn_weights_2_tmp.sum())

attn_weights_2_naive = softmax_naive(attn_scores_2)
#print("Weights: S", attn_weights_2_naive)
#print("Sum: ", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Weights: S", attn_weights_2)
print("Sum: ", attn_weights_2.sum())

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)