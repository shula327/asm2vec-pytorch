import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel
from asm2vec.model import ASM2VEC

# 定义对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# 示例数据集类
class CodeDataset(Dataset):
    def __init__(self, asm_data, src_data, labels):
        self.asm_data = asm_data
        self.src_data = src_data
        self.labels = labels

    def __len__(self):
        return len(self.asm_data)

    def __getitem__(self, idx):
        return self.asm_data[idx], self.src_data[idx], self.labels[idx]

# 准备示例数据
asm_data = [
    "mov eax, ebx\nadd eax, 1\nret",  # 汇编代码示例1
    "mov ecx, edx\nsub ecx, 2\nret"   # 汇编代码示例2
]
src_data = [
    "int add(int a, int b) { return a + b; }",  # 源代码示例1
    "int subtract(int a, int b) { return a - b; }"  # 源代码示例2
]
labels = [1, 0]  # 标签，1 表示相似，0 表示不相似

# 创建数据集和数据加载器
dataset = CodeDataset(asm_data, src_data, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 初始化 ASM2VEC 模型
vocab_size = 10000  # 词汇表大小
function_size = 1000  # 函数数量
embedding_size = 128  # 嵌入向量维度
asm_model = ASM2VEC(vocab_size, function_size, embedding_size)
asm_model.embeddings = nn.Embedding(50200, embedding_size)  # 调整嵌入矩阵的大小
asm_model = asm_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 BERT 模型
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
bert_model = RobertaModel.from_pretrained("microsoft/codebert-base")
bert_model = bert_model.to('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.eval()

# 定义对比损失函数和优化器
criterion = ContrastiveLoss()
optimizer = optim.Adam(list(asm_model.parameters()) + list(bert_model.parameters()), lr=0.001)

# 训练模型
epochs = 5
linear = nn.Linear(512, 768).to('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(epochs):
    asm_model.train()
    total_loss = 0
    for asm_code, src_code, label in dataloader:
        # 获取汇编代码嵌入向量
        asm_code = tokenizer(asm_code, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda' if torch.cuda.is_available() else 'cpu')
        print("ASM Code Input IDs:", asm_code['input_ids'])  # 打印输入数据
        max_index = asm_model.embeddings.num_embeddings - 1
        print(f"Embedding matrix size: {asm_model.embeddings.num_embeddings}")
        print(f"Max index in input data: {torch.max(asm_code['input_ids'])}")
        if torch.any(asm_code['input_ids'] > max_index):
            raise ValueError("Index out of range in input data")
        asm_embedding = asm_model.v(asm_code['input_ids'])
        asm_embedding = asm_embedding.view(-1, 512)  # 调整为合适的形状
        asm_embedding = linear(asm_embedding)

        # 获取源代码嵌入向量
        src_embedding = []
        for code in src_code:
            inputs = tokenizer(code, return_tensors='pt', truncation=True, padding=True, max_length=512).to('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                outputs = bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            src_embedding.append(cls_embedding)
        src_embedding = torch.stack(src_embedding).squeeze(1)

        # 计算对比损失
        label = label.to('cuda' if torch.cuda.is_available() else 'cpu').float()
        loss = criterion(asm_embedding, src_embedding, label)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    asm_embedding_norm = asm_embedding / asm_embedding.norm(dim=1, keepdim=True)
    src_embedding_norm = src_embedding / src_embedding.norm(dim=1, keepdim=True)
    cosine_sim = torch.mm(asm_embedding_norm, src_embedding_norm.t())
    print(f'Cosine Similarity after Epoch {epoch+1}:')
    print(cosine_sim)