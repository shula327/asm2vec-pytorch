from transformers import RobertaTokenizer, RobertaModel
import torch


# 加载 CodeBERT 的分词器和模型
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")


# 将模型设置为评估模式
model.eval()


def get_code_embedding(code: str):
    # 对源代码进行分词
    inputs = tokenizer(code, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取[CLS]标记的输出作为句子向量
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding


# 示例源代码
code_sample = """
def add(a, b):
    return a + b
"""

# 获取源代码的向量表示
embedding = get_code_embedding(code_sample)
print(embedding)

