import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging
import os
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeiboSentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertSentimentClassifier(nn.Module):
    def __init__(self, bert_model: BertModel, num_classes: int = 2):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class SentimentAnalyzer:
    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化情感分析器
        Args:
            model_path: 预训练模型路径
            device: 设备类型 ('cuda' 或 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置模型缓存目录
        cache_dir = os.path.join('models', 'bert-base-chinese')
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # 尝试从本地加载
            self.tokenizer = BertTokenizer.from_pretrained(cache_dir)
            self.bert = BertModel.from_pretrained(cache_dir)
            print("已从本地缓存加载BERT模型")
        except:
            print("正在从Hugging Face下载BERT模型...")
            # 下载并保存到本地
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', 
                                                         cache_dir=cache_dir)
            self.bert = AutoModel.from_pretrained('bert-base-chinese',
                                                cache_dir=cache_dir)
            
            # 保存到本地
            self.tokenizer.save_pretrained(cache_dir)
            self.bert.save_pretrained(cache_dir)
            print("BERT模型已下载并保存到本地")
        
        self.model = BertSentimentClassifier(self.bert)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.model.to(self.device)
        
    def load_model(self, model_path: str):
        """加载预训练模型"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
    def train(self, 
             train_texts: List[str], 
             train_labels: List[int],
             val_texts: List[str] = None,
             val_labels: List[int] = None,
             batch_size: int = 32,
             epochs: int = 3,
             learning_rate: float = 2e-5):
        """训练模型"""
        
        train_dataset = WeiboSentimentDataset(train_texts, train_labels, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_texts and val_labels:
            val_dataset = WeiboSentimentDataset(val_texts, val_labels, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(train_loader)
            logger.info(f'Epoch {epoch+1}/{epochs} - Average training loss: {avg_train_loss:.4f}')
            
            if val_texts and val_labels:
                val_accuracy = self.evaluate(val_loader)
                logger.info(f'Validation Accuracy: {val_accuracy:.4f}')
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                _, predictions = torch.max(outputs, dim=1)
                
                correct_predictions += torch.sum(predictions == labels)
                total_predictions += len(labels)
        
        return correct_predictions.double() / total_predictions
    
    def predict(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """预测文本情感"""
        self.model.eval()
        predictions = []
        probabilities = []
        
        dataset = WeiboSentimentDataset(texts, [0]*len(texts), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=32)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs[:, 1].cpu().numpy())  # 取正面情感的概率
        
        return predictions, probabilities

def main():
    # 示例使用
    analyzer = SentimentAnalyzer()
    
    # 示例训练数据
    train_texts = ["这个产品很好用", "服务态度差", "质量不错", "太贵了"]
    train_labels = [1, 0, 1, 0]  # 1表示正面情感，0表示负面情感
    
    # 训练模型
    analyzer.train(train_texts, train_labels)
    
    # 预测示例
    test_texts = ["这个很不错", "太差劲了"]
    predictions, probabilities = analyzer.predict(test_texts)
    
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        sentiment = "正面" if pred == 1 else "负面"
        print(f"文本: {text}")
        print(f"情感: {sentiment}")
        print(f"正面情感概率: {prob:.4f}\n")

if __name__ == "__main__":
    main() 