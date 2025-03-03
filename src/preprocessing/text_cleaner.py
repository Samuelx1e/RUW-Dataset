import pandas as pd
import jieba
import re
from typing import List, Set
import os
import logging

class TextCleaner:
    def __init__(self, stopwords_path: str = None):
        """
        初始化文本清洗器
        Args:
            stopwords_path: 停用词表路径
        """
        self.stopwords = self._load_stopwords(stopwords_path) if stopwords_path else set()
        jieba.setLogLevel(logging.INFO)
    
    def _load_stopwords(self, path: str) -> Set[str]:
        """
        加载停用词表
        """
        if not os.path.exists(path):
            print(f"警告: 停用词表文件 {path} 不存在")
            return set()
        with open(path, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f])
    
    def clean_text(self, text: str) -> str:
        """
        清理单条文本
        """
        if not isinstance(text, str):
            return ""
        
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # 去除@用户
        text = re.sub(r'@[\w\-]+', '', text)
        # 去除话题标签
        text = re.sub(r'#.*?#', '', text)
        # 去除特殊符号
        text = re.sub(r'[^\w\s]', '', text)
        # 去除多余空格
        text = ' '.join(text.split())
        
        return text
    
    def segment_text(self, text: str) -> List[str]:
        """
        对文本进行分词
        """
        words = jieba.lcut(text)
        # 去除停用词
        words = [w for w in words if w not in self.stopwords and len(w.strip()) > 0]
        return words
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理整个DataFrame
        """
        # 复制DataFrame避免修改原始数据
        df = df.copy()
        
        # 去重
        df = df.drop_duplicates(subset=['text'])
        
        # 清理文本
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # 分词
        df['words'] = df['cleaned_text'].apply(self.segment_text)
        
        # 过滤空文本
        df = df[df['words'].apply(len) > 0]
        
        return df

def main():
    # 示例使用
    cleaner = TextCleaner()
    df = pd.read_csv('data/RU_Dataset_cleaned.csv')
    processed_df = cleaner.process_dataframe(df)
    print(f"处理前数据量: {len(df)}")
    print(f"处理后数据量: {len(processed_df)}")
    print("\n示例处理结果:")
    print(processed_df[['text', 'cleaned_text', 'words']].head())

if __name__ == "__main__":
    main() 