import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import numpy as np
from collections import defaultdict
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordExtractor:
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x
        )
        
    def extract_tfidf_keywords(self, texts: List[List[str]], class_labels: List[int] = None) -> Dict[int, List[str]]:
        """使用TF-IDF提取关键词"""
        # 将分词列表转换为文本
        text_docs = [' '.join(text) for text in texts]
        
        # 计算TF-IDF矩阵
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_docs)
        feature_names = np.array(self.tfidf_vectorizer.get_feature_names_out())
        
        if class_labels is None:
            # 如果没有类别标签，将所有文本视为一个类别
            class_labels = [0] * len(texts)
        
        # 按类别提取关键词
        class_keywords = {}
        for label in set(class_labels):
            # 获取该类别的文档索引
            class_docs_idx = np.where(np.array(class_labels) == label)[0]
            
            # 计算该类别文档的平均TF-IDF值
            class_tfidf = tfidf_matrix[class_docs_idx].mean(axis=0).A1
            
            # 获取top_k关键词
            top_indices = class_tfidf.argsort()[-self.top_k:][::-1]
            class_keywords[label] = feature_names[top_indices].tolist()
        
        return class_keywords
    
    def extract_textrank_keywords(self, texts: List[List[str]], class_labels: List[int] = None) -> Dict[int, List[str]]:
        """使用TextRank提取关键词"""
        if class_labels is None:
            class_labels = [0] * len(texts)
        
        # 按类别组织文本
        class_texts = defaultdict(list)
        for text, label in zip(texts, class_labels):
            class_texts[label].extend(text)
        
        class_keywords = {}
        for label, words in class_texts.items():
            # 构建词图
            word_graph = self._build_word_graph(words)
            
            # 计算TextRank得分
            scores = nx.pagerank(word_graph)
            
            # 获取top_k关键词
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            class_keywords[label] = [word for word, score in sorted_words[:self.top_k]]
        
        return class_keywords
    
    def _build_word_graph(self, words: List[str], window_size: int = 3) -> nx.Graph:
        """构建词图"""
        graph = nx.Graph()
        
        # 添加节点
        for word in set(words):
            graph.add_node(word)
        
        # 添加边
        for i in range(len(words)):
            for j in range(i + 1, min(i + window_size, len(words))):
                if words[i] != words[j]:
                    if graph.has_edge(words[i], words[j]):
                        graph[words[i]][words[j]]['weight'] += 1
                    else:
                        graph.add_edge(words[i], words[j], weight=1)
        
        return graph
    
    def combine_keywords(self, tfidf_keywords: Dict[int, List[str]], 
                        textrank_keywords: Dict[int, List[str]]) -> Dict[int, List[str]]:
        """合并两种算法的结果"""
        combined_keywords = {}
        
        for label in tfidf_keywords.keys():
            # 获取两种算法的关键词
            tfidf_words = set(tfidf_keywords[label])
            textrank_words = set(textrank_keywords[label])
            
            # 优先选择同时出现在两个结果中的词
            common_words = tfidf_words.intersection(textrank_words)
            
            # 如果共同词不足top_k，则从两个结果中补充
            remaining_words = []
            if len(common_words) < self.top_k:
                remaining = self.top_k - len(common_words)
                unique_tfidf = list(tfidf_words - common_words)
                unique_textrank = list(textrank_words - common_words)
                
                # 交替从两个结果中选择词
                for i in range(remaining):
                    if i < len(unique_tfidf) and i % 2 == 0:
                        remaining_words.append(unique_tfidf[i//2])
                    elif i < len(unique_textrank):
                        remaining_words.append(unique_textrank[i//2])
            
            # 合并结果
            combined_keywords[label] = list(common_words) + remaining_words[:self.top_k-len(common_words)]
        
        return combined_keywords

def main():
    # 示例使用
    texts = [
        ['俄罗斯', '乌克兰', '战争', '影响', '制裁'],
        ['经济', '制裁', '影响', '全球', '金融'],
        ['和平', '谈判', '停火', '外交'],
        ['军事', '行动', '进展', '战略']
    ]
    class_labels = [0, 0, 1, 1]  # 两个类别的示例
    
    extractor = KeywordExtractor(top_k=5)
    
    # 使用TF-IDF提取关键词
    tfidf_keywords = extractor.extract_tfidf_keywords(texts, class_labels)
    print("\nTF-IDF关键词:")
    for label, keywords in tfidf_keywords.items():
        print(f"类别 {label}: {', '.join(keywords)}")
    
    # 使用TextRank提取关键词
    textrank_keywords = extractor.extract_textrank_keywords(texts, class_labels)
    print("\nTextRank关键词:")
    for label, keywords in textrank_keywords.items():
        print(f"类别 {label}: {', '.join(keywords)}")
    
    # 合并两种算法的结果
    combined_keywords = extractor.combine_keywords(tfidf_keywords, textrank_keywords)
    print("\n合并后的关键词:")
    for label, keywords in combined_keywords.items():
        print(f"类别 {label}: {', '.join(keywords)}")

if __name__ == "__main__":
    main() 