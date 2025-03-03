import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicModeling:
    def __init__(self, min_topics: int = 2, max_topics: int = 10):
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.best_lda_model = None
        self.best_kmeans_model = None
        self.dictionary = None
        self.corpus = None
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def prepare_data(self, texts: List[List[str]]):
        """准备数据"""
        # 为LDA准备数据
        self.dictionary = Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        # 为K-means准备数据
        self.vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        self.tfidf_matrix = self.vectorizer.fit_transform([' '.join(text) for text in texts])
        
    def train_lda(self, texts: List[List[str]]) -> Tuple[LdaModel, float]:
        """训练LDA模型"""
        self.prepare_data(texts)
        
        best_coherence = -np.inf
        best_model = None
        coherence_scores = []
        
        for num_topics in range(self.min_topics, self.max_topics + 1):
            lda_model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10
            )
            
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=texts,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            coherence = coherence_model.get_coherence()
            coherence_scores.append(coherence)
            
            logger.info(f'主题数 {num_topics} 的连贯性分数: {coherence:.4f}')
            
            if coherence > best_coherence:
                best_coherence = coherence
                best_model = lda_model
        
        self.best_lda_model = best_model
        return best_model, best_coherence
    
    def train_kmeans(self, texts: List[List[str]]) -> Tuple[KMeans, float, float]:
        """训练K-means模型"""
        if self.tfidf_matrix is None:
            self.prepare_data(texts)
        
        best_silhouette = -np.inf
        best_ch_score = -np.inf
        best_model = None
        
        silhouette_scores = []
        ch_scores = []
        
        for n_clusters in range(self.min_topics, self.max_topics + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(self.tfidf_matrix)
            
            silhouette = silhouette_score(self.tfidf_matrix, clusters)
            ch_score = calinski_harabasz_score(self.tfidf_matrix.toarray(), clusters)
            
            silhouette_scores.append(silhouette)
            ch_scores.append(ch_score)
            
            logger.info(f'聚类数 {n_clusters}:')
            logger.info(f'轮廓系数: {silhouette:.4f}')
            logger.info(f'Calinski-Harabasz指数: {ch_score:.4f}')
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_ch_score = ch_score
                best_model = kmeans
        
        self.best_kmeans_model = best_model
        return best_model, best_silhouette, best_ch_score
    
    def visualize_clusters(self, save_path: str = None):
        """可视化聚类结果"""
        if self.tfidf_matrix is None or self.best_kmeans_model is None:
            raise ValueError("请先训练模型")
        
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(self.tfidf_matrix.toarray())
        
        # 获取聚类标签
        labels = self.best_kmeans_model.labels_
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('文本聚类可视化 (t-SNE)')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def get_topic_words(self, num_words: int = 10) -> Dict[int, List[str]]:
        """获取每个主题的关键词"""
        if self.best_lda_model is None:
            raise ValueError("请先训练LDA模型")
        
        topic_words = {}
        for topic_id in range(self.best_lda_model.num_topics):
            words = self.best_lda_model.show_topic(topic_id, num_words)
            topic_words[topic_id] = [word for word, _ in words]
        
        return topic_words

def main():
    # 示例使用
    texts = [
        ['俄罗斯', '乌克兰', '战争', '影响'],
        ['经济', '制裁', '影响', '全球'],
        ['和平', '谈判', '停火'],
        ['军事', '行动', '进展'],
    ]
    
    topic_model = TopicModeling(min_topics=2, max_topics=4)
    
    # 训练LDA模型
    lda_model, coherence = topic_model.train_lda(texts)
    print(f"\nLDA最佳连贯性分数: {coherence:.4f}")
    
    # 训练K-means模型
    kmeans_model, silhouette, ch_score = topic_model.train_kmeans(texts)
    print(f"\nK-means最佳轮廓系数: {silhouette:.4f}")
    print(f"K-means最佳Calinski-Harabasz指数: {ch_score:.4f}")
    
    # 获取主题词
    topic_words = topic_model.get_topic_words()
    for topic_id, words in topic_words.items():
        print(f"\n主题 {topic_id + 1} 的关键词: {', '.join(words)}")
    
    # 可视化聚类结果
    topic_model.visualize_clusters("topic_clusters.png")

if __name__ == "__main__":
    main() 