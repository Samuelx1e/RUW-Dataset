import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yaml
import logging
from pathlib import Path

from preprocessing.text_cleaner import TextCleaner
from sentiment.bert_classifier import SentimentAnalyzer
from topic.topic_modeling import TopicModeling
from keywords.keyword_extractor import KeywordExtractor
from evolution.evolution_analyzer import EvolutionAnalyzer
from visualization.dashboard import Dashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        初始化Pipeline
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.text_cleaner = TextCleaner(
            stopwords_path=self.config['preprocessing']['stopwords_path']
        )
        self.sentiment_analyzer = SentimentAnalyzer(
            model_path=self.config['sentiment']['model_path']
        )
        self.topic_modeling = TopicModeling(
            min_topics=self.config['topic']['min_topics'],
            max_topics=self.config['topic']['max_topics']
        )
        self.keyword_extractor = KeywordExtractor(
            top_k=self.config['keywords']['top_k']
        )
        self.evolution_analyzer = EvolutionAnalyzer()
        
        self.processed_data = None
        self.sentiment_results = None
        self.topic_results = None
        self.keyword_results = None
        self.evolution_results = None
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """加载数据"""
        logger.info(f"正在加载数据: {data_path}")
        return pd.read_csv(data_path)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        logger.info("开始数据预处理...")
        processed_df = self.text_cleaner.process_dataframe(df)
        self.processed_data = processed_df
        return processed_df
    
    def analyze_sentiment(self) -> Tuple[List[int], List[float]]:
        """情感分析"""
        logger.info("开始情感分析...")
        predictions, probabilities = self.sentiment_analyzer.predict(
            self.processed_data['cleaned_text'].tolist()
        )
        self.sentiment_results = (predictions, probabilities)
        return predictions, probabilities
    
    def analyze_topics(self) -> Dict:
        """主题分析"""
        logger.info("开始主题分析...")
        # 训练LDA模型
        lda_model, coherence = self.topic_modeling.train_lda(
            self.processed_data['words'].tolist()
        )
        
        # 训练K-means模型
        kmeans_model, silhouette, ch_score = self.topic_modeling.train_kmeans(
            self.processed_data['words'].tolist()
        )
        
        # 获取主题词
        topic_words = self.topic_modeling.get_topic_words()
        
        self.topic_results = {
            'lda_model': lda_model,
            'kmeans_model': kmeans_model,
            'topic_words': topic_words,
            'metrics': {
                'coherence': coherence,
                'silhouette': silhouette,
                'ch_score': ch_score
            }
        }
        return self.topic_results
    
    def extract_keywords(self) -> Dict[int, List[str]]:
        """关键词提取"""
        logger.info("开始关键词提取...")
        # 使用两种算法提取关键词
        tfidf_keywords = self.keyword_extractor.extract_tfidf_keywords(
            self.processed_data['words'].tolist(),
            self.sentiment_results[0] if self.sentiment_results else None
        )
        
        textrank_keywords = self.keyword_extractor.extract_textrank_keywords(
            self.processed_data['words'].tolist(),
            self.sentiment_results[0] if self.sentiment_results else None
        )
        
        # 合并结果
        combined_keywords = self.keyword_extractor.combine_keywords(
            tfidf_keywords, textrank_keywords
        )
        
        self.keyword_results = combined_keywords
        return combined_keywords
    
    def analyze_evolution(self) -> Dict:
        """演化分析"""
        logger.info("开始演化分析...")
        # 计算每日参与度
        daily_stats = self.evolution_analyzer.calculate_daily_participation(
            self.processed_data
        )
        
        # 计算对抗指数
        confrontation_df = self.evolution_analyzer.calculate_confrontation_index(
            self.processed_data,
            sentiment_column='sentiment'
        )
        
        self.evolution_results = {
            'daily_stats': daily_stats,
            'confrontation': confrontation_df
        }
        return self.evolution_results
    
    def visualize(self, port: int = 8050):
        """启动可视化仪表盘"""
        logger.info("启动可视化仪表盘...")
        if not all([self.processed_data, self.topic_results, 
                   self.evolution_results, self.keyword_results]):
            raise ValueError("请先运行完整的分析流程")
        
        dashboard = Dashboard(
            df=self.processed_data,
            topic_data=self.topic_results['topic_words'],
            evolution_data=self.evolution_results['daily_stats'],
            keyword_data=self.keyword_results
        )
        dashboard.run_server(port=port)
    
    def run(self, data_path: str):
        """运行完整的分析流程"""
        # 加载数据
        df = self.load_data(data_path)
        
        # 预处理
        self.preprocess(df)
        
        # 情感分析
        self.analyze_sentiment()
        
        # 主题分析
        self.analyze_topics()
        
        # 关键词提取
        self.extract_keywords()
        
        # 演化分析
        self.analyze_evolution()
        
        # 保存结果
        self.save_results()
        
        # 启动可视化
        self.visualize()
    
    def save_results(self):
        """保存分析结果"""
        output_dir = Path(self.config['output']['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存处理后的数据
        self.processed_data.to_csv(
            output_dir / 'processed_data.csv',
            index=False
        )
        
        # 保存情感分析结果
        pd.DataFrame({
            'text': self.processed_data['text'],
            'sentiment': self.sentiment_results[0],
            'probability': self.sentiment_results[1]
        }).to_csv(output_dir / 'sentiment_results.csv', index=False)
        
        # 保存主题分析结果
        pd.DataFrame(self.topic_results['topic_words']).to_csv(
            output_dir / 'topic_results.csv'
        )
        
        # 保存关键词结果
        pd.DataFrame(self.keyword_results).to_csv(
            output_dir / 'keyword_results.csv'
        )
        
        # 保存演化分析结果
        self.evolution_results['daily_stats'].to_csv(
            output_dir / 'evolution_daily.csv',
            index=False
        )
        self.evolution_results['confrontation'].to_csv(
            output_dir / 'evolution_confrontation.csv',
            index=False
        )

def main():
    # 创建配置文件
    config = {
        'preprocessing': {
            'stopwords_path': 'data/stopwords.txt'
        },
        'sentiment': {
            'model_path': 'models/sentiment_model.pt'
        },
        'topic': {
            'min_topics': 2,
            'max_topics': 10
        },
        'keywords': {
            'top_k': 10
        },
        'output': {
            'dir': 'output'
        }
    }
    
    # 保存配置文件
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    with open(config_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    # 运行Pipeline
    pipeline = Pipeline()
    pipeline.run('data/RU_Dataset_cleaned.csv')

if __name__ == "__main__":
    main() 