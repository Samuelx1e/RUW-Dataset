import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import argparse

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
        # 只读取前200行数据用于测试
        # return pd.read_csv(data_path).head(50)
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
        texts = self.processed_data['cleaned_text'].tolist()
        predictions = []
        probabilities = []
        
        for text in tqdm(texts, desc="情感分析进度"):
            pred, prob = self.sentiment_analyzer.predict([text])
            predictions.extend(pred)
            probabilities.extend(prob)
            
        # 将情感分析结果保存到数据框中
        self.processed_data['sentiment'] = predictions
        self.processed_data['sentiment_probability'] = probabilities
            
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
        
        # 检查数据是否包含必要的列
        required_columns = ['created_at', 'text']
        if not all(col in self.processed_data.columns for col in required_columns):
            raise ValueError(f"数据中缺少必要的列: {required_columns}")
        
        # 创建数据副本进行处理
        analysis_df = self.processed_data.copy()
        
        # 验证created_at列的数据类型和内容
        if not pd.api.types.is_datetime64_any_dtype(analysis_df['created_at']):
            try:
                # 尝试将created_at列转换为datetime格式
                analysis_df['created_at'] = pd.to_datetime(
                    analysis_df['created_at'], 
                    format='mixed',  # 使用混合格式解析
                    errors='coerce'  # 将无法解析的值设为NaT
                )
            except Exception as e:
                logger.error(f"日期时间转换失败: {str(e)}")
                raise
        
        # 移除无效的日期时间记录
        analysis_df = analysis_df.dropna(subset=['created_at'])
        
        if len(analysis_df) == 0:
            raise ValueError("处理后没有有效的数据记录")
        
        # 计算每日参与度
        daily_stats = self.evolution_analyzer.calculate_daily_participation(
            analysis_df
        )
        
        # 计算对抗指数
        confrontation_df = self.evolution_analyzer.calculate_confrontation_index(
            analysis_df,
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
        if (self.processed_data is None or self.topic_results is None or 
            self.evolution_results is None or self.keyword_results is None):
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
        # self.visualize()
    
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

def load_from_output(output_dir: str = 'output'):
    """从输出目录加载已保存的分析结果并可视化"""
    logger.info("从输出文件加载分析结果...")
    pipeline = Pipeline()
    output_dir = Path(output_dir)
    
    # 加载处理后的数据
    pipeline.processed_data = pd.read_csv(output_dir / 'processed_data.csv')
    
    # 加载情感分析结果
    sentiment_df = pd.read_csv(output_dir / 'sentiment_results.csv')
    pipeline.sentiment_results = (
        sentiment_df['sentiment'].tolist(),
        sentiment_df['probability'].tolist()
    )
    
    # 加载主题分析结果
    topic_df = pd.read_csv(output_dir / 'topic_results.csv')
    pipeline.topic_results = {
        'topic_words': {
            f'主题{i+1}': topic_df[col].dropna().tolist()
            for i, col in enumerate(topic_df.columns)
        }
    }
    
    # 加载关键词结果
    try:
        keyword_df = pd.read_csv(output_dir / 'keyword_results.csv')
        logger.info(f"关键词数据形状: {keyword_df.shape}")
        logger.info(f"关键词数据列名: {keyword_df.columns.tolist()}")
        logger.info(f"关键词数据前几行:\n{keyword_df.head()}")
        
        # 将DataFrame转换为简单的词频字典
        pipeline.keyword_results = {}
        
        # 检查数据格式
        if not keyword_df.empty:
            # 如果是索引和值的格式
            if 'Unnamed: 0' in keyword_df.columns:
                word_col = 'Unnamed: 0'
                freq_col = keyword_df.columns[1]
            # 如果是普通的两列格式
            elif len(keyword_df.columns) >= 2:
                word_col = keyword_df.columns[0]
                freq_col = keyword_df.columns[1]
            else:
                raise ValueError("关键词数据格式不正确")
            
            # 转换数据
            for _, row in keyword_df.iterrows():
                word = str(row[word_col])
                try:
                    freq = float(row[freq_col])
                    if pd.notnull(freq) and word != 'nan':
                        pipeline.keyword_results[word] = freq
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        logger.error(f"加载关键词数据时出错: {str(e)}")
        logger.warning("使用示例关键词数据")
        pipeline.keyword_results = {
            'Example1': 100.0,
            'Example2': 80.0,
            'Example3': 60.0,
            'Example4': 40.0,
            'Example5': 20.0
        }
    
    logger.info(f"最终处理后的关键词数量: {len(pipeline.keyword_results)}")
    if len(pipeline.keyword_results) > 0:
        logger.info("部分关键词示例:")
        for word, freq in list(pipeline.keyword_results.items())[:5]:
            logger.info(f"{word}: {freq}")
    else:
        logger.warning("没有有效的关键词数据，使用示例数据")
        pipeline.keyword_results = {
            'Example1': 100.0,
            'Example2': 80.0,
            'Example3': 60.0,
            'Example4': 40.0,
            'Example5': 20.0
        }
    
    # 加载演化分析结果
    daily_stats = pd.read_csv(output_dir / 'evolution_daily.csv')
    # 添加participation列
    daily_stats['participation'] = daily_stats['total_influence']
    
    # 加载对抗指数数据
    confrontation = pd.read_csv(output_dir / 'evolution_confrontation.csv')
    # 确保daily_stats包含所有必要的列
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    daily_stats['confrontation_index'] = confrontation['confrontation_index']
    
    pipeline.evolution_results = {
        'daily_stats': daily_stats,
        'confrontation': confrontation
    }
    
    logger.info("数据加载完成，准备可视化...")
    
    # 设置字体路径
    import matplotlib.font_manager as fm
    import platform
    
    # 根据操作系统选择合适的字体
    if platform.system() == 'Windows':
        font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑
    elif platform.system() == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/PingFang.ttc'  # 苹方
    else:  # Linux
        font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
    
    # 检查字体文件是否存在
    if not Path(font_path).exists():
        # 尝试找到系统中的其他中文字体
        chinese_fonts = [f for f in fm.findSystemFonts() if 'simhei' in f.lower() or 'msyh' in f.lower()]
        if chinese_fonts:
            font_path = chinese_fonts[0]
        else:
            # 如果找不到中文字体，使用默认字体
            font_path = None
    
    # 更新Dashboard类的字体设置
    Dashboard.font_path = font_path
    
    pipeline.visualize()

def main():
    parser = argparse.ArgumentParser(description='运行分析Pipeline或加载已有结果进行可视化')
    parser.add_argument('--mode', type=str, choices=['run', 'visualize'], 
                       default='visualize', help='选择运行模式：完整分析(run)或仅可视化(visualize)')
    args = parser.parse_args()
    
    if args.mode == 'run':
        # 创建配置文件
        config = {
            'preprocessing': {
                'stopwords_path': '../data/stopwords.txt'
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
        pipeline.run('../data/RU_Dataset_cleaned.csv')
    else:
        # 直接加载输出文件并可视化
        load_from_output()

if __name__ == "__main__":
    main()