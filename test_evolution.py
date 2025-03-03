import pandas as pd
from src.pipeline import Pipeline
from src.evolution.evolution_analyzer import EvolutionAnalyzer
import yaml
from pathlib import Path

def create_mock_config():
    """创建测试用的配置文件"""
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
    
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    with open(config_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    return 'config/config.yaml'

def test_evolution_analysis():
    # 1. 创建更完整的测试数据
    test_data = pd.DataFrame({
        'msg_id': ['id1', 'id2', 'id3'],  # 添加msg_id列
        'text': ['这是测试文本1', '这是测试文本2', '这是测试文本3'],
        'created_at': ['2024-03-01', '2024-03-02', '2024-03-03'],
        'cleaned_text': ['测试文本1', '测试文本2', '测试文本3'],
        'words': [['测试', '文本'], ['测试', '文本'], ['测试', '文本']],
        'sentiment': [1, 0, 1],  # 情感分析结果
        'user_id': ['user1', 'user2', 'user3'],  # 添加user_id列
        'reply_to': [None, 'id1', 'id2'],  # 添加reply_to列
        'repost_id': [None, None, None],  # 添加repost_id列
        'like_count': [10, 5, 8],  # 添加like_count列
        'repost_count': [2, 1, 3],  # 添加repost_count列
        'reply_count': [3, 2, 1]  # 添加reply_count列
    })

    # 2. 创建Pipeline实例并覆盖__init__方法
    class TestPipeline(Pipeline):
        def __init__(self):
            self.config = self._load_config(create_mock_config())
            self.evolution_analyzer = EvolutionAnalyzer()
            self.processed_data = None
            self.evolution_results = None
    
    # 3. 初始化测试Pipeline
    pipeline = TestPipeline()
    
    # 4. 设置processed_data
    pipeline.processed_data = test_data
    
    # 5. 运行演化分析
    try:
        evolution_results = pipeline.analyze_evolution()
        print("演化分析结果：")
        print("\n日常统计：")
        print(evolution_results['daily_stats'])
        print("\n对抗指数：")
        print(evolution_results['confrontation'])
    except Exception as e:
        print(f"测试过程中出现错误：{str(e)}")

if __name__ == "__main__":
    test_evolution_analysis() 