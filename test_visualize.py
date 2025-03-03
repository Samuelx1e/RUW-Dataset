import pandas as pd
import numpy as np
from src.pipeline import Pipeline
from wordcloud import WordCloud
import matplotlib.font_manager as fm

def test_visualize():
    # 获取系统中的中文字体
    fonts = [f for f in fm.findSystemFonts() if 'simhei' in f.lower() or 'msyh' in f.lower()]
    if not fonts:
        font_path = 'C:/Windows/Fonts/simhei.ttf'
    else:
        font_path = fonts[0]
    
    # 创建Pipeline实例，但跳过实际的模型初始化
    pipeline = Pipeline.__new__(Pipeline)
    
    # 创建更完整的测试数据
    pipeline.processed_data = pd.DataFrame({
        'text': ['测试文本1', '测试文本2', '测试文本3'],
        'cleaned_text': ['测试', '文本', '示例'],
        'sentiment': [1, 0, 1],
        'created_at': pd.date_range(start='2024-01-01', periods=3),
        'attitudes_count': [100, 200, 300],
        'reposts_count': [10, 20, 30],
        'comments_count': [5, 15, 25],
        'user_id': ['user1', 'user2', 'user3'],
        'user_followers_count': [1000, 2000, 3000],
        'user_friends_count': [500, 1000, 1500],
        'words': [['测试', '文本'], ['示例', '数据'], ['可视化', '分析']],
        'sentiment_probability': [0.8, 0.7, 0.9]
    })
    
    # 准备测试数据
    # 使用嵌套字典格式
    topic_words = {
        'Topic 1': {'词1': 10, '词2': 8, '词3': 6},
        'Topic 2': {'词4': 9, '词5': 7, '词6': 5}
    }
    
    evolution_data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=3),
        'post_count': [10, 15, 20],
        'user_count': [5, 8, 10],
        'engagement_rate': [0.1, 0.2, 0.3],
        'avg_sentiment': [0.6, 0.5, 0.7],
        'participation': [50, 75, 100],
        'total_interactions': [150, 200, 250],
        'confrontation_index': [0.3, 0.5, 0.7]
    })
    
    # 使用嵌套字典格式
    keyword_data = {
        'positive': {'好': 10, '优秀': 8, '棒': 6},
        'negative': {'差': 9, '糟糕': 7, '坏': 5},
        'neutral': {'一般': 8, '普通': 6, '正常': 4}
    }
    
    try:
        from src.visualization.dashboard import Dashboard
        
        class ChineseDashboard(Dashboard):
            def generate_wordcloud(self, word_freq, title=None):
                # 如果是嵌套字典，合并为单层字典
                if isinstance(word_freq, dict) and any(isinstance(v, dict) for v in word_freq.values()):
                    combined_freq = {}
                    for category in word_freq.values():
                        if isinstance(category, dict):
                            combined_freq.update(category)
                    word_freq = combined_freq
                
                # 打印调试信息
                print(f"Generating wordcloud for: {title}")
                print("Word frequencies:", word_freq)
                
                wc = WordCloud(
                    font_path=font_path,
                    width=400,
                    height=200,
                    background_color='white',
                    min_font_size=10,
                    max_font_size=100
                )
                wc.generate_from_frequencies(word_freq)
                return wc.to_image()
        
        # 使用支持中文的Dashboard
        dashboard = ChineseDashboard(
            df=pipeline.processed_data,
            topic_data=topic_words,
            evolution_data=evolution_data,
            keyword_data=keyword_data
        )
        dashboard.run_server(port=8051)
        print("可视化仪表盘启动成功！")
    except Exception as e:
        print(f"可视化失败: {str(e)}")
        print(f"使用的字体路径: {font_path}")
        # 打印更多调试信息
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualize()