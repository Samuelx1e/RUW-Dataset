import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvolutionAnalyzer:
    def __init__(self):
        """初始化演化分析器"""
        pass
    
    def calculate_user_influence(self, followers_count: int, verified: bool, 
                               verified_type: int) -> float:
        """计算用户影响力"""
        # 基础影响力来自粉丝数（取对数平滑）
        base_influence = np.log1p(followers_count) / 20.0  # 归一化到0-1范围
        
        # 认证用户加权
        auth_weight = 1.0
        if verified:
            if verified_type == 3:  # 官方认证
                auth_weight = 2.0
            elif verified_type in [0, 1, 2]:  # 其他认证类型
                auth_weight = 1.5
        
        return min(base_influence * auth_weight, 1.0)  # 确保不超过1
    
    def calculate_text_influence(self, reposts: int, comments: int, 
                               attitudes: int) -> float:
        """计算文本传播影响力"""
        # 计算各项传播指标的对数值
        repost_score = np.log1p(reposts)
        comment_score = np.log1p(comments)
        attitude_score = np.log1p(attitudes)
        
        # 加权平均，转发权重最高
        weighted_score = (0.5 * repost_score + 0.3 * comment_score + 0.2 * attitude_score)
        
        # 归一化到0-1范围
        max_possible_score = np.log1p(10000)  # 假设单条微博最大互动量为10000
        normalized_score = min(weighted_score / max_possible_score, 1.0)
        
        return normalized_score
    
    def calculate_post_influence(self, row: pd.Series) -> float:
        """计算单条微博的综合影响力"""
        text_influence = self.calculate_text_influence(
            row['reposts_count'],
            row['comments_count'],
            row['attitudes_count']
        )
        
        user_influence = self.calculate_user_influence(
            row['followers_count'],
            row['verified'],
            row['verified_type']
        )
        
        # 按0.6:0.4的比例计算综合影响力
        return 0.6 * text_influence + 0.4 * user_influence
    
    def calculate_daily_participation(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算每日参与度"""
        # 确保时间列为datetime类型
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # 计算每条微博的影响力
        df['influence'] = df.apply(self.calculate_post_influence, axis=1)
        
        # 按日期分组计算
        daily_stats = df.groupby(df['created_at'].dt.date).agg({
            'influence': 'sum',
            'msg_id': 'count'  # 计算每日微博数量
        }).reset_index()
        
        # 计算总影响力
        total_influence = daily_stats['influence'].sum()
        
        # 计算日参与度
        daily_stats['participation'] = daily_stats['influence'] / total_influence
        
        # 计算演化指数 = 参与度 * 数据量
        daily_stats['evolution_index'] = daily_stats['participation'] * daily_stats['msg_id']
        
        return daily_stats
    
    def calculate_confrontation_index(self, df: pd.DataFrame, 
                                    sentiment_column: str = 'sentiment') -> pd.DataFrame:
        """计算对抗指数"""
        # 确保时间列为datetime类型
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # 计算每条微博的影响力
        df['influence'] = df.apply(self.calculate_post_influence, axis=1)
        
        # 按日期和情感分组计算
        daily_sentiment = df.groupby([df['created_at'].dt.date, sentiment_column]).agg({
            'influence': 'sum',
            'msg_id': 'count'
        }).reset_index()
        
        # 计算对抗指数
        pivot_table = daily_sentiment.pivot(
            index='created_at',
            columns=sentiment_column,
            values=['influence', 'msg_id']
        ).fillna(0)
        
        # 计算正负情感的归一化权重
        total_influence = pivot_table['influence'].sum().sum()
        confrontation_index = abs(
            pivot_table['influence'][1].values / total_influence - 
            pivot_table['influence'][0].values / total_influence
        )
        
        result_df = pd.DataFrame({
            'date': pivot_table.index,
            'confrontation_index': confrontation_index,
            'positive_count': pivot_table['msg_id'][1],
            'negative_count': pivot_table['msg_id'][0],
            'positive_influence': pivot_table['influence'][1],
            'negative_influence': pivot_table['influence'][0]
        })
        
        return result_df
    
    def plot_evolution_metrics(self, daily_stats: pd.DataFrame, 
                             confrontation_df: pd.DataFrame,
                             save_path: str = None):
        """可视化演化指标"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # 绘制日参与度
        ax1.plot(daily_stats['created_at'], daily_stats['participation'], 
                marker='o', linestyle='-', linewidth=2)
        ax1.set_title('日参与度变化')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('参与度')
        ax1.grid(True)
        
        # 绘制演化指数
        ax2.plot(daily_stats['created_at'], daily_stats['evolution_index'],
                marker='s', linestyle='-', linewidth=2, color='green')
        ax2.set_title('演化指数变化')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('演化指数')
        ax2.grid(True)
        
        # 绘制对抗指数
        ax3.plot(confrontation_df['date'], confrontation_df['confrontation_index'],
                marker='^', linestyle='-', linewidth=2, color='red')
        ax3.set_title('对抗指数变化')
        ax3.set_xlabel('日期')
        ax3.set_ylabel('对抗指数')
        ax3.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def main():
    # 示例使用
    # 创建示例数据
    data = {
        'created_at': pd.date_range(start='2022-02-27', end='2022-03-03', freq='D').repeat(3),
        'reposts_count': np.random.randint(0, 1000, 15),
        'comments_count': np.random.randint(0, 500, 15),
        'attitudes_count': np.random.randint(0, 2000, 15),
        'followers_count': np.random.randint(1000, 100000, 15),
        'verified': [True] * 15,
        'verified_type': np.random.choice([0, 1, 2, 3], 15),
        'sentiment': np.random.choice([0, 1], 15),  # 0表示负面，1表示正面
        'msg_id': range(15)
    }
    df = pd.DataFrame(data)
    
    analyzer = EvolutionAnalyzer()
    
    # 计算每日参与度和演化指数
    daily_stats = analyzer.calculate_daily_participation(df)
    print("\n每日统计:")
    print(daily_stats)
    
    # 计算对抗指数
    confrontation_df = analyzer.calculate_confrontation_index(df)
    print("\n对抗指数:")
    print(confrontation_df)
    
    # 可视化结果
    analyzer.plot_evolution_metrics(daily_stats, confrontation_df, "evolution_metrics.png")

if __name__ == "__main__":
    main() 