import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List
import logging
import matplotlib.font_manager as fm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dashboard:
    def __init__(self, df: pd.DataFrame, topic_data: Dict, 
                 evolution_data: pd.DataFrame, keyword_data: Dict):
        """
        初始化仪表盘
        Args:
            df: 原始数据DataFrame
            topic_data: 主题聚类结果
            evolution_data: 演化分析结果
            keyword_data: 关键词提取结果
        """
        self.df = df
        self.topic_data = topic_data
        self.evolution_data = evolution_data
        self.keyword_data = keyword_data
        self.app = dash.Dash(__name__)
        
        # 尝试找到系统中可用的字体
        self.font_path = self._get_system_font()
        
        self.setup_layout()
        self.setup_callbacks()
    
    def _get_system_font(self):
        """获取系统中可用的字体"""
        try:
            # 获取系统字体列表
            system_fonts = fm.findSystemFonts()
            
            # 优先尝试常见的中文字体
            chinese_fonts = [
                'SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun',
                'FangSong', 'KaiTi', 'STHeiti', 'STKaiti', 'STSong',
                'STFangsong', 'PingFang SC', 'Hiragino Sans GB'
            ]
            
            # 在系统字体中查找中文字体
            for font_name in chinese_fonts:
                for font_path in system_fonts:
                    if font_name.lower() in font_path.lower():
                        logger.info(f"找到中文字体: {font_path}")
                        return font_path
            
            # 如果找不到中文字体，使用任意可用的系统字体
            if system_fonts:
                logger.warning("未找到中文字体，使用系统默认字体")
                return system_fonts[0]
            
            logger.warning("未找到任何可用字体")
            return None
            
        except Exception as e:
            logger.error(f"获取系统字体时出错: {str(e)}")
            return None
    
    def generate_wordcloud(self, keywords: List[str], weights: List[float] = None) -> str:
        """生成词云图并转换为base64编码"""
        if weights is None:
            weights = [1] * len(keywords)
        
        word_freq = dict(zip(keywords, weights))
        
        try:
            # 配置词云图参数
            wc_params = {
                'width': 800,
                'height': 400,
                'background_color': 'white',
                'prefer_horizontal': 0.7,
                'relative_scaling': 0.5,
                'min_font_size': 10,
                'max_font_size': 100,
                'random_state': 42
            }
            
            # 如果找到了可用字体，则使用它
            if self.font_path:
                wc_params['font_path'] = self.font_path
            
            wc = WordCloud(**wc_params)
            
            # 确保有足够的词频数据
            if not word_freq:
                word_freq = {'示例': 100, '词云': 80, '测试': 60}
            
            wc.generate_from_frequencies(word_freq)
            
            # 将词云图转换为base64编码
            img = io.BytesIO()
            wc.to_image().save(img, format='PNG')
            return 'data:image/png;base64,{}'.format(
                base64.b64encode(img.getvalue()).decode()
            )
            
        except Exception as e:
            logger.error(f"生成词云图时出错: {str(e)}")
            # 返回一个简单的错误提示图像
            return ''
    
    def setup_layout(self):
        """设置仪表盘布局"""
        self.app.layout = html.Div([
            # 标题
            html.H1('微博舆情分析仪表盘', style={'textAlign': 'center'}),
            
            # 时间范围选择器
            html.Div([
                html.Label('选择时间范围：'),
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=self.df['created_at'].min(),
                    end_date=self.df['created_at'].max(),
                    display_format='YYYY-MM-DD'
                )
            ], style={'margin': '20px'}),
            
            # 主要指标卡片
            html.Div([
                html.Div([
                    html.H4('总微博数'),
                    html.H2(id='total-posts')
                ], className='metric-card'),
                html.Div([
                    html.H4('总互动量'),
                    html.H2(id='total-interactions')
                ], className='metric-card'),
                html.Div([
                    html.H4('平均情感得分'),
                    html.H2(id='avg-sentiment')
                ], className='metric-card')
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
            
            # 图表区域
            html.Div([
                # 左侧：演化趋势图
                html.Div([
                    html.H3('舆情演化趋势'),
                    dcc.Graph(id='evolution-trend')
                ], style={'width': '60%'}),
                
                # 右侧：词云图
                html.Div([
                    html.H3('热点词云'),
                    html.Img(id='wordcloud-img', style={'width': '100%'})
                ], style={'width': '40%'})
            ], style={'display': 'flex', 'margin': '20px'}),
            
            # 底部：主题分布和情感分析
            html.Div([
                # 左侧：主题分布
                html.Div([
                    html.H3('主题分布'),
                    dcc.Graph(id='topic-distribution')
                ], style={'width': '50%'}),
                
                # 右侧：情感分析
                html.Div([
                    html.H3('情感分析'),
                    dcc.Graph(id='sentiment-analysis')
                ], style={'width': '50%'})
            ], style={'display': 'flex', 'margin': '20px'})
        ])
    
    def setup_callbacks(self):
        """设置回调函数"""
        @self.app.callback(
            [Output('total-posts', 'children'),
             Output('total-interactions', 'children'),
             Output('avg-sentiment', 'children'),
             Output('evolution-trend', 'figure'),
             Output('wordcloud-img', 'src'),
             Output('topic-distribution', 'figure'),
             Output('sentiment-analysis', 'figure')],
            [Input('date-picker', 'start_date'),
             Input('date-picker', 'end_date')]
        )
        def update_dashboard(start_date, end_date):
            # 过滤数据
            mask = (self.df['created_at'] >= start_date) & (self.df['created_at'] <= end_date)
            filtered_df = self.df[mask]
            
            # 计算主要指标
            total_posts = len(filtered_df)
            total_interactions = filtered_df['reposts_count'].sum() + \
                               filtered_df['comments_count'].sum() + \
                               filtered_df['attitudes_count'].sum()
            avg_sentiment = filtered_df['sentiment'].mean()
            
            # 生成演化趋势图
            evolution_fig = make_subplots(rows=2, cols=1,
                                        subplot_titles=('参与度趋势', '对抗指数趋势'))
            evolution_fig.add_trace(
                go.Scatter(x=self.evolution_data['date'],
                          y=self.evolution_data['participation'],
                          mode='lines+markers',
                          name='参与度'),
                row=1, col=1
            )
            evolution_fig.add_trace(
                go.Scatter(x=self.evolution_data['date'],
                          y=self.evolution_data['confrontation_index'],
                          mode='lines+markers',
                          name='对抗指数'),
                row=2, col=1
            )
            
            # 生成词云图
            wordcloud_src = self.generate_wordcloud(
                list(self.keyword_data.keys()),
                list(self.keyword_data.values())
            )
            
            # 生成主题分布图
            topic_fig = px.pie(
                values=list(self.topic_data.values()),
                names=list(self.topic_data.keys()),
                title='主题分布'
            )
            
            # 生成情感分析图
            sentiment_data = filtered_df.groupby('sentiment').size()
            sentiment_fig = px.bar(
                x=['负面', '中性', '正面'],
                y=sentiment_data.values,
                title='情感分布'
            )
            
            return (
                f"{total_posts:,}",
                f"{total_interactions:,}",
                f"{avg_sentiment:.2f}",
                evolution_fig,
                wordcloud_src,
                topic_fig,
                sentiment_fig
            )
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """运行仪表盘服务器"""
        self.app.run_server(debug=debug, port=port)

def main():
    # 示例数据
    df = pd.DataFrame({
        'created_at': pd.date_range(start='2022-02-27', end='2022-03-03', freq='H'),
        'reposts_count': np.random.randint(0, 1000, 120),
        'comments_count': np.random.randint(0, 500, 120),
        'attitudes_count': np.random.randint(0, 2000, 120),
        'sentiment': np.random.choice([0, 1, 2], 120),
        'topic': np.random.choice(['经济', '政治', '军事', '外交'], 120)
    })
    
    # 示例主题数据
    topic_data = {
        '经济': 30,
        '政治': 25,
        '军事': 20,
        '外交': 25
    }
    
    # 示例演化数据
    evolution_data = pd.DataFrame({
        'date': pd.date_range(start='2022-02-27', end='2022-03-03'),
        'participation': np.random.random(5),
        'confrontation_index': np.random.random(5)
    })
    
    # 示例关键词数据
    keyword_data = {
        '俄罗斯': 100,
        '乌克兰': 90,
        '制裁': 80,
        '战争': 70,
        'SWIFT': 60,
        '谈判': 50,
        '和平': 40,
        '军事': 30,
        '经济': 20,
        '外交': 10
    }
    
    # 创建并运行仪表盘
    dashboard = Dashboard(df, topic_data, evolution_data, keyword_data)
    dashboard.run_server()

if __name__ == "__main__":

    main() 