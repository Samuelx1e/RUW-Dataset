import os
import shutil
from init_data import create_stopwords

def init_project():
    """初始化项目结构和必要文件"""
    
    # 创建目录结构
    directories = [
        'data',
        'models/bert-base-chinese',
        'output',
        'src/preprocessing',
        'src/sentiment',
        'src/topic',
        'src/keywords',
        'src/evolution',
        'src/visualization'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # 创建停用词表
    create_stopwords()
    
    print("项目初始化完成！")
    print("已创建以下目录：")
    for directory in directories:
        print(f"- {directory}")
    print("\n请确保：")
    print("1. 数据文件已放入 data 目录")
    print("2. 首次运行时会自动下载BERT模型（需要网络连接）")

if __name__ == "__main__":
    init_project() 