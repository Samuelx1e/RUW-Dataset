import os

def create_stopwords():
    """创建基础的中文停用词表"""
    basic_stopwords = [
        '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
        '之', '于', '去', '也', '但', '又', '或', '如', '什么', '这',
        '那', '这个', '那个', '这些', '那些', '这样', '那样', '之类',
        '的话', '说', '对于', '等', '等等', '呢', '吧', '啊', '啦',
        # 添加俄乌相关的特殊停用词
        '转发', '微博', '视频', '链接', '网页', '全文', 'http', 'com',
        '转载', '来自', '观看', '评论', '回复', '点击', '详情', '...', 
        '…', '——', '...全文', '展开全文'
    ]
    
    # 创建data目录
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 写入停用词
    with open('data/stopwords.txt', 'w', encoding='utf-8') as f:
        for word in basic_stopwords:
            f.write(word + '\n')
    
    print("停用词表已创建：data/stopwords.txt")

if __name__ == "__main__":
    create_stopwords() 