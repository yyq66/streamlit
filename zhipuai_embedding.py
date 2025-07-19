# 引入类型提示
from typing import List
from langchain_core.embeddings import Embeddings

# 自定义智谱AI Embedding类
class ZhipuAIEmbeddings(Embeddings):
    # 初始化函数
    def __init__(self):
        """
        导入并实例化智谱AI的Embedding模型。
        """
        from zhipuai import ZhipuAI
        self.client = ZhipuAI()

    # 重写对字符串列表计算embedding的embed_documents方法
    def embed_documents(self,texts: List[str]) -> List[List[float]]:
        """
        对字符串列表计算embedding。
        :param texts: 文本列表
        :return: 文本的嵌入向量列表
        """
        result = []
        for i in range(0,len(texts),64):
            batch = texts[i:i+64]
            embeddings = self.client.embeddings.create(
                model="embedding-3",
                input=batch
            )
            result.extend(embeddings.embedding for embeddings in embeddings.data)
        return result
    
    # 重写对单个文本str计算embedding的embed_query方法
    def embed_query(self, text:str) -> List[float]:
        """
        对单个文本计算embedding。
        :param text: 文本字符串
        :return: 文本的嵌入向量
        """
        return self.embed_documents([text])[0]