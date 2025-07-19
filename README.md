# 部署到 Streamlit Cloud
### 1.打开网站
链接：[Streamlit Community Cloud](https://share.streamlit.io/)，单击工作区中Create app，然后指定存储库、分支和主文件路径


### 2.设置
选择python版本3.10
Secrets中填写自己的智谱API_KEY：ZHIPUAI_API_KEY = "xxxxxx"

### 3.完成
点击 Delop 按钮，加载依赖
完成后应用程序部署到 Streamlit Community Cloud，并且可以从世界各地访问！

**注意：**
系统里的 sqlite3 太老，需要你手动指定使用新版本，因此推荐：

在 requirements.txt 中加上：
```
pysqlite3-binary
chromadb
```

然后在代码最前面手动 patch：

```python
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
```
⚠️ 注意：这段 patch 一定要写在 任何导入 chromadb 之前，否则不会生效。
