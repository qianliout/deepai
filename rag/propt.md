我准备使用langchain和rag技术实现个人知识库的功能
    要求： 
    - 在deepai/rag目录下实现
    - 我的电脑是mac M1,使用gpu训练
    - 我需要通过这个项目学习整个 rag和langchain的相关 知识，所以这个项目的知识点应该尽量全面
    - 我使用通义百炼大模型平台提供的大模型底座
    - 我是一个新后，所以需要代码结构清晰，日志完善，注释完善
    - 代码中涉及数据结构或超参数的，最好使用pydantic库用来指明数据结构,如果是 tensor 格式，要写好 shape,
    - 要在注释中重点关注数据的流转
    - 提供的命令应该尽量简单：如： python main.py quick 快速测试
    - 如果需要用到数据库等其他工具，需要告诉我怎么使用 docker进行安装
    - 所有的配置 config.py 中统一配置管理

你的实现太全面，需要简化   
    - 只提供交互式问答的功能，不需要提供api的功能
    - 暂时只支持txt文本类型的知识
    - 只使用ChromaDB向量数据库，底层使用SQLite进行存储
    - 支持流式对话，且需要支持会话上下文保持功能
    - 知识文件存放本地，不考虑MinIO存储
    - 对话数据保存在redis,暂时不需要es,mysql等存储
    - 暂时不需要性能监控等额外功能
    - 我主要是想学习rag和大模型的知识，所以你的重点应放到rag和llm上
    - 不考虑兼容之前代码，删除无用的代码
    - 整理代码，使用结构更清晰，日志更完善
  
rag代码中还是有足的地方
    - 我的知识文件是中文，应该添加中文分词的相关功能，在做中文分词时提供两种分词方式，1：简单的手工分词，2：使用jieba分词器
    - 代码应该做前置检查，最好提供一个可以快速检查的命令：python main.py check
    在检查时不仅要检查配置是否存在，还应该检查配置是否正确，比如要检查配置的api_key是否能调通模型，提供的redis是否能连接通等等
    所以：项目中最好不要出现这样的代码,
        if not DASHSCOPE_AVAILABLE:
        if not self.api_key:
    - 查询扩展： _expand_query(self, query: str) 添加常同义词synonyms，不能手工维护
      - 可以使用：https://github.com/chatopera/Synonyms 这个库
    - 如果返回值是Dict[str, Any]类型的，请你定义成数据类的class
    - 上述所有的更改都不需要考虑对现有代码兼容



对于rag项目继续帮助我改代码
    - 我的python虚拟环境是deepai2,可以执行：conda activate aideep2
    - 去除使用Synonyms进行query expander的功能， 改为手动实现的一个简单同义词替换功能
    - ChineseTokenizer 应该拆分成两个类:SampleTokenizer和JiebaTokenizer
    - 你现在是按一个完整的rag的项目来实现，但是我现在是想通过这个项目还学习rag的相关知识，你实现的很多功能太过于复杂，可以尽量简化
    - 我想通过这个项目学习到的知识点有：
      - rag的全部功能及在业界的最佳实践
      - 分词:SampleTokenizer和JiebaTokenizer
      - 向量存储（使用ChromaDB向量数据库，底层使用SQLite进行存储）
      - 混合检索(结合关键词检索（BM25）+ 向量检索)
      - 会话上下文保持(用户的query及回答等信息应该存在redis中)
      - 流式对话
      - 检索优化：多路召回（Hybrid Search)动态调整 Top-K,设定相似度阈值,重排序（Rerank）,Query扩展 等等优化方法。
    - 当前项目的基础上改正，且不需要兼容现有代码  


对于rag项目继续帮助我改代码
    - 日志系统太过复杂,请简化，参考bert里的日志方式就好
    - 提供子命令的方式太过复杂，参考bert/main.py
    - 全局配置时进行拆分，参考bert/config.py，定义成
    - 所有的修改不考虑兼容性，删除无用的代码
    

对于rag项目继续帮助我改代码,
    1：日志存放到/Users/liuqianli/work/python/deepai/logs/rag/目录下
    2：运行这些命令时还是有报错，请你验证每一个命令，如有错，请改正

对于rag项目继续帮助我改代码,
    - 合并simple_rag_chain.py到rag_chain.py中
    - 合并simple_retriever.py到retriever.py中
    - 合并test_imports.py，test_split.py和system_checker.py到一个新的check.py文件中
    - 不仅仅是文件合并，还得保证程序正确 
    - 所有的修改不考虑兼容性，删除无用的代码