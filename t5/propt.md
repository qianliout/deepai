请你使用python的pytorch库，实现t5框架：
    要求： 
    - 在deepai/t5目录下实现
    - 我的电脑是mac M1,使用gpu训练
    - 我需要通过这个项目学习整个t5知识,但是比如embedding,tokenizer等这些知识我已学会，所以你可以直接使用pytorch提供的embedding,tokenizer等工具
    - 我是一个新后，所以需要代码结构清晰，日志完善，注释完善
    - 代码中涉及数据结构或超参数的，最好使用pydantic库用来指明数据结构,如果是 tensor 格式，要写好 shape,
    - 要在注释中重点关注数据的流转
    - 提供的命令应该尽量简单：如： python main.py quick 快速测试,运行
    - 所有的配置 config.py 中统一配置管理
    - 日志存放到/Users/liuqianli/work/python/deepai/logs/t5/目录下
    - 预训练模型存放到：/Users/liuqianli/work/python/deepai/saved_model/t5/
    - 下载的数据集及分词器等使用默认的cache_dir,在我的电脑中是 /User/liuqianli/.cache/huggingface/datasets/
    - 加载数据集或分词器时，默认直接读本地，不去hugginface上下载
    - 对于项目中可能会用到的目录都需要在config.py中显式指定，而不是在散落在代码中去拼接，当然这些目录可以在使用的时候才去创建
    - 所有这些目录，不需要做为参数传递,在使用时直接从全局配置中取

再修改代码
    - model.py里的代码太复杂，需要做如下优化：
      - T5Block里最好做成encoder和decoder
      - 每个函数的入参和出参不能太多，太多了不容易理解


运行失败，还需要调试