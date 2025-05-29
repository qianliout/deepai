我准备使用Python的torch库从0实现bert框架，请你给我实现
    要求： 
    1:不用参考现有的项目文件，按你的理解从0实现,且可以新建文件,在deepai/bert目录下实现
    1：我的电脑是mac M1,使用gpu训练
    2：可使用使用hugginface 上的Salesforce/wikitext数据集下做预训练
    4：可以使用hugginface下载数据集，且可以使用huggingface的transformer库。
    7：实现完整的bert框架，我需要使用这份代码串联学习
    8：保存训练好的模型
    9: 可以使用pytorch中提供的方法，如nn.Embedding,nn.LayerNorm等,但是需要有bert框架的完整实现，供我系统学习
    10: 代码结构清晰，日志完善，注释完善
    11: 代码中涉及数据结构或超参数的，最好使用pydantic库用来指明数据结构,
    13: 写好训练和测试代码的入口函数，让我可以一键运行


这份bert的代码，有几个地方我还是不满意
1：可以参考transformer的代码，
2：bert中用到了transformer的Encoder，这份代码没有体现
3：bert的训练中会有两个任务：masked language model和next sentence prediction，这份代码需要体现
4：加上bert的做微调代码，使用noob123/imdb_review_3000 这个数据集做微调,这个数据集会输出两类，negative和positive
5: 自注意力，AddNorm等这些代码属于transformer架构的代码，可以统一整理，这样可以使bert的建构更清晰，更容易学习


还有几个问题需要改
1：参数或中间变量是tensor类型的，在注释里加上详细shape
2：代码中有README.md和ARCHITECTURE.md和GETTING_STARTED.md三个文档，请合并并精简
4：我对各种mask的创建和后续操作很不明白，你需要着重简化这部分逻辑，并加上详细的注释
5：里面有run.py,run_bert.py,main.py,test_setup.py,请精简并合并，程序运行时不需要手动传参，这份代码我只是用于学习，如果要改参数，我可以直接改config.py里的数据
6：bert_data_loader.py和data_loader.py两个文件是否可以合并并精简
7：可以大致的区分为model,trainer,inference,data_loader,transformer,fine_tuning 这几个大模块，请重新整理代码。
3: 最后请重新整理代码，合并功能重复的代码,对于数据流转过程，加上详细的注释。

在 bert中实现了bert框架，但是有很多重复的代码及逻辑，请你在bert2中重新实现，要求
1：参考bert中的代码，可以大致的区分为model,trainer,inference,data_loader,transformer,fine_tuning 这几个大模块
2：有详细的注释和日志，参数或中间变量是tensor类型的，在注释里加上详细shape
3: 这份代码我只是用于学习，程序运行时不需要手动传参,如果要改参数，我可以直接改config.py里的数据


在 transformer中实现了transformer框架，但还是有重复的代码及逻辑，请你在transformer2中重新实现，要求
1：参考transformer中的代码,要写明数据流转的过程，以及流转过程中的shape变化
2：要有详细的注释和日志，参数或中间变量是tensor类型的，在注释里加上详细shape
3: 这份代码我只是用于学习，程序运行时不需要手动传参,如果要改参数，我可以直接改config.py里的数据，这一步可参考bert2中的实现
4：简化程序运行的命令，便于于debug


项目几个地方要改
对于bert,
    1：日志存放到/Users/liuqianli/work/python/deepai/logs/bert/目录下
    2：预训练模型存放到：/Users/liuqianli/work/python/deepai/saved_model/bert/
    3: 下载的数据集及分词器等使用默认的cache_dir,在我的电脑中是 /User/liuqianli/.cache/huggingface/datasets/
    4: 加载数据集或分词器时，默认直接读本地，不去hugginface上下载
    5: 对于项目中可能会用到的目录都需要在config.py中显式指定，而不是在散落在代码中去拼接，当然这些目录可以在使用的时候才去创建
    6: 所有这些目录，不需要做为参数传递,在使用时直接从全局配置中取
    7:output_dir 这个变量没有可读性，要改正


对于transformer,transformer2 是一样处理，只有存的目录不同,
    1：日志存放到/Users/liuqianli/work/python/deepai/logs/transformer/目录下
    2：预训练模型存放到：/Users/liuqianli/work/python/deepai/saved_model/transformer/
对于transformer2,
    1：日志存放到/Users/liuqianli/work/python/deepai/logs/transformer2/目录下
    2：预训练模型存放到：/Users/liuqianli/work/python/deepai/saved_model/transformer2/

