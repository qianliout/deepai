我准备使用Python的torch库从0实现transformer框架，请你给我实现
    要求： 
    1:不用参考现有的项目文件，按你的理解从0实现,且可以新建文件,在deepai/transformer/auge2目录下实现
    1：我的电脑是mac M1,使用gpu训练
    2：可使用使用hugginface 上的Helsinki-NLP/opus_books数据集下的"en-it"子集，
    3：训练时使用10000条数据训练，2000条验证
    4：可以使用hugginface下载数据集，但是不再使用hugginface的其他功能，包括：预训练模型，tokenizer，读取数据等等
    5: 自已实现分词功能，并把分词后的词典保存在本地
    6：使用hugginface的数据集，但是需要自己写代码读取数据，并处理成训练所需要的数据格式
    7：实现完整的transformer框架，包括：编码器，解码器，注意力机制，前馈网络，残差连接，层归一化，位置编码，多头注意力机制，自注意力机制，等等，我需要使用这份代码串联学习
    8：保存训练好的模型
    9:尽量使用pytorch中提供的方法，如nn.Embedding,nn.LayerNorm等，但是需要保留完整的transformer框架，我需要从头学习
    10: 代码结构清晰，日志完善，注释完善
    11: 代码中涉及数据结构或超参数的，最好使用pydantic库用来指明数据结构,
    13: 写好训练和测试代码的入口函数，让我可以一键运行
