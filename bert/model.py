"""
BERT模型模块 - 完整的BERT实现
包含基础模型、预训练模型、分类模型
重点关注数据流转和shape变化，包含详细的shape注释
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging

from config import BERT_CONFIG
from transformer import TransformerEncoderLayer, LayerNorm

logger = logging.getLogger("BERT")


class BertEmbeddings(nn.Module):
    """
    BERT嵌入层：词嵌入 + 位置嵌入 + 类型嵌入

    数据流：
    input_ids: (batch_size, seq_len) -> word_embeddings: (batch_size, seq_len, hidden_size)
    position_ids: (batch_size, seq_len) -> position_embeddings: (batch_size, seq_len, hidden_size)
    token_type_ids: (batch_size, seq_len) -> token_type_embeddings: (batch_size, seq_len, hidden_size)
    -> 三者相加 -> LayerNorm -> Dropout -> (batch_size, seq_len, hidden_size)
    """

    def __init__(self):
        super().__init__()

        # 词嵌入：将token ID转换为向量
        self.word_embeddings = nn.Embedding(BERT_CONFIG.vocab_size, BERT_CONFIG.hidden_size, padding_idx=0)

        # 位置嵌入：编码token在序列中的位置
        self.position_embeddings = nn.Embedding(BERT_CONFIG.max_position_embeddings, BERT_CONFIG.hidden_size)

        # 类型嵌入：区分句子A和句子B（用于NSP任务）
        self.token_type_embeddings = nn.Embedding(BERT_CONFIG.type_vocab_size, BERT_CONFIG.hidden_size)

        # LayerNorm和Dropout
        self.LayerNorm = LayerNorm(BERT_CONFIG.hidden_size, eps=BERT_CONFIG.layer_norm_eps)
        self.dropout = nn.Dropout(BERT_CONFIG.hidden_dropout_prob)

        # 注册位置id缓冲区（不需要梯度的参数）
        self.register_buffer(
            "position_ids",
            torch.arange(BERT_CONFIG.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
            self,
            input_ids: torch.Tensor,  # (batch_size, seq_len)
            token_type_ids: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
            position_ids: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: 输入token ids (batch_size, seq_len)
            token_type_ids: token类型ids (batch_size, seq_len)，0表示句子A，1表示句子B
            position_ids: 位置ids (batch_size, seq_len)

        Returns:
            embeddings: 嵌入向量 (batch_size, seq_len, hidden_size)

        数据流转详解：
        1. input_ids [101, 2023, 2003, 102, 0, 0] -> word_embeddings (batch_size, seq_len, hidden_size)
        2. position_ids [0, 1, 2, 3, 4, 5] -> position_embeddings (batch_size, seq_len, hidden_size)
        3. token_type_ids [0, 0, 0, 0, 0, 0] -> token_type_embeddings (batch_size, seq_len, hidden_size)
        4. 三个嵌入相加 -> LayerNorm -> Dropout
        """
        batch_size, seq_len = input_ids.shape

        # 获取位置ids
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]  # (1, seq_len) -> 广播到 (batch_size, seq_len)

        # 获取token类型ids
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)  # (batch_size, seq_len)，全0表示单句

        # 计算各种嵌入
        word_embeddings = self.word_embeddings(input_ids)  # (batch_size, seq_len, hidden_size)
        position_embeddings = self.position_embeddings(position_ids)  # (batch_size, seq_len, hidden_size)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # (batch_size, seq_len, hidden_size)

        # 合并嵌入：三个向量逐元素相加
        embeddings = word_embeddings + position_embeddings + token_type_embeddings  # (batch_size, seq_len, hidden_size)

        # LayerNorm和Dropout
        embeddings = self.LayerNorm(embeddings)  # (batch_size, seq_len, hidden_size)
        embeddings = self.dropout(embeddings)  # (batch_size, seq_len, hidden_size)

        return embeddings


class BertEncoder(nn.Module):
    """
    BERT编码器 - 多层Transformer编码器堆叠

    数据流：
    input: (batch_size, seq_len, hidden_size)
    -> layer1: (batch_size, seq_len, hidden_size)
    -> layer2: (batch_size, seq_len, hidden_size)
    -> ...
    -> layerN: (batch_size, seq_len, hidden_size)
    """

    def __init__(self):
        super().__init__()

        # 创建多层Transformer编码器层
        self.layer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=BERT_CONFIG.hidden_size,
                    n_heads=BERT_CONFIG.num_attention_heads,
                    d_ff=BERT_CONFIG.intermediate_size,
                    dropout=BERT_CONFIG.hidden_dropout_prob,
                    activation=BERT_CONFIG.hidden_act,
                )
                for _ in range(BERT_CONFIG.num_hidden_layers)
            ]
        )

    def forward(
            self,
            hidden_states: torch.Tensor,  # (batch_size, seq_len, hidden_size)
            attention_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
            output_attentions: bool = False,
            output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: 输入隐藏状态 (batch_size, seq_len, hidden_size)
            attention_mask: 注意力掩码 (batch_size, seq_len)，1表示真实token，0表示padding
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出所有隐藏状态

        Returns:
            包含输出的字典：
            - last_hidden_state: 最后一层输出 (batch_size, seq_len, hidden_size)
            - hidden_states: 所有层输出 tuple of (batch_size, seq_len, hidden_size)
            - attentions: 所有层注意力权重 tuple of (batch_size, n_heads, seq_len, seq_len)

        数据流转详解：
        每一层都会对输入进行以下变换：
        1. 自注意力：学习token之间的关系
        2. 残差连接 + LayerNorm：稳定训练
        3. 前馈网络：非线性变换
        4. 残差连接 + LayerNorm：稳定训练
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 逐层处理
        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 通过当前层
            layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)
            # BERT 的核心是由多层 Transformer 编码器堆叠而成。每一层都会对输入的 hidden states 进行进一步的特征提取和变换。
            hidden_states = layer_outputs[0]  # (batch_size, seq_len, hidden_size)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }


class BertPooler(nn.Module):
    """
    BERT池化层 - 将序列表示转换为句子表示

    数据流：
    sequence_output: (batch_size, seq_len, hidden_size)
    -> 取第一个token: (batch_size, hidden_size)
    -> Linear: (batch_size, hidden_size)
    -> Tanh: (batch_size, hidden_size)
    """

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(BERT_CONFIG.hidden_size, BERT_CONFIG.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            hidden_states: 编码器输出 (batch_size, seq_len, hidden_size)

        Returns:
            pooled_output: 池化输出 (batch_size, hidden_size)

        池化逻辑：
        取第一个token（[CLS]）的隐藏状态作为整个句子的表示
        这个表示通常用于分类任务
        """
        # 取第一个token（[CLS]）的隐藏状态
        first_token_tensor = hidden_states[:, 0]  # (batch_size, hidden_size)

        # 通过线性层和激活函数
        pooled_output = self.dense(first_token_tensor)  # (batch_size, hidden_size)
        pooled_output = self.activation(pooled_output)  # (batch_size, hidden_size)

        return pooled_output


class BertModel(nn.Module):
    """
    基础BERT模型 - 不包含任务特定的头

    数据流：
    input_ids: (batch_size, seq_len)
    -> embeddings: (batch_size, seq_len, hidden_size)
    -> encoder: (batch_size, seq_len, hidden_size)
    -> pooler: (batch_size, hidden_size)
    """

    def __init__(self, add_pooling_layer: bool = True):
        super().__init__()

        # 各个组件
        self.embeddings = BertEmbeddings()
        self.encoder = BertEncoder()
        self.pooler = BertPooler() if add_pooling_layer else None

        # 初始化权重
        self.apply(self._init_weights)

        logger.info(f"BERT模型初始化完成，参数数量: {self.count_parameters():,}")

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=BERT_CONFIG.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=BERT_CONFIG.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def count_parameters(self) -> int:
        """计算参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.embeddings.word_embeddings = value

    def _create_attention_mask_from_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        从输入ids创建注意力掩码

        Args:
            input_ids: 输入token ids (batch_size, seq_len)

        Returns:
            attention_mask: 注意力掩码 (batch_size, seq_len)

        掩码逻辑：
        - 真实token（非0）-> 1（可以注意）
        - padding token（0）-> 0（不能注意）
        """
        return (input_ids != 0).long()  # 非0位置为1，0位置为0

    def forward(
            self,
            input_ids: torch.Tensor,  # (batch_size, seq_len)
            attention_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
            token_type_ids: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
            position_ids: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入token ids (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            token_type_ids: token类型ids (batch_size, seq_len)
            position_ids: 位置ids (batch_size, seq_len)
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出所有隐藏状态

        Returns:
            包含模型输出的字典：
            - last_hidden_state: 最后一层输出 (batch_size, seq_len, hidden_size)
            - pooler_output: 池化输出 (batch_size, hidden_size)
            - hidden_states: 所有层输出（如果requested）
            - attentions: 所有层注意力权重（如果requested）

        完整数据流转：
        1. input_ids -> embeddings: token转换为向量表示
        2. embeddings -> encoder: 多层Transformer处理
        3. encoder_output -> pooler: 生成句子级表示
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        # 创建注意力掩码（如果没有提供）
        if attention_mask is None:
            attention_mask = self._create_attention_mask_from_input_ids(input_ids)

        # 嵌入层：将token转换为向量表示
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )  # (batch_size, seq_len, hidden_size)

        # 编码器：多层Transformer处理
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs["last_hidden_state"]  # (batch_size, seq_len, hidden_size)

        # 池化层：生成句子级表示
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None  # (batch_size, hidden_size)

        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
            "hidden_states": encoder_outputs.get("hidden_states"),
            "attentions": encoder_outputs.get("attentions"),
        }


# ===== 任务特定的头部模块 =====


class BertLMPredictionHead(nn.Module):
    """
    BERT语言模型预测头 - 用于MLM任务

    数据流：
    hidden_states: (batch_size, seq_len, hidden_size)
    -> transform: (batch_size, seq_len, hidden_size)
    -> activation: (batch_size, seq_len, hidden_size)
    -> layer_norm: (batch_size, seq_len, hidden_size)
    -> decoder: (batch_size, seq_len, vocab_size)
    """

    def __init__(self):
        super().__init__()

        # 变换层
        self.transform = nn.Linear(BERT_CONFIG.hidden_size, BERT_CONFIG.hidden_size)
        self.activation = F.gelu
        self.layer_norm = LayerNorm(BERT_CONFIG.hidden_size, eps=BERT_CONFIG.layer_norm_eps)

        # 输出层：投影到词汇表大小
        self.decoder = nn.Linear(BERT_CONFIG.hidden_size, BERT_CONFIG.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(BERT_CONFIG.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            hidden_states: 编码器输出 (batch_size, seq_len, hidden_size)

        Returns:
            prediction_scores: 预测分数 (batch_size, seq_len, vocab_size)

        数据流转：
        1. 线性变换：增强表示能力
        2. GELU激活：非线性变换
        3. LayerNorm：归一化
        4. 投影到词汇表：得到每个位置对所有词的预测分数
        """
        hidden_states = self.transform(hidden_states)  # (batch_size, seq_len, hidden_size)
        hidden_states = self.activation(hidden_states)  # (batch_size, seq_len, hidden_size)
        hidden_states = self.layer_norm(hidden_states)  # (batch_size, seq_len, hidden_size)
        output = self.decoder(hidden_states)  # (batch_size, seq_len, vocab_size)
        return output


class BertNSPHead(nn.Module):
    """
    BERT下一句预测头 - 用于NSP任务

    数据流：
    pooled_output: (batch_size, hidden_size)
    -> linear: (batch_size, 2)
    """

    def __init__(self):
        super().__init__()
        self.seq_relationship = nn.Linear(BERT_CONFIG.hidden_size, 2)  # 二分类：IsNext/NotNext

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            pooled_output: 池化输出 (batch_size, hidden_size)

        Returns:
            seq_relationship_scores: NSP分数 (batch_size, 2)

        NSP任务：
        判断两个句子是否是连续的
        0: NotNext（不连续）
        1: IsNext（连续）
        """
        return self.seq_relationship(pooled_output)  # (batch_size, 2)


class BertClassificationHead(nn.Module):
    """
    BERT分类头 - 用于序列分类任务

    数据流：
    pooled_output: (batch_size, hidden_size)
    -> dropout: (batch_size, hidden_size)
    -> classifier: (batch_size, num_labels)
    """

    def __init__(self, num_labels: int):
        super().__init__()
        self.dropout = nn.Dropout(BERT_CONFIG.hidden_dropout_prob)
        self.classifier = nn.Linear(BERT_CONFIG.hidden_size, num_labels)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            pooled_output: 池化输出 (batch_size, hidden_size)

        Returns:
            logits: 分类logits (batch_size, num_labels)
        """
        pooled_output = self.dropout(pooled_output)  # (batch_size, hidden_size)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)
        return logits


# ===== 完整的BERT模型 =====


class BertForPreTraining(nn.Module):
    """
    用于预训练的BERT - 包含MLM和NSP两个任务

    数据流：
    input -> BertModel -> MLM头 + NSP头 -> 预测分数
    """

    def __init__(self):
        super().__init__()

        self.bert = BertModel()
        self.cls = BertLMPredictionHead()
        self.nsp = BertNSPHead()

        # TODO 这里权重共享是什么意思
        # 权重共享：MLM头的输出权重与输入嵌入权重共享
        self.cls.decoder.weight = self.bert.embeddings.word_embeddings.weight

        # 初始化权重
        self.apply(self._init_weights)

        logger.info(f"BertForPreTraining初始化完成，参数数量: {self.count_parameters():,}")

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=BERT_CONFIG.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=BERT_CONFIG.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def count_parameters(self) -> int:
        """计算参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            input_ids: torch.Tensor,  # (batch_size, seq_len)
            attention_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
            token_type_ids: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
            position_ids: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
            labels: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
            next_sentence_label: Optional[torch.Tensor] = None,  # (batch_size,)
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入token ids (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            token_type_ids: token类型ids (batch_size, seq_len)
            position_ids: 位置ids (batch_size, seq_len)
            labels: MLM标签 (batch_size, seq_len)，-100表示不计算损失的位置
            next_sentence_label: NSP标签 (batch_size,)，0表示NotNext，1表示IsNext
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出所有隐藏状态

        Returns:
            包含损失和预测的字典：
            - loss: 总损失（MLM损失 + NSP损失）
            - prediction_logits: MLM预测logits (batch_size, seq_len, vocab_size)
            - seq_relationship_logits: NSP预测logits (batch_size, 2)
            - hidden_states: 所有层输出（如果requested）
            - attentions: 所有层注意力权重（如果requested）

        预训练任务流程：
        1. 输入包含[MASK] token和句子对的序列
        2. BERT编码器处理序列
        3. MLM头预测[MASK]位置的词
        4. NSP头预测两个句子是否连续
        5. 计算两个任务的联合损失
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs["last_hidden_state"]  # (batch_size, seq_len, hidden_size)
        pooled_output = outputs["pooler_output"]  # (batch_size, hidden_size)

        # MLM预测
        prediction_scores = self.cls(sequence_output)  # (batch_size, seq_len, vocab_size)

        # NSP预测
        seq_relationship_scores = self.nsp(pooled_output)  # (batch_size, 2)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            # MLM损失
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, BERT_CONFIG.vocab_size),  # (batch_size * seq_len, vocab_size)
                labels.view(-1),  # (batch_size * seq_len,)
            )

            # NSP损失
            next_sentence_loss = loss_fct(
                seq_relationship_scores.view(-1, 2),  # (batch_size, 2)
                next_sentence_label.view(-1),  # (batch_size,)
            )

            # 总损失
            total_loss = masked_lm_loss + next_sentence_loss

        return {
            "loss": total_loss,
            "prediction_logits": prediction_scores,
            "seq_relationship_logits": seq_relationship_scores,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.bert.embeddings.word_embeddings = value
        self.cls.decoder.weight = value


class BertForSequenceClassification(nn.Module):
    """
    用于序列分类的BERT - 用于微调任务

    数据流：
    input -> BertModel -> 分类头 -> 分类logits
    """

    def __init__(self, num_labels: int = 2):
        super().__init__()
        self.num_labels = num_labels

        self.bert = BertModel()
        self.classifier = BertClassificationHead(num_labels)

        # 初始化权重
        self.apply(self._init_weights)

        logger.info(f"BertForSequenceClassification初始化完成，参数数量: {self.count_parameters():,}")

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=BERT_CONFIG.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=BERT_CONFIG.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def count_parameters(self) -> int:
        """计算参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            input_ids: torch.Tensor,  # (batch_size, seq_len)
            attention_mask: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
            token_type_ids: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
            position_ids: Optional[torch.Tensor] = None,  # (batch_size, seq_len)
            labels: Optional[torch.Tensor] = None,  # (batch_size,)
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入token ids (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            token_type_ids: token类型ids (batch_size, seq_len)
            position_ids: 位置ids (batch_size, seq_len)
            labels: 分类标签 (batch_size,)
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出所有隐藏状态

        Returns:
            包含损失和预测的字典：
            - loss: 分类损失（如果提供了labels）
            - logits: 分类logits (batch_size, num_labels)
            - hidden_states: 所有层输出（如果requested）
            - attentions: 所有层注意力权重（如果requested）

        分类任务流程：
        1. 输入文本序列
        2. BERT编码器处理序列
        3. 取[CLS] token的表示
        4. 分类头预测类别
        5. 计算分类损失
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs["pooler_output"]  # (batch_size, hidden_size)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # 回归任务
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # 分类任务
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.bert.embeddings.word_embeddings = value


# 导出所有模型类
__all__ = ["BertModel", "BertForPreTraining", "BertForSequenceClassification"]
