"""
T5推理模块
负责模型推理、文本生成和结果解析
重点关注推理过程中的数据流转和生成策略
"""

import torch
import torch.nn.functional as F
from transformers import T5Tokenizer
import os
import json
from typing import Dict, List, Optional, Tuple, Union
import logging
from pydantic import BaseModel

from config import T5_CONFIG, DATA_CONFIG, TRAINING_CONFIG, get_device
from model import T5ForConditionalGeneration

logger = logging.getLogger("T5")


class GenerationConfig(BaseModel):
    """生成配置的数据结构定义"""
    
    max_length: int = 128  # 最大生成长度
    min_length: int = 1    # 最小生成长度
    num_beams: int = 4     # beam search的beam数量
    do_sample: bool = False  # 是否使用采样
    temperature: float = 1.0  # 采样温度
    top_k: int = 50        # top-k采样
    top_p: float = 1.0     # top-p采样
    repetition_penalty: float = 1.0  # 重复惩罚
    length_penalty: float = 1.0      # 长度惩罚
    early_stopping: bool = True      # 是否早停
    
    class Config:
        extra = "forbid"


class T5Inference:
    """
    T5推理器
    
    负责：
    1. 模型加载
    2. 文本预处理
    3. 推理生成
    4. 结果后处理
    """
    
    def __init__(self, model_path: str):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径
        """
        logger.info("初始化T5推理器...")
        
        # 设备设置
        self.device = get_device()
        logger.info(f"使用设备: {self.device}")
        
        # 加载tokenizer
        logger.info(f"加载tokenizer: {DATA_CONFIG.tokenizer_name}")
        try:
            # 首先尝试使用本地文件
            self.tokenizer = T5Tokenizer.from_pretrained(
                DATA_CONFIG.tokenizer_name,
                cache_dir=TRAINING_CONFIG.cache_dir,
                local_files_only=True
            )
        except Exception as e:
            logger.info(f"本地tokenizer不存在，从网络下载: {e}")
            # 如果本地文件不存在，从网络下载
            self.tokenizer = T5Tokenizer.from_pretrained(
                DATA_CONFIG.tokenizer_name,
                cache_dir=TRAINING_CONFIG.cache_dir,
                local_files_only=False
            )
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 默认生成配置
        self.generation_config = GenerationConfig()
        
        logger.info("T5推理器初始化完成")
    
    def _load_model(self, model_path: str) -> T5ForConditionalGeneration:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            model: 加载的模型
        """
        logger.info(f"从 {model_path} 加载模型...")
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 初始化模型
        model = T5ForConditionalGeneration()
        
        # 加载权重
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info("模型权重加载成功")
        else:
            logger.warning(f"未找到模型权重文件: {model_file}")
        
        # 加载配置（如果存在）
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info("模型配置加载成功")
        
        return model
    
    def preprocess_input(self, input_text: str, task_prefix: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        预处理输入文本
        
        Args:
            input_text: 输入文本
            task_prefix: 任务前缀
            
        Returns:
            inputs: 预处理后的输入
            
        数据流转：
        原始文本 -> 添加前缀 -> tokenize -> tensor (1, seq_len)
        """
        # 添加任务前缀
        if task_prefix is None:
            task_prefix = DATA_CONFIG.task_prefix
        
        full_input = task_prefix + input_text
        
        # Tokenize
        inputs = self.tokenizer(
            full_input,
            max_length=DATA_CONFIG.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def postprocess_output(self, output_ids: torch.Tensor) -> List[str]:
        """
        后处理输出
        
        Args:
            output_ids: 输出token ids (batch_size, seq_len)
            
        Returns:
            texts: 解码后的文本列表
            
        数据流转：
        tensor (batch_size, seq_len) -> List[str]
        """
        # 解码
        texts = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return texts
    
    @torch.no_grad()
    def generate(
        self,
        input_text: str,
        task_prefix: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None
    ) -> str:
        """
        生成文本
        
        Args:
            input_text: 输入文本
            task_prefix: 任务前缀
            generation_config: 生成配置
            
        Returns:
            generated_text: 生成的文本
        """
        # 使用默认配置
        if generation_config is None:
            generation_config = self.generation_config
        
        # 预处理输入
        inputs = self.preprocess_input(input_text, task_prefix)
        
        # 生成
        if generation_config.do_sample:
            # 采样生成
            output_ids = self._generate_with_sampling(inputs, generation_config)
        else:
            # Beam search生成
            output_ids = self._generate_with_beam_search(inputs, generation_config)
        
        # 后处理输出
        generated_texts = self.postprocess_output(output_ids)
        
        return generated_texts[0] if generated_texts else ""
    
    def _generate_with_beam_search(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        """
        使用beam search生成
        
        Args:
            inputs: 输入数据
            generation_config: 生成配置
            
        Returns:
            output_ids: 生成的token ids
        """
        batch_size = inputs["input_ids"].shape[0]
        num_beams = generation_config.num_beams
        
        # 编码器前向传播
        encoder_hidden_states = self.model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )  # (batch_size, encoder_seq_len, d_model)

        # 初始化beam search
        beam_scores = torch.zeros((batch_size, num_beams), device=self.device)
        beam_scores[:, 1:] = -1e9  # 只有第一个beam有效
        beam_scores = beam_scores.view(-1)  # (batch_size * num_beams,)

        # 扩展编码器输出
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(
            batch_size, num_beams, -1, -1
        ).contiguous().view(batch_size * num_beams, -1, encoder_hidden_states.shape[-1])
        
        encoder_attention_mask = inputs["attention_mask"].unsqueeze(1).expand(
            batch_size, num_beams, -1
        ).contiguous().view(batch_size * num_beams, -1)
        
        # 初始化解码器输入
        decoder_input_ids = torch.full(
            (batch_size * num_beams, 1),
            T5_CONFIG.decoder_start_token_id,
            device=self.device,
            dtype=torch.long
        )
        
        # 生成循环
        for step in range(generation_config.max_length):
            # 解码器前向传播
            decoder_hidden_states = self.model.decoder(
                decoder_input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )  # (batch_size * num_beams, decoder_seq_len, d_model)

            # 获取下一个token的logits
            next_token_logits = self.model.lm_head(decoder_hidden_states[:, -1, :])
            
            # 应用长度惩罚
            if generation_config.length_penalty != 1.0:
                length_penalty = ((5 + step + 1) / 6) ** generation_config.length_penalty
                next_token_logits = next_token_logits / length_penalty
            
            # 计算分数
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores + beam_scores[:, None]
            
            # 重塑为 (batch_size, num_beams * vocab_size)
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # 选择top-k
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            # 更新beam
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # 重新排列beam
            beam_outputs = []
            for batch_idx in range(batch_size):
                beam_outputs.append([])
                for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(next_tokens[batch_idx], next_token_scores[batch_idx], next_indices[batch_idx])
                ):
                    batch_beam_idx = batch_idx * num_beams + next_index
                    beam_outputs[-1].append((next_score, next_token, batch_beam_idx))
            
            # 选择最佳beam
            beam_scores = []
            beam_tokens = []
            beam_idx = []
            
            for batch_idx in range(batch_size):
                beam_outputs[batch_idx] = sorted(beam_outputs[batch_idx], key=lambda x: x[0], reverse=True)
                for i in range(num_beams):
                    score, token, idx = beam_outputs[batch_idx][i]
                    beam_scores.append(score)
                    beam_tokens.append(token)
                    beam_idx.append(idx)
            
            beam_scores = torch.tensor(beam_scores, device=self.device)
            beam_tokens = torch.tensor(beam_tokens, device=self.device)
            beam_idx = torch.tensor(beam_idx, device=self.device)
            
            # 更新decoder_input_ids
            decoder_input_ids = decoder_input_ids[beam_idx]
            decoder_input_ids = torch.cat([decoder_input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            
            # 检查是否结束
            if torch.all(beam_tokens == T5_CONFIG.eos_token_id):
                break
        
        # 返回最佳序列
        return decoder_input_ids[:batch_size]
    
    def _generate_with_sampling(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_config: GenerationConfig
    ) -> torch.Tensor:
        """
        使用采样生成
        
        Args:
            inputs: 输入数据
            generation_config: 生成配置
            
        Returns:
            output_ids: 生成的token ids
        """
        batch_size = inputs["input_ids"].shape[0]
        
        # 编码器前向传播
        encoder_hidden_states = self.model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )  # (batch_size, encoder_seq_len, d_model)
        
        # 初始化解码器输入
        decoder_input_ids = torch.full(
            (batch_size, 1),
            T5_CONFIG.decoder_start_token_id,
            device=self.device,
            dtype=torch.long
        )
        
        # 生成循环
        for step in range(generation_config.max_length):
            # 解码器前向传播
            decoder_hidden_states = self.model.decoder(
                decoder_input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=inputs["attention_mask"],
            )  # (batch_size, decoder_seq_len, d_model)

            # 获取下一个token的logits
            next_token_logits = self.model.lm_head(decoder_hidden_states[:, -1, :])
            
            # 应用温度
            if generation_config.temperature != 1.0:
                next_token_logits = next_token_logits / generation_config.temperature
            
            # Top-k采样
            if generation_config.top_k > 0:
                top_k = min(generation_config.top_k, next_token_logits.size(-1))
                top_k_logits, _ = torch.topk(next_token_logits, top_k)
                min_top_k = top_k_logits[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.full_like(next_token_logits, float('-inf')),
                    next_token_logits
                )
            
            # Top-p采样
            if generation_config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
            
            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # 添加到序列
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens.unsqueeze(1)], dim=-1)
            
            # 检查是否结束
            if torch.all(next_tokens == T5_CONFIG.eos_token_id):
                break
        
        return decoder_input_ids
    
    def answer_question(self, question: str, context: str) -> str:
        """
        问答任务
        
        Args:
            question: 问题
            context: 上下文
            
        Returns:
            answer: 答案
        """
        input_text = f"question: {question} context: {context}"
        return self.generate(input_text, task_prefix="")
    
    def summarize_text(self, text: str) -> str:
        """
        文本摘要任务
        
        Args:
            text: 输入文本
            
        Returns:
            summary: 摘要
        """
        return self.generate(text, task_prefix="summarize: ")
    
    def translate_text(self, text: str, source_lang: str = "English", target_lang: str = "German") -> str:
        """
        翻译任务
        
        Args:
            text: 输入文本
            source_lang: 源语言
            target_lang: 目标语言
            
        Returns:
            translation: 翻译结果
        """
        task_prefix = f"translate {source_lang} to {target_lang}: "
        return self.generate(text, task_prefix=task_prefix)


if __name__ == "__main__":
    # 测试推理器
    from config import setup_logging
    
    # 设置日志
    setup_logging()
    
    # 模型路径（需要先训练模型）
    model_path = TRAINING_CONFIG.pretrain_best_dir
    
    if os.path.exists(model_path):
        # 创建推理器
        inference = T5Inference(model_path)
        
        # 测试生成
        test_input = "The weather is nice today."
        result = inference.generate(test_input)
        logger.info(f"输入: {test_input}")
        logger.info(f"输出: {result}")
    else:
        logger.warning(f"模型路径不存在: {model_path}")
        logger.info("请先训练模型")
    
    logger.info("推理器测试完成!")
