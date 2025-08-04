"""
LLM客户端管理器
统一管理DeepSeek和Ollama模型调用
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from openai import AsyncOpenAI
import ollama

# 尝试相对导入，如果失败则使用绝对导入
try:
    from ..config.config import get_config, get_model_config
    from ..utils.logger import get_logger, log_performance, log_model_call
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from config.config import get_config, get_model_config
    from utils.logger import get_logger, log_performance, log_model_call

logger = get_logger("llm_client")

class LLMClient:
    """LLM客户端统一接口"""
    
    def __init__(self):
        self.config = get_config()
        self.model_config = get_model_config()
        
        # 初始化客户端
        self._init_clients()
    
    def _init_clients(self):
        """初始化各种LLM客户端"""
        llm_config = self.model_config["llm"]
        
        if llm_config["provider"] == "deepseek":
            # DeepSeek API客户端
            self.deepseek_client = AsyncOpenAI(
                api_key=self.config.models.get("deepseek_api_key", ""),
                base_url=llm_config["api_base"]
            )
            logger.info("DeepSeek客户端初始化成功")
        
        elif llm_config["provider"] == "ollama":
            # Ollama客户端
            self.ollama_client = ollama.AsyncClient(
                host=llm_config["api_base"].replace("/v1", "")
            )
            logger.info("Ollama客户端初始化成功")
    
    @log_performance()
    async def generate_response(self, messages: List[Dict[str, str]], 
                              temperature: float = None,
                              max_tokens: int = None,
                              stream: bool = False) -> str:
        """生成响应"""
        llm_config = self.model_config["llm"]
        
        # 使用配置中的默认值
        if temperature is None:
            temperature = llm_config.get("temperature", 0.1)
        if max_tokens is None:
            max_tokens = llm_config.get("max_tokens", 2048)
        
        start_time = time.time()
        
        try:
            if llm_config["provider"] == "deepseek":
                response = await self._call_deepseek(
                    messages, temperature, max_tokens, stream
                )
            elif llm_config["provider"] == "ollama":
                response = await self._call_ollama(
                    messages, temperature, max_tokens, stream
                )
            else:
                raise ValueError(f"不支持的LLM提供商: {llm_config['provider']}")
            
            # 记录模型调用日志
            end_time = time.time()
            duration = end_time - start_time
            
            input_tokens = sum(len(msg["content"]) // 4 for msg in messages)  # 粗略估算
            output_tokens = len(response) // 4 if isinstance(response, str) else 0
            
            log_model_call(
                model_name=llm_config["model_name"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration=duration
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            raise
    
    async def _call_deepseek(self, messages: List[Dict[str, str]], 
                           temperature: float, max_tokens: int, 
                           stream: bool) -> str:
        """调用DeepSeek API"""
        llm_config = self.model_config["llm"]
        
        try:
            response = await self.deepseek_client.chat.completions.create(
                model=llm_config["model_name"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                # 流式响应处理
                content = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                return content
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {str(e)}")
            raise
    
    async def _call_ollama(self, messages: List[Dict[str, str]], 
                         temperature: float, max_tokens: int, 
                         stream: bool) -> str:
        """调用Ollama API"""
        llm_config = self.model_config["llm"]
        
        try:
            # 转换消息格式
            ollama_messages = []
            for msg in messages:
                ollama_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            response = await self.ollama_client.chat(
                model=llm_config["model_name"],
                messages=ollama_messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                stream=stream
            )
            
            if stream:
                # 流式响应处理
                content = ""
                async for chunk in response:
                    if chunk.get("message", {}).get("content"):
                        content += chunk["message"]["content"]
                return content
            else:
                return response["message"]["content"]
                
        except Exception as e:
            logger.error(f"Ollama API调用失败: {str(e)}")
            raise
    
    async def generate_stream(self, messages: List[Dict[str, str]], 
                            temperature: float = None,
                            max_tokens: int = None) -> AsyncGenerator[str, None]:
        """流式生成响应"""
        llm_config = self.model_config["llm"]
        
        if temperature is None:
            temperature = llm_config.get("temperature", 0.1)
        if max_tokens is None:
            max_tokens = llm_config.get("max_tokens", 2048)
        
        try:
            if llm_config["provider"] == "deepseek":
                async for chunk in self._stream_deepseek(messages, temperature, max_tokens):
                    yield chunk
            elif llm_config["provider"] == "ollama":
                async for chunk in self._stream_ollama(messages, temperature, max_tokens):
                    yield chunk
            else:
                raise ValueError(f"不支持的LLM提供商: {llm_config['provider']}")
                
        except Exception as e:
            logger.error(f"流式LLM调用失败: {str(e)}")
            raise
    
    async def _stream_deepseek(self, messages: List[Dict[str, str]], 
                             temperature: float, max_tokens: int) -> AsyncGenerator[str, None]:
        """DeepSeek流式响应"""
        llm_config = self.model_config["llm"]
        
        response = await self.deepseek_client.chat.completions.create(
            model=llm_config["model_name"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def _stream_ollama(self, messages: List[Dict[str, str]], 
                           temperature: float, max_tokens: int) -> AsyncGenerator[str, None]:
        """Ollama流式响应"""
        llm_config = self.model_config["llm"]
        
        # 转换消息格式
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        response = await self.ollama_client.chat(
            model=llm_config["model_name"],
            messages=ollama_messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            stream=True
        )
        
        async for chunk in response:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]
    
    async def classify_query(self, query: str) -> Dict[str, Any]:
        """查询分类"""
        classification_prompt = f"""
请分析以下用户查询，并返回JSON格式的分类结果：

查询: {query}

请返回以下格式的JSON：
{{
    "type": "vulnerability_inquiry|image_vulnerability_check|host_security_status|remediation_guidance|security_statistics|general",
    "intent": "查询意图的简短描述",
    "entities": ["提取的实体列表"],
    "confidence": 0.0-1.0的置信度分数
}}
"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的AIOps查询分类器。"},
            {"role": "user", "content": classification_prompt}
        ]
        
        response = await self.generate_response(messages, temperature=0.1)
        
        try:
            import json
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            logger.warning(f"查询分类结果解析失败: {response}")
            return {
                "type": "general",
                "intent": "无法确定",
                "entities": [],
                "confidence": 0.5
            }
    
    async def should_retrieve(self, query: str, context: List[Dict[str, str]] = None) -> bool:
        """判断是否需要检索外部知识"""
        decision_prompt = f"""
请判断以下查询是否需要检索外部知识库来回答：

查询: {query}

判断标准：
1. 如果查询涉及具体的CVE、漏洞、主机、镜像等信息，需要检索
2. 如果查询是一般性问候或简单对话，不需要检索
3. 如果查询需要最新的安全信息或具体数据，需要检索

请只回答 "是" 或 "否"
"""
        
        messages = [
            {"role": "system", "content": "你是一个智能检索决策器。"},
            {"role": "user", "content": decision_prompt}
        ]
        
        if context:
            # 添加对话上下文
            for msg in context[-3:]:  # 只使用最近3轮对话
                messages.insert(-1, msg)
        
        response = await self.generate_response(messages, temperature=0.1)
        
        return "是" in response.strip()
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            test_messages = [
                {"role": "user", "content": "Hello"}
            ]
            
            response = await self.generate_response(test_messages, max_tokens=10)
            return len(response) > 0
            
        except Exception as e:
            logger.error(f"LLM健康检查失败: {str(e)}")
            return False

# 全局LLM客户端实例
_llm_client = None

async def get_llm_client() -> LLMClient:
    """获取LLM客户端实例"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
