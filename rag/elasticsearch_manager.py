"""
Elasticsearch文档存储管理模块

该模块负责将文档数据解析并存储到Elasticsearch中，提供关键词检索功能。
支持文档索引、搜索、更新和删除操作。

数据流：
1. 文档解析 -> 文本提取 -> 分词处理 -> ES索引
2. 关键词查询 -> ES搜索 -> 结果排序 -> 返回文档

学习要点：
1. Elasticsearch的基本操作和索引管理
2. 中文文本的分词和索引策略
3. 关键词检索和相关性评分
4. 文档存储的数据结构设计
"""

import json
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from config import defaultConfig
from logger import get_logger

# 尝试导入Elasticsearch，如果失败则使用模拟版本
try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import ConnectionError, NotFoundError, RequestError
    ES_AVAILABLE = True
except ImportError:
    print("⚠️  elasticsearch库未安装，使用模拟ES客户端")
    ES_AVAILABLE = False
    ConnectionError = Exception
    NotFoundError = Exception
    RequestError = Exception


@dataclass
class DocumentRecord:
    """文档记录数据结构"""
    doc_id: str
    title: str
    content: str
    source: str
    doc_type: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    content_length: int
    keywords: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        # 转换datetime为字符串
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentRecord':
        """从字典创建实例"""
        # 转换字符串为datetime
        if isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    doc_id: str
    title: str
    content: str
    score: float
    highlights: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


class MockElasticsearch:
    """模拟Elasticsearch客户端"""
    
    def __init__(self, **kwargs):
        """初始化模拟ES客户端"""
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.indices: Dict[str, Dict[str, Any]] = {}
        self.connected = True
    
    def ping(self) -> bool:
        """模拟ping操作"""
        return True
    
    def info(self) -> Dict[str, Any]:
        """获取集群信息"""
        return {
            "version": {"number": "mock-8.0.0"},
            "cluster_name": "mock-cluster"
        }
    
    def indices_exists(self, index: str) -> bool:
        """检查索引是否存在"""
        return index in self.indices
    
    def indices_create(self, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """创建索引"""
        self.indices[index] = {
            "settings": body.get("settings", {}),
            "mappings": body.get("mappings", {})
        }
        return {"acknowledged": True}
    
    def index(self, index: str, id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """索引文档"""
        if index not in self.indices:
            self.indices_create(index, {})
        
        if index not in self.documents:
            self.documents[index] = {}
        
        self.documents[index][id] = body
        return {"_id": id, "result": "created"}
    
    def get(self, index: str, id: str) -> Dict[str, Any]:
        """获取文档"""
        if index not in self.documents or id not in self.documents[index]:
            raise NotFoundError("Document not found")
        
        return {
            "_id": id,
            "_source": self.documents[index][id],
            "found": True
        }
    
    def search(self, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """搜索文档"""
        if index not in self.documents:
            return {"hits": {"total": {"value": 0}, "hits": []}}
        
        # 简单的模拟搜索
        query = body.get("query", {})
        size = body.get("size", 10)
        
        hits = []
        for doc_id, doc in self.documents[index].items():
            # 简单的关键词匹配
            if self._match_query(doc, query):
                hits.append({
                    "_id": doc_id,
                    "_score": 1.0,
                    "_source": doc,
                    "highlight": {}
                })
        
        return {
            "hits": {
                "total": {"value": len(hits)},
                "hits": hits[:size]
            }
        }
    
    def _match_query(self, doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """简单的查询匹配"""
        if "match" in query:
            for field, value in query["match"].items():
                if field in doc and str(value).lower() in str(doc[field]).lower():
                    return True
        elif "multi_match" in query:
            query_text = query["multi_match"]["query"].lower()
            fields = query["multi_match"]["fields"]
            for field in fields:
                if field in doc and query_text in str(doc[field]).lower():
                    return True
        return False


class ElasticsearchManager:
    """Elasticsearch文档存储管理器
    
    负责文档的索引、搜索、更新和删除操作。
    支持中文分词和关键词检索。
    
    Attributes:
        es_client: Elasticsearch客户端
        index_name: 索引名称
        logger: 日志记录器
    """
    
    def __init__(self):
        """初始化ES管理器"""
        self.logger = get_logger("ElasticsearchManager")
        self.config = defaultConfig.elasticsearch
        self.index_name = self.config.index_name
        
        # 初始化ES连接
        self._init_elasticsearch_connection()
        
        # 确保索引存在
        self._ensure_index_exists()
        
        self.logger.info(f"Elasticsearch管理器初始化完成 | 索引: {self.index_name}")
    
    def _init_elasticsearch_connection(self):
        """初始化Elasticsearch连接"""
        try:
            if ES_AVAILABLE:
                es_config = self.config
                self.es_client = Elasticsearch(
                    [{"host": es_config.host, "port": es_config.port}],
                    http_auth=(es_config.username, es_config.password) if es_config.username else None,
                    use_ssl=es_config.use_ssl,
                    verify_certs=es_config.verify_certs,
                    timeout=es_config.timeout,
                    max_retries=es_config.max_retries
                )
                
                # 测试连接
                if self.es_client.ping():
                    self.logger.info("Elasticsearch连接成功")
                else:
                    raise ConnectionError("Elasticsearch ping失败")
            else:
                raise ImportError("elasticsearch库未安装")
                
        except Exception as e:
            self.logger.warning(f"Elasticsearch连接失败，使用模拟ES客户端: {e}")
            self.es_client = MockElasticsearch()
            self.logger.info("已启用模拟Elasticsearch客户端")
    
    def _ensure_index_exists(self):
        """确保索引存在"""
        try:
            if hasattr(self.es_client, 'indices') and hasattr(self.es_client.indices, 'exists'):
                # 真实ES客户端
                if not self.es_client.indices.exists(index=self.index_name):
                    self._create_index()
            else:
                # 模拟ES客户端
                if not self.es_client.indices_exists(self.index_name):
                    self._create_index()
                    
        except Exception as e:
            self.logger.error(f"检查索引存在性失败: {e}")
            raise
    
    def _create_index(self):
        """创建索引"""
        try:
            # 定义索引映射和设置
            index_body = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "chinese_analyzer": {
                                "type": "custom",
                                "tokenizer": "ik_max_word",
                                "filter": ["lowercase", "stop"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": "chinese_analyzer",
                            "search_analyzer": "chinese_analyzer"
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "chinese_analyzer",
                            "search_analyzer": "chinese_analyzer"
                        },
                        "source": {"type": "keyword"},
                        "doc_type": {"type": "keyword"},
                        "metadata": {"type": "object"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"},
                        "content_length": {"type": "integer"},
                        "keywords": {"type": "keyword"}
                    }
                }
            }
            
            if hasattr(self.es_client, 'indices') and hasattr(self.es_client.indices, 'create'):
                # 真实ES客户端
                self.es_client.indices.create(index=self.index_name, body=index_body)
            else:
                # 模拟ES客户端
                self.es_client.indices_create(self.index_name, index_body)
            
            self.logger.info(f"创建索引成功: {self.index_name}")
            
        except Exception as e:
            self.logger.error(f"创建索引失败: {e}")
            raise
    
    def index_document(self, document: DocumentRecord) -> bool:
        """索引文档
        
        Args:
            document: 文档记录
            
        Returns:
            是否索引成功
        """
        try:
            doc_body = document.to_dict()
            
            if hasattr(self.es_client, 'index'):
                # 真实ES客户端
                response = self.es_client.index(
                    index=self.index_name,
                    id=document.doc_id,
                    body=doc_body
                )
            else:
                # 模拟ES客户端
                response = self.es_client.index(
                    index=self.index_name,
                    id=document.doc_id,
                    body=doc_body
                )
            
            self.logger.debug(f"文档索引成功: {document.doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"文档索引失败: {e}")
            return False
    
    def search_documents(
        self, 
        query: str, 
        size: int = 10,
        fields: List[str] = None
    ) -> List[SearchResult]:
        """搜索文档
        
        Args:
            query: 搜索查询
            size: 返回结果数量
            fields: 搜索字段列表
            
        Returns:
            搜索结果列表
        """
        try:
            if fields is None:
                fields = ["title^2", "content", "keywords^1.5"]
            
            # 构建搜索查询
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": fields,
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "content": {"fragment_size": 150, "number_of_fragments": 3}
                    }
                },
                "size": size,
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"updated_at": {"order": "desc"}}
                ]
            }
            
            if hasattr(self.es_client, 'search'):
                # 真实ES客户端
                response = self.es_client.search(index=self.index_name, body=search_body)
            else:
                # 模拟ES客户端
                response = self.es_client.search(index=self.index_name, body=search_body)
            
            # 解析搜索结果
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                highlights = []
                
                if "highlight" in hit:
                    for field, fragments in hit["highlight"].items():
                        highlights.extend(fragments)
                
                result = SearchResult(
                    doc_id=source["doc_id"],
                    title=source["title"],
                    content=source["content"],
                    score=hit["_score"],
                    highlights=highlights,
                    metadata=source.get("metadata", {})
                )
                results.append(result)
            
            self.logger.debug(f"搜索完成: 查询='{query}', 结果数={len(results)}")
            return results
            
        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """获取文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档记录或None
        """
        try:
            if hasattr(self.es_client, 'get'):
                # 真实ES客户端
                response = self.es_client.get(index=self.index_name, id=doc_id)
            else:
                # 模拟ES客户端
                response = self.es_client.get(index=self.index_name, id=doc_id)
            
            if response.get("found", False):
                return DocumentRecord.from_dict(response["_source"])
            else:
                return None
                
        except (NotFoundError, Exception) as e:
            self.logger.debug(f"文档不存在或获取失败: {doc_id}, {e}")
            return None
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取ES连接信息
        
        Returns:
            连接信息字典
        """
        try:
            if hasattr(self.es_client, 'info'):
                # 真实ES客户端
                info = self.es_client.info()
            else:
                # 模拟ES客户端
                info = self.es_client.info()
            
            return {
                "connected": True,
                "cluster_name": info.get("cluster_name", "unknown"),
                "version": info.get("version", {}).get("number", "unknown"),
                "index_name": self.index_name,
                "host": self.config.host,
                "port": self.config.port
            }
        except Exception as e:
            self.logger.error(f"获取ES连接信息失败: {e}")
            return {"connected": False, "error": str(e)}
