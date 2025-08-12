from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any


class AdDetection(BaseModel):
    """广告识别模型"""

    is_ad: bool = Field(..., description="是否为广告。")
    ad_type: Optional[str] = Field(
        None,
        description="广告类型，如“电商广告”、“品牌宣传”等。如果不是广告，则为null。",
    )
    ad_keywords: Optional[List[str]] = Field(
        None, description="涉及广告的关键词列表，如“扫码”、“加vx”等。如果无，则为null。"
    )
    reasoning: str = Field(..., description="判断为广告或非广告的理由。")


class SentimentAnalysis(BaseModel):
    """情感分析模型"""

    overall_score: int = Field(
        ...,
        ge=-10,
        le=10,
        description="基于情感倾向的总体评分，负分代表负面，正分代表正面，0代表中性。情感强度越高，分数绝对值越大。",
    )
    positive_score: int = Field(
        ..., ge=0, le=10, description="正面情感的分值，范围0-10。"
    )
    negative_score: int = Field(
        ..., ge=0, le=10, description="负面情感的分值，范围0-10。"
    )
    sentiment_keywords: List[str] = Field(
        ...,
        description="影响情感判断的关键词列表，例如“剧毒”、“触目惊心”、“还好没有”。",
    )
    analysis_details: str = Field(
        ...,
        description="简要分析情感产生的原因，包括识别到的情感词、否定词、程度副词以及它们对情感倾向的影响。",
    )


class EventTriple(BaseModel):
    """事件三元组模型"""

    subject: str = Field(..., description="事件的施事或主体。")
    predicate: str = Field(..., description="事件的谓词或动作。")
    object: str = Field(..., description="事件的受事或客体。")
    description: str = Field(..., description="结合主谓客体，用一句话概括事件。")


class AnalysisResult(BaseModel):
    """最终分析结果模型"""

    event_triples: List[EventTriple]
    sentiment_analysis: SentimentAnalysis
    ad_detection: AdDetection


class LLMResponse(BaseModel):
    """LLM输出的顶层模型"""

    analysis_result: Optional[AnalysisResult] = None
