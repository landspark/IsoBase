# 搜索工具 (Search Tool)

> For English documentation, see [README.md](./README.md)

`isobase.llm.tools.search` 模块为大语言模型（LLM）提供了一套企业级、双引擎架构的联网搜索工具集成方案。它既能无缝利用现代 LLM 服务商原生自带的搜索能力，也能在你需要时优雅地挂载第三方的定制化搜索引擎。

## 架构设计

该搜索模块采用了 **双架构 (Dual-Architecture)** 设计，旨在最大限度地减少开发配置，同时在不同的 LLM 厂商间保持极致的健壮性。

1. **原生/内置搜索 (Native/Built-in Search)**
   - 当你使用的 LLM 提供商原生支持联网搜索时（例如 OpenAI、Qwen 阿里云百炼、GLM 智谱），`SearchTool` 会被 `IsoBase` 底层的适配器（如 `OpenAIChat`）拦截。
   - 适配器会自动将这个中立的工具转化为该厂商特有的原生 JSON Schema（例如 `{"type": "web_search"}`）。
   - 搜索行为完全由 LLM 厂商在服务端闭环执行，这为你节省了 Token、降低了延迟，并免去了维护搜索 API Key 的麻烦。

2. **自定义搜索服务 (Custom Search Provider)**
   - 对于暂不自带搜索能力的 LLM（如 DeepSeek 官方 API、Anthropic），或者当你需要严格控制使用特定的外接搜索引擎时，你可以将一个 `BaseSearchProvider` 挂载到 `SearchTool` 内部。
   - 此时，`SearchTool` 会退化为一个标准的本地 Function Calling 工具。当大模型决定要搜索时，`IsoBase` 会在本地执行你的搜索代码（例如请求 Tavily 或 Brave），并将抓取到的网页摘要塞回对话历史，供大模型进行二次思考。

## 功能特性

- **解耦的搜索逻辑**：你可以通过继承 `BaseSearchProvider` 协议，轻松地把你们企业内网的 ElasticSearch、知识库（RAG）或私有 Bing 接口包装成搜索服务，而无需改动任何 LLM 调度逻辑。
- **开箱即用的内置提供商**：
  - `BraveSearchProvider`：基于轻量级 REST API 的传统搜索，拥有极其慷慨的免费额度（每月 2000 次），适合生产环境的低成本兜底。
  - `TavilySearchProvider`：专为 AI Agent 打造的深度搜索。它不仅返回摘要，还在后台帮你解析并提取干净的网页 Markdown 正文全文供模型阅读。
- **严格的模型命名约束**：无论底层挂载的是什么稀奇古怪的搜索提供商，对外暴露给大模型的工具名称永远被强制统一为 `web_search`（除非开发者主动覆盖）。这极大降低了大模型识别工具的理解成本和幻觉率。

## 快速开始

### 使用原生搜索（推荐搭配 OpenAI / Qwen）

你只需要实例化一个空的 `SearchTool`，剩下的全由底层的 `OpenAIChat` 适配器去和厂商沟通。

```python
from isobase.llm import OpenAIChat
from isobase.llm.tools.search import SearchTool

# 默认 force_external=False。适配器会将其转化为原生的 {"type": "web_search"}。
search_tool = SearchTool()

client = OpenAIChat(api_key="sk-...", tools=[search_tool])
response = client.ask("请问东京今天的天气如何？")
print(response.content)
```

### 使用外置自定义搜索（搭配 Anthropic / DeepSeek）

将一个具体的 `BaseSearchProvider` 实例传给 `SearchTool`。如果你想在使用 OpenAI 时也强行禁用它的内置搜索，转而使用你的本地搜索，可以设置 `force_external=True`。

```python
from isobase.llm import AnthropicMessages
from isobase.llm.tools.search import SearchTool, TavilySearchProvider

tavily_provider = TavilySearchProvider(api_key="tvly-...")

# 对于 Anthropic 或被 force_external=True 标记时，它会走标准的本地工具调用流程
search_tool = SearchTool(provider=tavily_provider, force_external=True)

client = AnthropicMessages(api_key="sk-ant-...", tools=[search_tool])
response = client.ask("帮我总结一下今天关于日本签证费上涨的新闻。")
print(response.content)
```

## 创建企业内网搜索提供商

想要让大模型学会搜索企业内网的数据？只需继承 `BaseSearchProvider` 并实现 `search` 方法即可：

```python
from typing import Any, Dict, List
from isobase.llm.tools.search import BaseSearchProvider, SearchTool

class CorporateWikiSearchProvider(BaseSearchProvider):
    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        # 在这里写你请求内网数据库或知识库的代码
        return [
            {
                "title": "公司内部远程办公规定",
                "url": "https://wiki.corp.local/policy",
                "snippet": "关于2026年的员工差旅与打卡要求..."
            }
        ]

# 把你的内网知识库包装成工具交给大模型！
internal_search = SearchTool(provider=CorporateWikiSearchProvider(), force_external=True)
```
