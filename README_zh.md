# Tuzi MCP Tools

[![PyPI version](https://badge.fury.io/py/tuzi-mcp-tools.svg)](https://badge.fury.io/py/tuzi-mcp-tools)

[English](README.md) | 简体中文

一个提供 **CLI** 和 **MCP 服务器** 双重接口的 Python 包，用于使用 Tu-zi.com API 生成图像和进行调研。

## 功能特性

- **双重接口**：CLI 和 MCP 服务器，采用清洁架构
- **异步任务管理**：提交/屏障模式，支持高效并行处理
- **GPT 图像生成**：GPT 文本转图像，支持对话连续性
- **FLUX 图像生成**：FLUX 文本转图像，支持对话跟踪
- **对话管理**：在多个任务间继续图像编辑

## 安装

```bash
pipx install tuzi-mcp-tools
```

## 设置

设置您的 Tu-zi.com API 密钥：

```bash
export TUZI_API_KEY='your_api_key_here'
```

## MCP 服务器使用

```json
{
  "mcpServers": {
    "tuzi": {
      "command": "uvx",
      "args": ["tuzi-mcp-tools"],
      "env": {
        "TUZI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

### 可用的 MCP 工具

MCP 服务器提供**异步任务管理**，采用提交/屏障模式进行高效图像生成。

#### `submit_gpt_image`
提交 GPT 图像生成任务进行异步处理，立即返回任务 ID。

#### `submit_flux_image`
提交 FLUX 图像生成任务进行异步处理，立即返回任务 ID。

#### `task_barrier`
等待所有已提交的图像生成任务完成并下载结果。报告对话 ID 以便任务跟踪。

## CLI 使用

```
uvx --from tuzi-mcp-tools tuzi --help
```