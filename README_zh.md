# Tuzi MCP Tools

[![PyPI version](https://badge.fury.io/py/tuzi-mcp-tools.svg)](https://badge.fury.io/py/tuzi-mcp-tools)

[English](README.md) | 简体中文

一个提供 **CLI** 和 **MCP 服务器** 双重接口的 Python 包，用于使用 Tu-zi.com API 生成图像和进行调研。

## 功能特性

- **双重接口**：CLI 和 MCP 服务器
- **GPT 图像生成**：GPT 文本转图像
- **FLUX 图像生成**：FLUX 文本转图像
- **调研/查询**：具有实时网络搜索功能的高级 AI 驱动研究
- **多种格式**：PNG、JPEG、WebP 格式及质量设置
- **实时进度**：流式生成和进度跟踪

## 安装

```bash
pipx install tuzi-mcp-tools
```

## 设置

设置您的 Tu-zi.com API 密钥：

```bash
export TUZI_API_KEY='your_api_key_here'
```

## CLI 使用

### 图像生成

```bash
# 使用智能模型选择生成图像
tuzi gpt-image "山峦上的美丽日落"

# 高质量自定义选项
tuzi gpt-image "一只可爱的猫" --quality high --size 1024x1536 --format png

# 透明背景
tuzi gpt-image "公司标志" --background transparent --output logo.png

# 使用参考图像进行风格转换或修改
tuzi gpt-image "让这个更加丰富多彩" --input-image reference.jpg

# 使用新提示转换现有图像
tuzi gpt-image "将其转换为赛博朋克场景" --input-image photo.png
```

### FLUX 图像生成

```bash
# 使用 FLUX 生成高质量图像
tuzi flux-image "山峦上的美丽日落"

# 自定义宽高比和格式
tuzi flux-image "未来主义城市景观" --aspect-ratio 16:9 --format webp

# 使用参考图像进行风格转换或修改
tuzi flux-image "让这个更加丰富多彩" --input-image reference.jpg --seed 42

# 使用新提示转换现有图像
tuzi flux-image "将其转换为赛博朋克场景" --input-image photo.png --aspect-ratio 16:9

# 超宽全景图像
tuzi flux-image "山地景观全景" --aspect-ratio 21:9 --output panorama.png
```

### 调研/查询

```bash
# 使用网络搜索功能提问
tuzi survey "AI 的最新发展是什么？"

# 获取当前信息
tuzi survey "纽约当前的天气如何？"

# 显示思考过程
tuzi survey "解释量子计算" --show-thinking

# 为复杂主题启用深度分析模式
tuzi survey "分析量子计算对密码学的影响" --deep
```

### CLI 选项

#### 图像生成选项 (`tuzi gpt-image`)

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--quality` | 图像质量 (low, medium, high, auto) | `auto` |
| `--size` | 尺寸 (1024x1024, 1536x1024, 1024x1536, auto) | `auto` |
| `--format` | 输出格式 (png, jpeg, webp) | `png` |
| `--background` | 背景 (opaque, transparent) | `opaque` |
| `--output` | 输出文件路径 | 自动生成 |
| `--input-image` | 参考图像文件路径 | `None` |
| `--compression` | 压缩级别 0-100 (JPEG/WebP) | `None` |
| `--no-stream` | 禁用流式响应 | `False` |
| `--verbose` | 显示完整 API 响应 | `False` |

#### 调研选项 (`tuzi survey`)

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--no-stream` | 禁用流式响应 | `False` |
| `--verbose` | 显示详细响应信息 | `False` |
| `--show-thinking` | 除最终答案外还显示思考过程 | `False` |
| `--deep` | 启用高级分析模式 | `False` |

#### FLUX 生成选项 (`tuzi flux-image`)

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--aspect-ratio` | 图像宽高比 (1:1, 16:9, 9:16, 4:3, 3:4, 21:9, 9:21) | `1:1` |
| `--format` | 输出格式 (png, jpg, jpeg, webp) | `png` |
| `--seed` | 可重现生成种子 | `None` |
| `--input-image` | 参考图像文件路径 | `None` |
| `--output` | 输出文件路径 | 自动生成 |
| `--verbose` | 显示完整 API 响应 | `False` |

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

#### `gpt_image`
使用 GPT 生成图像。

**参数：**
- `prompt` (字符串)：图像生成的文本提示
- `quality` (字符串)：图像质量 (auto, low, medium, high)
- `size` (字符串)：图像尺寸 (auto, 1024x1024, 1536x1024, 1024x1536)
- `format` (字符串)：输出格式 (png, jpeg, webp)
- `background` (字符串)：背景类型 (opaque, transparent)
- `compression` (整数)：JPEG/WebP 的压缩级别 0-100
- `output_path` (字符串)：保存图像的完整路径
- `input_image_path` (字符串，可选)：参考图像文件路径

#### `flux_image`
使用 FLUX 生成图像。

**参数：**
- `prompt` (字符串)：FLUX 图像生成的文本提示
- `aspect_ratio` (字符串)：图像宽高比 (1:1, 16:9, 9:16, 4:3, 3:4, 21:9, 9:21)
- `output_format` (字符串)：输出格式 (png, jpg, jpeg, webp)
- `seed` (整数，可选)：可重现生成种子
- `input_image_path` (字符串，可选)：参考图像文件路径
- `output_path` (字符串)：保存图像的完整路径

#### `survey`
使用具有实时网络搜索功能的高级 AI 调研/查询主题。

**参数：**
- `prompt` (字符串)：自然语言查询/问题
- `show_thinking` (布尔值)：是否在响应中包含思考过程
- `deep` (布尔值)：是否启用高级分析模式（默认：False）