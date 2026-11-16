# Omni Studio

本地多模态聊天网页：文字、图片、视频、音频输入，流式文本回答，多轮会话，以及多模型服务切换。Python 中间层 + 无构建步骤的静态前端；不需要 Node、不加载模型权重、不占 GPU。

## 启动

在已安装 SGLang 依赖的环境中：

```bash
WEBUI_PYTHON=/usr/bin/python3 ./examples/multimodal_webui/deploy.sh
```

打开 **http://127.0.0.1:7860**。脚本在前台运行，Ctrl+C 停止网页及中间层；不会停止模型服务。

全新环境可用 `./examples/multimodal_webui/deploy.sh --install` 安装依赖到此目录的 `.venv`（需要系统 `python3-venv`）。音视频还需要 `ffmpeg` 和 `ffprobe`。BeeBee 使用 checkpoint 的本地 tokenizer，建议使用已成功运行该 checkpoint 的 Python / transformers 环境。

远程服务器可以转发端口：

```bash
ssh -L 7860:127.0.0.1:7860 user@server
```

局域网访问可指定 `--host 0.0.0.0 --port 7860`。设置环境变量 `WEBUI_TOKEN` 可启用浏览器密码登录（用户名任意，密码为该变量）；公网部署时应由 HTTPS 反向代理提供 TLS。网页和附件 API 使用同一认证。

## 两个或更多模型服务

网页端口和模型服务端口独立配置。以下命令在两个终端分别启动模型，每个服务占用一张卡：

```bash
./examples/multimodal_webui/launch_model.sh --gpu 0 --port 18006
./examples/multimodal_webui/launch_model.sh --gpu 1 --port 18007
```

默认使用 1106 BeeBee checkpoint 和已经验证的启动参数。可用 `--model-path /path/to/checkpoint` 切换 checkpoint，`--tokenizer-path` 单独指定 tokenizer，`--python` 指定运行环境。其他模型使用 `--profile generic --model-path /path/to/model`；通用模式也接受 Hugging Face 模型 ID。`--` 后可以追加 SGLang 原生参数，例如 `-- --tp-size 2`；多卡时用 `--gpu 0,1` 指定可见 GPU，并设置匹配的 `--tp-size`。运行 `launch_model.sh --help` 查看全部选项。默认模型服务仅监听本机；需要远程访问时可指定 `--host 0.0.0.0`。

另开一个终端启动网页，例如使用 7861 端口：

```bash
WEBUI_PYTHON=/usr/bin/python3 ./examples/multimodal_webui/deploy.sh --port 7861 --scan-ports 18006,18007
```

打开 `http://127.0.0.1:7861`，选择对应端口的模型服务。远程使用时相应调整 SSH 转发端口。

先独立启动模型服务。网页每 10 秒扫描 Linux 本机监听端口，并请求 `/v1/models` 确认服务；也会探测默认端口 `8000,30000,18005,18006`。可以通过 `--scan-ports 18005,18006,8001` 设置额外候选端口。

同一模型在两个端口部署会显示为两个选项，不会被合并。下线的服务标记离线；恢复后使用相同标识。刷新不加载或重启模型。只扫描 loopback，不扫描局域网。Web UI 与模型须处于同一网络命名空间；容器之间需显式安排网络。

点击左侧「模型服务」可添加地址和名称，保存在 `services.local.json`。也可从 [services.example.json](services.example.json) 复制配置，修改成实际模型：

```bash
cp examples/multimodal_webui/services.example.json examples/multimodal_webui/services.local.json
```

**示例的第二个服务是 OpenAI 适配器，不要不加修改地用于第二个 BeeBee 模型。** 一般无需配置适配器：中间层读取 `/get_model_info` 中的模型路径及其 `config.json`，自动识别 BeeBee。无法读取本地模型配置、或模型 ID 使用别名时，显式配置 `adapter`、`tokenizer_path`、`capabilities`。

支持的配置字段：

| 字段 | 含义 |
| --- | --- |
| `url` | 本机 HTTP 根地址，`/v1` 后缀也可输入 |
| `name` | 网页显示名称 |
| `adapter` | `openai` 或 `beebee`，可省略自动判断 |
| `tokenizer_path` | BeeBee checkpoint/tokenizer 目录 |
| `capabilities` | 输入能力，如 `["text","image","video","audio"]` |
| `api_key_env` | 上游 API key 所在环境变量名；不把密钥写入网页 |

配置会在下次发现时重新读取。能力来自模型配置及名称推断，无法识别时显示「待确认」并允许尝试；最终是否支持由模型决定。视频抽帧对支持图片的标准视觉模型也适用。

## 对话与媒体协议

- **标准服务**：使用 `/v1/chat/completions`，传入 `chat_template_kwargs.enable_thinking`。图片为 `image_url`，音频为 `input_audio`（WAV），视频为带时间戳的均匀采样图像。服务必须支持对应模态及 chat-template 参数。
- **BeeBee**：使用 `/generate`，从当前 checkpoint 的 `AutoTokenizer.apply_chat_template` 渲染原生 prompt，显式传递 `enable_thinking`。普通图片使用 temporal pair。视频可选 2/4/6 FPS 采样，采样帧数受设置中的帧数上限约束；27B 格式按每组 FPS 张图添加一组视觉边界，每帧只传一次、不复制，14B 格式则将全部采样帧放入单组视觉边界。音频 `<audio>` 位于 user 内容中。视觉标记位于所属 user turn 之前，下采样通过 `image_downsample_ratios` 传入。
- 图片最长边默认 1280，按 32 对齐；JPEG quality 90。视频默认均匀采样 16 帧，可选 8/32/64。音视频时长上限 10 分钟，可选同时读取视频音轨。音频转换为 16 kHz 单声道 WAV。只提供**媒体输入 / 文本输出**，不合成语音或视频。
- 默认 non-thinking、1024 输出 token、downsample=4、temperature=0.7、top_p=0.8、presence_penalty=1.5。设置面板支持修改。Downsample 仅对 BeeBee 生效，网页媒体预处理与 benchmark 不完全相同。
- 推理和正文分别展示；首段空推理如实标记，重新开启的推理不会被丢弃。原始输出可下载 JSON。达到 token 上限会提示。按钮可停止流式连接，停止后的已收到内容仍保留。
- 对话保存在此浏览器的 localStorage，最多 30 个会话；媒体保存在 `.data/media`，闲置 7 天后在下次上传时清理。每个附件最大 100 MB，总存储 2 GB；每条消息最多 8 个附件，完整历史最多 24 个附件、128 帧。
- 服务列表及附件存储属于同一个本地工作空间，不是多用户隔离产品。清空对话只清空浏览器历史，附件按上述规则过期。切换模型会携带当前会话上下文；想要独立上下文请新建对话。

## 验证

CPU 协议回归测试：

```bash
FLASHINFER_WORKSPACE_BASE=/tmp/omni-test-flashinfer \
  /usr/bin/python3 -m pytest test/registered/unit/webui/test_multimodal_gateway.py -q
```

覆盖多端口模型发现/离线恢复、两种流式协议、原生多轮媒体顺序、thinking 空段/重开段、附件校验及配置保持。此测试使用模拟上游服务，不验证模型回答质量。

本次还使用两个临时模拟服务做了 Chromium 实测：实际上传 PNG、带音轨 MP4、WAV，检查模型切换、多轮历史、停止生成、推理展示、HTML 转义以及 390px 手机布局。BeeBee prompt 另外使用当前 1106 checkpoint 的真实 tokenizer 验证。截图和验证记录位于仓库 `outputs/webui-preview/`；未启动 GPU 模型做回答质量验证。

界面附带约 120 KB 的 Noto Sans SC 字符子集用于中文界面兜底，不需要外部字体 CDN；普通中文系统字体优先。字体遵循 SIL OFL，许可证保留在 `static/OFL-NotoSansSC.txt`。
