# image-depth-server

`image-depth-server` 是一个 Rust HTTP 微服务。客户端传入酷狗封面数字 hash，服务端下载封面图片，使用 Depth Anything ONNX 模型生成灰度深度图，并返回可直接用于 `<img>` 的 PNG Data URL。

## 功能

- `GET /depth/{hash}`：生成或读取缓存中的深度图。
- `GET /health`：检查服务状态和模型输入尺寸。
- 使用 sled 持久化缓存，相同 hash 和相同模型配置下会复用结果。
- GitHub Release 提供 Linux x64 和 Windows x64 裸可执行文件。

## 模型文件

Release 产物不包含模型文件。运行前需要下载 `model_quantized.onnx`，默认放在程序工作目录：

```powershell
Invoke-WebRequest -Uri "https://huggingface.co/Xenova/depth-anything-small-hf/resolve/main/onnx/model_quantized.onnx" -OutFile "model_quantized.onnx"
```

Linux/macOS：

```bash
curl -L -o model_quantized.onnx "https://huggingface.co/Xenova/depth-anything-small-hf/resolve/main/onnx/model_quantized.onnx"
```

也可以放在任意位置，并通过 `--model` 或环境变量 `DEPTH_MODEL_PATH` 指定。

## 运行

Windows：

```powershell
.\image-depth-server-windows-x64.exe --model .\model_quantized.onnx --port 7860
```

Linux：

```bash
chmod +x ./image-depth-server-linux-x64
./image-depth-server-linux-x64 --model ./model_quantized.onnx --port 7860
```

源码运行：

```bash
cargo run --release -- --model ./model_quantized.onnx --port 7860
```

## 配置

| 参数 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--port` | `DEPTH_PORT` | `7860` | HTTP 服务端口 |
| `--model` | `DEPTH_MODEL_PATH` | `./model_quantized.onnx` | ONNX 模型路径 |
| `--cache-dir` | `DEPTH_CACHE_DIR` | `./cache` | sled 缓存目录 |
| `--download-timeout-secs` | `DEPTH_DOWNLOAD_TIMEOUT_SECS` | `10` | 封面下载超时秒数 |
| `--max-download-bytes` | `DEPTH_MAX_DOWNLOAD_BYTES` | `5242880` | 下载响应最大字节数 |
| `--max-image-size` | `DEPTH_MAX_IMAGE_SIZE` | `1024` | 返回深度图最大边长 |
| `--target-size` | `DEPTH_TARGET_SIZE` | `518` | 动态输入模型的推理尺寸 |
| `--hash-digits` | `DEPTH_HASH_DIGITS` | `20` | hash 数字位数，必须至少为 8 |
| `--infer-concurrency` | `DEPTH_INFER_CONCURRENCY` | `1` | 推理并发数 |

## 接口示例

```http
GET /depth/20190917112902491378
```

成功响应：

```json
{
  "hash": "20190917112902491378",
  "width": 400,
  "height": 400,
  "dataUrl": "data:image/png;base64,...",
  "createdAt": 1793376000000
}
```

健康检查：

```http
GET /health
```

完整接口说明见 [docs/接口文档.md](docs/接口文档.md)。

## 注意事项

- 首次请求需要下载图片并执行 ONNX 推理，耗时会高于缓存命中。
- 深度图是单张图片内的相对深度归一化结果，不适合跨图片比较绝对距离。
- 更换模型文件或输入尺寸后，缓存键会变化，旧缓存不会被复用。
