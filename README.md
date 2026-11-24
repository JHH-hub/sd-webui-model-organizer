# SD WebUI Model Organizer（模型整理助手）

轻量、无冲突的 Stable Diffusion WebUI / Forge / Forge Neo 模型整理与 C 站下载扩展（核心引擎与命令行同样适用于 **ComfyUI** 用户）。

> 不篡改内置 Extra Networks 卡片 DOM，可与 **Civitai Helper** 等已有插件零冲突共存。

---

## 核心功能

1. **批量规范重命名**
   - 22 种模型与伴随文件联动重命名（`.safetensors` / `.ckpt` / `.civitai.info` / `.json` / `.preview.*` / `.weilin-info.json` / `.source.json` 等），两阶段临时文件原子更名 + 失败自动回滚。
   - Checkpoint 自动保留版本号，同目录同名模型自动追加版本号防覆盖。
   - 每次批量改名持久化写入 `data/rename_history.json`，保留历史撤销兜底（可一键还原原文件名）。
2. **一键整理体检**
   - **触发词 `.json` 递归补全**：从 `.civitai.info` 的 `trainedWords` 剥离 `<lora:...>` 标签后填入 `activation text`，永不覆盖已有手写提示词。
   - **预览图规范化**：将同名图片规范为 `.preview.<真实扩展名>`（或经 Pillow 无损转码为真 PNG），拒绝盲目改后缀造成的“伪 PNG”。
   - **SHA-256 联网补全**：为手动下载但缺失 `.civitai.info` 的模型按 Hash 回填元数据、预览图与触发词（已内置未收录 Hash 缓存）。
   - **孤儿伴随文件扫描**：只读扫描无对应模型本体的残留 `.info` / `.json` / `.preview.*`，确认后安全移入 `data/quarantine/` 隔离目录，绝不直接删除。
3. **C 站轻量下载与自动入库（原生支持 `.com / .red`）**
   - 粘贴 `civitai.com` / `civitai.red` 模型链接或 ID，解析版本并自动识别推荐入库目录（LoRA / Checkpoint / Embedding / LyCORIS）及子目录。
   - `.part` 断点续传下载模型、保存规范 `.civitai.info` 与 `.preview.<ext>` 封面（可选跳过 NSFW）。
   - 下载完成后自动按 C 站模型名规范改名，并同步提取 `trainedWords` 写入原生 `.json` 触发词。

---

## 安装方法（SD WebUI / Forge / Forge Neo）

1. 在 WebUI 安装界面「从网址安装」填入 `https://github.com/JHH-hub/sd-webui-model-organizer.git`，或直接克隆 / 解压到 `extensions/sd-webui-model-organizer`。
2. **完全关闭并重新启动** SD WebUI / Forge，在顶部标签栏点击 **「模型整理助手」**。
3. 如需配置代理或 API Key，可前往 `Settings → Model Organizer` 设置（或在下载页临时填写代理）。

---

## 独立 CLI 用法（ComfyUI / 命令行）

```bash
# 预览规范重命名建议（只读）
python organizer_cli.py scan "D:\ComfyUI\models\loras" --recursive

# 批量重命名（加 --dry-run 先预演）
python organizer_cli.py rename "D:\ComfyUI\models\loras" --recursive

# 撤销最近一次批量重命名
python organizer_cli.py undo

# 触发词补全 / 预览图规范化 / 孤儿扫描
python organizer_cli.py sync-triggers "D:\ComfyUI\models\loras" --recursive
python organizer_cli.py normalize-previews "D:\ComfyUI\models\loras" --recursive --mode keep_ext
python organizer_cli.py scan-orphans "D:\ComfyUI\models\loras" --recursive

# C 站下载并自动整理入库
python organizer_cli.py download "https://civitai.red/models/123456" "D:\ComfyUI\models\loras" --domain civitai.red
```

---

## 开源协议

[MIT License](./LICENSE) · By [像素离散 Pxlsan](https://pxlsan.cn)
