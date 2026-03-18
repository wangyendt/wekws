# 唤醒词录音标注台

这是一个独立的 Streamlit WebUI，用来做：

- 浏览器内录音
- 后端调用当前项目的 `infer_wav` 推理
- 人工把录音标成 `嗨小问` / `你好问问` / `非唤醒词`
- 自动统计 precision、recall、overall accuracy 和混淆矩阵
- 左侧历史录音列表点击回放

## 运行方式

在 `examples/hi_xiaowen/s0` 目录下执行：

```bash
streamlit run record_label_webui/app.py
```

## 数据存储

运行时数据保存在：

- `record_label_webui/runtime/recordings/`：录下来的音频
- `record_label_webui/runtime/records.json`：推理结果和人工标签

这些运行时文件已加入 `.gitignore`，不会默认提交。
