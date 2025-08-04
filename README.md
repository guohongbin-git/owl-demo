# OWLv2 通用物体检测 Web 应用

这是一个基于 Hugging Face 的 OWLv2 模型构建的通用物体检测 Web 应用程序。它允许用户上传图像，手动框选感兴趣的区域（ROI），然后输入文本查询来检测该区域内的物体。检测结果将可视化地显示在图像上，并列出检测到的物体信息。

## 功能特性

*   **图像上传**：支持上传常见的图像格式。
*   **交互式 ROI 选择**：用户可以在上传的图像上拖动、移动和调整大小来精确选择感兴趣的区域。
*   **文本查询**：支持通过文本描述来指定要检测的物体类别。
*   **物体检测**：利用 OWLv2 模型进行零样本（Zero-shot）物体检测。
*   **结果可视化**：在处理后的图像上绘制边界框和标签。
*   **结果列表**：以列表形式显示检测到的物体信息（分数、标签、边界框）。
*   **图像保存**：支持下载带有检测结果的图像。

## 技术栈

*   **后端**：Python 3.10+ (Flask)
*   **前端**：HTML, CSS, JavaScript (Cropper.js)
*   **AI 模型**：Hugging Face Transformers (OWLv2)
*   **图像处理**：Pillow

## 环境设置

强烈建议使用 `venv` 或 `conda` 创建一个独立的 Python 虚拟环境来管理项目依赖。

### 1. 创建并激活虚拟环境

**使用 `venv` (推荐)**

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

**使用 `conda` (如果您偏好)**

```bash
conda create -n owlv2_env python=3.10 -y
conda activate owlv2_env
```

### 2. 安装依赖

激活虚拟环境后，安装 `requirements.txt` 中列出的所有依赖：

```bash
pip install -r requirements.txt
```

### 3. 下载 Cropper.js

Cropper.js 是通过 CDN 引入的，因此无需手动下载。确保您的网络连接正常。

## 运行应用程序

在项目根目录下，确保您的虚拟环境已激活，然后运行 Flask 应用：

```bash
python app.py
```

应用程序启动后，您将在终端看到类似以下输出：

```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: XXX-XXX-XXX
```

在浏览器中打开 `http://127.0.0.1:5000` 即可访问 Web 界面。

## 使用方法

1.  **上传图像**：点击“上传图像”按钮，选择您要进行物体检测的图片。
2.  **框选感兴趣区域 (ROI)**：图片加载后，您会看到一个可拖动和调整大小的框。拖动鼠标可以绘制新的框，点击框内部可以移动框，拖动框的边缘或角落可以调整框的大小。
3.  **输入查询文本**：在“查询文本”输入框中，输入您想要检测的物体描述，多个物体描述请使用逗号 `,` 分隔（例如：`a photo of a cat, a photo of a dog`）。
4.  **开始检测**：点击“开始检测”按钮。
5.  **查看结果**：等待片刻，处理后的图像将显示在下方，并在图像上绘制出检测到的物体边界框和标签。同时，下方会列出检测到的物体信息。
6.  **保存图像**：点击“保存图像”按钮，可以将带有检测结果的图像下载到本地。

## 故障排除

*   **`IndentationError`**：Python 缩进错误。请检查 `app.py` 文件中的缩进是否正确，Python 严格依赖缩进来定义代码块。
*   **`cannot identify image file`**：这通常是由于后端在尝试读取图像文件时，文件流已关闭或数据不完整。请确保前端正确发送了图像数据，并且后端只读取一次文件流。
*   **Cropper.js 相关问题**：如果 Cropper.js 界面没有正确显示或功能异常，请检查浏览器开发者工具的控制台是否有 JavaScript 错误，并确保 Cropper.js 的 CDN 链接可以正常访问。

## 贡献

欢迎对本项目进行贡献！如果您有任何改进建议或发现 Bug，请随时提交 Issue 或 Pull Request。

## 许可证

本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件（如果存在）。

## 上传到 GitHub

如果您希望将此项目分享到 GitHub，请按照以下步骤操作：

1.  **初始化 Git 仓库**：
    ```bash
    git init
    ```
2.  **添加文件**：
    ```bash
    git add .
    ```
3.  **提交更改**：
    ```bash
    git commit -m "Initial commit: OWLv2 Object Detection Web App"
    ```
4.  **在 GitHub 上创建新仓库**：访问 GitHub 网站，创建一个新的空仓库（不要初始化 README、.gitignore 或 License）。
5.  **关联远程仓库并推送**：
    ```bash
    git remote add origin <您的GitHub仓库URL>
    git push -u origin master
    ```

请确保替换 `<您的GitHub仓库URL>` 为您在 GitHub 上创建的仓库的实际 URL。
