from flask import Flask, request, render_template, jsonify, send_file
from transformers import AutoProcessor, Owlv2ForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import io
import base64
import os

app = Flask(__name__)

# 加载模型和处理器 (只加载一次)
processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    try:
        # 读取文件内容一次
        image_bytes_content = file.read()
        print(f"Received file: {file.filename}, MimeType: {file.mimetype}, Size: {len(image_bytes_content)} bytes")
        print(f"First 20 bytes of image data: {image_bytes_content[:20]}")

        # 使用原始图像内容创建PIL Image对象
        original_image = Image.open(io.BytesIO(image_bytes_content)).convert("RGB")
        print("Original image opened successfully by PIL.")
        original_image_size = original_image.size # Store original size

        # 初始化用于模型推理的图像和裁剪偏移
        image_for_inference = original_image.copy()
        cropped_offset_x, cropped_offset_y = 0, 0

        # 获取ROI坐标
        roi_x = request.form.get('roi_x')
        roi_y = request.form.get('roi_y')
        roi_width = request.form.get('roi_width')
        roi_height = request.form.get('roi_height')

        if roi_x and roi_y and roi_width and roi_height:
            try:
                roi_x = int(float(roi_x))
                roi_y = int(float(roi_y))
                roi_width = int(float(roi_width))
                roi_height = int(float(roi_height))
                
                # 裁剪图像用于推理
                image_for_inference = original_image.crop((roi_x, roi_y, roi_x + roi_width, roi_y + roi_height))
                cropped_offset_x, cropped_offset_y = roi_x, roi_y
                print(f"Image cropped to ROI: {roi_x},{roi_y},{roi_width},{roi_height}")
            except ValueError:
                print("Invalid ROI coordinates received. Ignoring ROI.")
                pass

        texts_str = request.form.get('query_texts', '')
        if not texts_str:
            return jsonify({'error': 'No query texts provided'}), 400
        
        # 将文本查询字符串转换为列表的列表
        texts = [[t.strip()] for t in texts_str.split(',')]

        # 运行推理
        inputs = processor(images=image_for_inference, text=texts, return_tensors="pt")
        outputs = model(**inputs)

        # 解析输出
        target_sizes = torch.tensor([image_for_inference.size[::-1]])
        results = processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

        detected_objects = []
        # 创建一个可绘制的原始图像副本，用于绘制边界框
        draw_image = original_image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # 尝试加载一个默认字体，如果失败则使用PIL的默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 20) # 假设arial.ttf存在于系统路径
        except IOError:
            font = ImageFont.load_default()

        for i, result in enumerate(results):
            for score, label_idx, box in zip(result["scores"], result["labels"], result["boxes"]):
                # 边界框是相对于裁剪图像的，需要调整回原始图像坐标
                x_min, y_min, x_max, y_max = box.tolist()
                x_min += cropped_offset_x
                y_min += cropped_offset_y
                x_max += cropped_offset_x
                y_max += cropped_offset_y

                adjusted_box = [round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)]
                predicted_label = texts[0][label_idx] # 假设只有一个文本列表

                detected_objects.append({
                    'score': round(score.item(), 3),
                    'label': predicted_label,
                    'box': adjusted_box
                })

                # 绘制边界框
                draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=3)
                
                # 绘制标签
                draw.text((x_min + 5, y_min + 5), f"{predicted_label}: {round(score.item(), 2)}", fill="red", font=font)

        # 将处理后的图像转换为Base64编码
        buffered = io.BytesIO()
        draw_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'status': 'success',
            'detected_image': encoded_image,
            'objects': detected_objects
        })

    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)