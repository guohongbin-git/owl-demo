from transformers import AutoProcessor, Owlv2ForObjectDetection
from PIL import Image
import requests

# 1. 加载模型和处理器
processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# 2. 准备输入
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 为了演示，我们创建一个简单的空白图像
# image = Image.new('RGB', (600, 400), color = 'white')

texts = [["a photo of a cat", "a photo of a dog"]]

# 3. 运行推理
inputs = processor(images=image, text=texts, return_tensors="pt")
outputs = model(**inputs)

# 4. 解析输出 (这里只打印原始输出，后续可以进一步处理)
import torch
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt

# ... (之前的代码) ...

# 4. 解析输出并可视化
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

print("检测结果:")
for i, result in enumerate(results):
    print(f"图像 {i+1}:")
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        predicted_label = texts[0][label]
        print(f"  分数: {round(score.item(), 3)}, 标签: {predicted_label}, 边界框: {box}")

        # 绘制边界框
        x_min, y_min, x_max, y_max = box
        draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=3)
        
        # 绘制标签
        # 尝试加载一个默认字体，如果失败则使用PIL的默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 20) # 假设arial.ttf存在于系统路径
        except IOError:
            font = ImageFont.load_default()
        draw.text((x_min + 5, y_min + 5), f"{predicted_label}: {round(score.item(), 2)}", fill="red", font=font)

# 显示图像
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.axis('off')
plt.show()

