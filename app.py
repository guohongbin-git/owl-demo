from flask import Flask, request, render_template, jsonify, Response
from transformers import AutoProcessor, Owlv2ForObjectDetection, BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
import base64
import os
import uuid
import json
import time
import traceback

app = Flask(__name__)

# --- Model Loading ---
print("Loading models, please wait...")
owl_processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Models loaded successfully.")

# --- Task Storage (In-memory) ---
TASKS = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'query_image' not in request.files:
        return jsonify({'error': 'No query image provided'}), 400
    
    query_file = request.files['query_image']
    try:
        query_image_bytes = query_file.read()
        query_image = Image.open(io.BytesIO(query_image_bytes)).convert("RGB")

        roi_x = int(float(request.form.get('roi_x', 0)))
        roi_y = int(float(request.form.get('roi_y', 0)))
        roi_width = int(float(request.form.get('roi_width', query_image.width)))
        roi_height = int(float(request.form.get('roi_height', query_image.height)))
        query_image_cropped = query_image.crop((roi_x, roi_y, roi_x + roi_width, roi_y + roi_height))

        blip_inputs = blip_processor(images=query_image_cropped, return_tensors="pt")
        out = blip_model.generate(**blip_inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

        return jsonify({'caption': caption})
    except Exception as e:
        print(f"Error during caption generation: {e}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/start_detection', methods=['POST'])
def start_detection():
    if 'target_image' not in request.files or 'query_image' not in request.files or 'query_text' not in request.form:
        return jsonify({'error': 'Missing required form data.'}), 400

    task_id = str(uuid.uuid4())
    
    TASKS[task_id] = {
        "query_image_bytes": request.files['query_image'].read(),
        "target_image_bytes": request.files['target_image'].read(),
        "query_text": request.form['query_text'],
        "roi": {
            "x": int(float(request.form.get('roi_x', 0))),
            "y": int(float(request.form.get('roi_y', 0))),
            "width": int(float(request.form.get('roi_width', 0))),
            "height": int(float(request.form.get('roi_height', 0)))
        },
        "owl_threshold": float(request.form.get('owl_threshold', 0.1)),
        "clip_threshold": float(request.form.get('clip_threshold', 0.2)),
    }
    print(f"TASK {task_id}: Created and stored.", flush=True)
    return jsonify({"task_id": task_id})

@app.route('/stream/<task_id>')
def stream(task_id):
    def generate():
        task = TASKS.get(task_id)
        if not task:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Task not found.'})}\\n\n"
            return

        try:
            query_text = task["query_text"]
            yield f"data: {json.dumps({'type': 'caption', 'caption': query_text})}\\n\n"

            # Prepare images
            query_image = Image.open(io.BytesIO(task["query_image_bytes"])).convert("RGB")
            target_image = Image.open(io.BytesIO(task["target_image_bytes"])).convert("RGB")
            roi = task["roi"]
            if roi["width"] == 0 or roi["height"] == 0: roi["width"], roi["height"] = query_image.size
            query_image_cropped = query_image.crop((roi["x"], roi["y"], roi["x"] + roi["width"], roi["y"] + roi["height"]))

            # OWLv2
            texts = [[query_text]]
            inputs = owl_processor(text=texts, images=target_image, return_tensors="pt")
            outputs = owl_model(**inputs)
            target_sizes = torch.tensor([target_image.size[::-1]])
            results = owl_processor.post_process_grounded_object_detection(outputs, target_sizes=target_sizes, threshold=task["owl_threshold"])
            candidate_boxes = results[0]["boxes"]
            print(f"TASK {task_id}: Found {len(candidate_boxes)} candidates using text: '{query_text}'", flush=True)

            if len(candidate_boxes) == 0:
                yield f"data: {json.dumps({'type': 'status', 'message': 'No candidates found.'})}\\n\n"
            else:
                # Send candidates
                initial_candidates = []
                for i, box in enumerate(candidate_boxes):
                    candidate_img = target_image.crop(box.tolist())
                    buffered = io.BytesIO()
                    candidate_img.save(buffered, format="PNG")
                    encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    initial_candidates.append({'id': i, 'thumbnail': encoded_img})
                yield f"data: {json.dumps({'type': 'candidates', 'candidates': initial_candidates})}\\n\n"

                # CLIP
                query_clip_inputs = clip_processor(images=query_image_cropped, return_tensors="pt")
                query_features = clip_model.get_image_features(**query_clip_inputs)
                query_features /= query_features.norm(p=2, dim=-1, keepdim=True)

                for i, box in enumerate(candidate_boxes):
                    candidate_img = target_image.crop(box.tolist())
                    candidate_clip_inputs = clip_processor(images=candidate_img, return_tensors="pt")
                    candidate_features = clip_model.get_image_features(**candidate_clip_inputs)
                    candidate_features /= candidate_features.norm(p=2, dim=-1, keepdim=True)
                    score = (query_features @ candidate_features.T).squeeze(0).item()
                    yield f"data: {json.dumps({'type': 'update', 'id': i, 'score': round(score, 3)})}\\n\n"
                    time.sleep(0.05)

            yield f"data: {json.dumps({'type': 'done'})}\\n\n"

        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\\n\n"
        finally:
            if task_id in TASKS:
                del TASKS[task_id]
                print(f"TASK {task_id}: Cleaned up.", flush=True)

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
