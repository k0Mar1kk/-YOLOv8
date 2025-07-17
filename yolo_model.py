from ultralytics import YOLO
import os
import shutil
import cv2
import numpy as np

model = YOLO('yolov8n.pt')
RESULTS_PATH = "/results"

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]

def process_image(image_path):
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    image = cv2.imread(image_path)
    image_orig = image.copy()
    h_or, w_or = image.shape[:2]
    image = cv2.resize(image, (640, 640))
    results = model(image)[0]
    
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    masks = results.masks.data.cpu().numpy()

    for i, mask in enumerate(masks):
        color = colors[int(classes[i]) % len(colors)]
        mask_resized = cv2.resize(mask, (w_or, h_or))
        color_mask = np.zeros((h_or, w_or, 3), dtype=np.uint8)
        color_mask[mask_resized > 0] = color
        mask_filename = os.path.join('results', f"{classes_names[classes[i]]}_{i}.png")
        cv2.imwrite(mask_filename, color_mask)
        image_orig = cv2.addWeighted(image_orig, 1.0, color_mask, 0.5, 0)

    new_image_path = os.path.join('results', os.path.splitext(os.path.basename(image_path))[0] + '_segmented' + os.path.splitext(image_path)[1])
    cv2.imwrite(new_image_path, image_orig)
    print(f"Segmented image saved to {new_image_path}")

def predict_image(image_path):
    results = model(image_path)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    output_path = os.path.join(RESULTS_PATH, os.path.basename(image_path))
    
    objects_count = 0
    for r in results:
        r.save(filename=output_path)
        objects_count += len(r.boxes)
    
    return os.path.basename(output_path), objects_count

def predict_video(video_path):
    os.makedirs(RESULTS_PATH, exist_ok=True)

    results = model.predict(
        source=video_path,
        save=True,
        project=RESULTS_PATH,
        name="predict",
        exist_ok=True
    )

    video_name = os.path.basename(video_path)
    saved_video_path = os.path.join(RESULTS_PATH, "predict", video_name)
    final_path = os.path.join(RESULTS_PATH, video_name)
    
    objects_count = 0
    for r in results:
        objects_count += len(r.boxes)
    
    if os.path.exists(saved_video_path):
        shutil.move(saved_video_path, final_path)

    return os.path.basename(final_path), objects_count
