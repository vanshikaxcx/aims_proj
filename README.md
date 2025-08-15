# Dense Scene Localization via Natural Language Queries

A comprehensive deep learning project that enables pixel-accurate object localization in complex scenes using natural language descriptions. This project represents a major learning journey into computer vision and multimodal AI systems.

---

## Problem Statement

Traditional object detection models are limited to fixed vocabularies (e.g., 80 COCO classes) and cannot understand complex relational queries. This project aims to build a system that can:
- Understand natural language queries like "the vendor selling vegetables to a customer"
- Locate specific objects and people in dense, crowded scenes
- Return pixel-accurate segmentation masks rather than just bounding boxes
- Handle open-vocabulary queries beyond predefined classes

---

## Approach Overview

### Current Working Solution: OWLv2 + SAM Pipeline
- **OWLv2**: Open-vocabulary object detection for initial localization
- **SAM**: Segment Anything Model for pixel-accurate mask refinement
- **Performance**: 85% success rate, under 450ms inference time per image
- **Dataset**: Fine-tuned on COCO 2017

### Experimental Approaches Explored
1. **YOLO Family** (YOLOv5, YOLOv8) – Limited to closed vocabulary detection
2. **Grounding DINO** – Good accuracy but slow inference (~0.9 FPS)
3. **OWLv2 + Flickr30k** – Dataset preprocessing challenges with complex annotation formats
4. **Faster R-CNN + MattNet** – Advanced relationship modeling on Visual Genome dataset (in progress)

---

## Key Results

| Model           | Dataset    | mAP / IoU  | Inference Speed | Vocabulary           |
|-----------------|----------- |----------- |---------------- |----------------------|
| YOLOv8          | COCO       | 53.9 mAP   | 35-50 FPS       | Closed (80 classes)  |
| Grounding DINO  | RefCOCO    | 52.5 mAP   | 0.9 FPS         | Open vocabulary      |
| OWLv2           | COCO       | 31.2 mAP   | 25 FPS          | Open vocabulary      |
| **OWLv2 + SAM** | **COCO**   | **71 IoU** | **2.2 FPS**     | **Open vocabulary**  |

---

## Installation & Setup

Clone the repository
git clone <repository-url>
cd dense-scene-localization

Create and activate a conda environment
conda create -n scene_loc python=3.8 -y
conda activate scene_loc

Install required dependencies
pip install torch torchvision transformers
pip install segment-anything
pip install opencv-python matplotlib

Download SAM model checkpoint (approx 2.4GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

text

---

## Usage

After setting up, you can initialize the pipeline and run inference with your images and text queries.

from pipeline import OWLv2SAMPipeline

pipeline = OWLv2SAMPipeline()

image = load_image("marketplace.jpg")
query = "the vendor selling vegetables"
masks, boxes, scores = pipeline.predict(image, [query])

pipeline.visualize(image, masks, query)

text

---

## Project Timeline

- **Weeks 1-8**: Explored YOLO family models on COCO dataset; learned object detection basics and debugging
- **Weeks 9-20**: Developed and trained Grounding DINO with COCO and RefCOCO datasets; faced installation and training challenges
- **Weeks 21-27**: Worked with OWLv2 on Flickr30k Entities; overcame XML data preprocessing struggles
- **Weeks 28-44**: Finalized OWLv2 + SAM pipeline on COCO for high accuracy segmentation
- **Weeks 45-Present**: Researching Faster R-CNN + MattNet for relationship understanding on Visual Genome; training in progress

---

## Technical Challenges Overcome

### Memory Management
- Frequently encountered CUDA out-of-memory errors with large models.
- Solutions included gradient accumulation, mixed precision training, and sequential model loading to manage GPU memory efficiently.

### Dataset Integration
- Worked with multiple dataset annotation formats: COCO JSON, Flickr30k XML, and Visual Genome scene graphs.
- Built custom preprocessing pipelines and applied extensive error handling to handle real-world dataset irregularities.

### Coordinate System Alignment
- Resolved inconsistencies between normalized coordinates (used by OWLv2) and absolute pixel coordinates (required by SAM).
- Developed systematic coordinate transformations and debugging visualizations for precise alignment.

---

## Current Limitations

1. The current pipeline struggles with complex relational queries involving multiple interacting objects.
2. Achieving high-quality segmentation requires approximately 450ms per image, limiting real-time use.
3. The models require substantial hardware resources, ideally a GPU with 12GB or more VRAM.

---

## Future Work

- Complete the training and evaluation of the Faster R-CNN + MattNet model for explicit relationship modeling.
- Implement model quantization and optimization for deployment on mobile and edge devices.
- Integrate multilingual query support to enhance accessibility.
- Extend capabilities to real-time video processing and 3D scene understanding.

---

## Learning Outcomes

This project provided hands-on experience in:
- Modern computer vision techniques: object detection, segmentation, and attention mechanisms.
- Multimodal AI: combining vision and language through open-vocabulary detection models.
- MLOps essentials: dataset curation, model training, debugging, evaluation, and deployment.
- Research skills: reading scientific papers, implementing complex models, designing experiments, and interpreting results.

---

## Dependencies

- PyTorch 2.2 or later
- Transformers 4.21 or later
- OpenCV 4.8 or later
- Segment Anything Model (SAM)
- CUDA 11.8 or later

---
