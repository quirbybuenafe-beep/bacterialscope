BACTERIALSCOPE expects an object-detection model at:
app/src/main/assets/bacterial_model.tflite

Supported model families:
- YOLO TFLite output [1, channels, boxes] or [1, boxes, channels]
- TFLite Detection_PostProcess output with boxes, classes, scores, count

The label order in bacterial_labels.txt exactly matches the 24 class channels
embedded in the current bacterial_model.tflite (model version 2). Do not reorder the file.

Do not rename an unrelated detector model or invent labels; use the matching trained model.
