# BACTERIALSCOPE

BACTERIALSCOPE is an Android Studio camera app for real-time, on-device bacterial sample detection. It uses CameraX and TensorFlow Lite with live bounding boxes, class counts, confidence control, zoom, torch control, and full-quality annotated capture.

## Gallery

Captured scans persist inside the app and appear in the Culture Gallery. The gallery includes:

- full-image viewing with swipe navigation and pinch zoom;
- a selectable thumbnail strip;
- saved capture details and class counts;
- Save to Photos, delete, and CSV summary export;
- compatibility with captures saved by the earlier FungiScope build.

## Post-evaluation learning

After a detected capture, BACTERIALSCOPE opens a post-evaluation panel showing the image and model prediction. Selecting the correct label stores a durable visual example of that detected object. Future detection crops use rotation-tolerant weighted nearest-neighbor voting across the saved examples. Broad class remapping is not used for inference because it can incorrectly collapse visually different samples into one dominant raw class such as `SRA-SB-019`. A newer correction replaces a near-identical older example so an accidental answer does not permanently conflict with later feedback.

The clean YOLO training crops are the durable source of learning memory. If the compact preference cache is lost or stale, it is rebuilt from those crops. Raw labels that have repeatedly been corrected are shown with an amber **REVIEW** marker; visually matched corrections are shown as **LEARNED**. The Culture Gallery includes a Learning Memory control for reviewing counts and resetting active memory without deleting exportable training files.

Version 9 also contains `bacterial_learning_seed.json`, a compact visual-memory seed generated from `BACTERIALSCOPE_Training_20260721_092942.zip`. It preserves 198 confirmed corrections without embedding the original 150 MB of photographs in the APK. New phone corrections receive higher matching weight than the bundled seed, so later feedback can override an older imported example.

## Accuracy post-processing

Version 10 adds an RSSI-style accuracy pipeline. Live results are spatially tracked and must appear in two consecutive analyzed frames before being shown. Confidence-weighted label history resists isolated class spikes, while learned visual corrections receive more voting weight and unreliable raw labels receive less.

Captured photos are analyzed three times: normally, with mild lighting enhancement, and with stronger enhancement. Spatially matching results are fused into one confidence-weighted box and label consensus. Weak detections found by only one pass are rejected, uncertain class disagreements are marked for review, and overlap suppression prevents one object from being reported under multiple bacterial labels.

Version 11 supports portrait and horizontal device rotation. The horizontal camera layout keeps Flash at the upper right, the capture shutter fixed at the middle right, Gallery at the lower right, and the detection controls in a compact bottom panel.

Corrections calibrate results after inference; they do not rewrite the immutable TFLite weights. The detector must still produce a bounding box before the app can visually learn an example. Objects the TFLite model misses completely require exported data and model retraining.

## Exportable YOLO training data

Every time **Teach app** is confirmed, BACTERIALSCOPE saves a clean crop of the evaluated object and its user-corrected YOLO annotation. Open Gallery → Info → **Export Training ZIP** to choose where the phone saves a portable dataset containing:

- `images/train` and `labels/train` in YOLO detection format;
- an automatic validation split once at least five examples are available;
- `data.yaml` with the exact 24-class order used by the app;
- `manifest.csv`, per-sample metadata, and the post-evaluation audit;
- training and TFLite export commands for use in Visual Studio Code.

Samples marked **No bacteria / false detection** are exported as negative images with intentionally empty YOLO label files. Review model-generated bounding boxes before retraining.

## Add the trained model

1. Export the matching bacterial object detector as TFLite.
2. Copy it to `app/src/main/assets/bacterial_model.tflite`.
3. Keep `app/src/main/assets/bacterial_labels.txt` in the exact embedded training order. The supplied build contains 24 labels and the matching model.
4. Build and run. The header shows the input/output shape after the model loads.

Without `bacterial_model.tflite`, the camera, capture, and gallery remain usable while detection clearly reports that the model is missing.

## Build

Open this folder in Android Studio and run the `app` configuration, or use `gradlew.bat assembleDebug`.
