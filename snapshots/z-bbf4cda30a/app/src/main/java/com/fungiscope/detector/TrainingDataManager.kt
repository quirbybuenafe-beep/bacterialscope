package com.fungiscope.detector

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.math.ceil
import kotlin.math.floor

/**
 * Saves user-confirmed detector results as portable YOLO training samples.
 * The saved image is a clean crop around the evaluated detection, never the
 * gallery image with boxes drawn over it.
 */
class TrainingDataManager(
    private val context: Context
) {
    data class ExportSummary(
        val samples: Int,
        val positiveSamples: Int,
        val negativeSamples: Int
    )

    fun saveConfirmedExample(
        sourceBitmap: Bitmap,
        prediction: Detection,
        correctLabel: String,
        labels: List<String>,
        sourceImageName: String
    ) {
        val classIndex = if (correctLabel == LocalLearningEngine.NO_BACTERIA) {
            null
        } else {
            labels.indexOf(correctLabel).takeIf { it >= 0 }
                ?: error("Unknown training label: $correctLabel")
        }

        val sourceBox = prediction.box
        val horizontalPadding = sourceBox.width() * CROP_PADDING
        val verticalPadding = sourceBox.height() * CROP_PADDING
        val cropLeft = floor(sourceBox.left - horizontalPadding).toInt().coerceIn(0, sourceBitmap.width - 1)
        val cropTop = floor(sourceBox.top - verticalPadding).toInt().coerceIn(0, sourceBitmap.height - 1)
        val cropRight = ceil(sourceBox.right + horizontalPadding).toInt().coerceIn(cropLeft + 1, sourceBitmap.width)
        val cropBottom = ceil(sourceBox.bottom + verticalPadding).toInt().coerceIn(cropTop + 1, sourceBitmap.height)
        val cropWidth = cropRight - cropLeft
        val cropHeight = cropBottom - cropTop
        require(cropWidth > 1 && cropHeight > 1) { "Detection crop is too small" }

        val sampleId = buildSampleId(sourceImageName)
        val imageFile = File(imagesDirectory().apply { mkdirs() }, "$sampleId.jpg")
        val labelFile = File(labelsDirectory().apply { mkdirs() }, "$sampleId.txt")
        val metadataFile = File(metadataDirectory().apply { mkdirs() }, "$sampleId.json")

        val crop = Bitmap.createBitmap(sourceBitmap, cropLeft, cropTop, cropWidth, cropHeight)
        try {
            FileOutputStream(imageFile).use { output ->
                check(crop.compress(Bitmap.CompressFormat.JPEG, 96, output)) {
                    "Could not encode training image"
                }
            }
        } finally {
            if (crop !== sourceBitmap && !crop.isRecycled) crop.recycle()
        }

        if (classIndex == null) {
            labelFile.writeText("")
        } else {
            val normalizedCenterX = ((sourceBox.centerX() - cropLeft) / cropWidth).coerceIn(0f, 1f)
            val normalizedCenterY = ((sourceBox.centerY() - cropTop) / cropHeight).coerceIn(0f, 1f)
            val normalizedWidth = (sourceBox.width() / cropWidth).coerceIn(0f, 1f)
            val normalizedHeight = (sourceBox.height() / cropHeight).coerceIn(0f, 1f)
            labelFile.writeText(
                String.format(
                    Locale.US,
                    "%d %.6f %.6f %.6f %.6f\n",
                    classIndex,
                    normalizedCenterX,
                    normalizedCenterY,
                    normalizedWidth,
                    normalizedHeight
                )
            )
        }

        val createdEpoch = System.currentTimeMillis()
        metadataFile.writeText(
            JSONObject()
                .put("sample", sampleId)
                .put("source_image", sourceImageName)
                .put("captured_at", SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date(createdEpoch)))
                .put("created_epoch", createdEpoch)
                .put("predicted_label", prediction.label)
                .put("correct_label", correctLabel)
                .put("confidence", prediction.confidence.toDouble())
                .put("negative_sample", classIndex == null)
                .toString(2)
        )
    }

    fun sampleCount(): Int = trainingSamples().size

    fun exportZip(output: OutputStream, labels: List<String>): ExportSummary {
        val samples = trainingSamples()
        require(samples.isNotEmpty()) { "No confirmed training samples yet" }

        var positiveSamples = 0
        var negativeSamples = 0
        val useValidationSplit = samples.size >= MINIMUM_SPLIT_SIZE

        ZipOutputStream(output.buffered()).use { zip ->
            samples.forEachIndexed { index, imageFile ->
                val sampleId = imageFile.nameWithoutExtension
                val labelFile = File(labelsDirectory(), "$sampleId.txt")
                val metadataFile = File(metadataDirectory(), "$sampleId.json")
                val isValidation = useValidationSplit && index % VALIDATION_INTERVAL == 0
                val split = if (isValidation) "val" else "train"

                addFile(zip, imageFile, "images/$split/${imageFile.name}")
                addFile(zip, labelFile, "labels/$split/${labelFile.name}")
                if (labelFile.length() == 0L) negativeSamples++ else positiveSamples++
                if (metadataFile.exists()) addFile(zip, metadataFile, "metadata/${metadataFile.name}")
            }

            addText(zip, "classes.txt", labels.joinToString("\n", postfix = "\n"))
            addText(zip, "data.yaml", buildDataYaml(labels, useValidationSplit))
            addText(zip, "manifest.csv", buildManifest(samples))
            addText(zip, "README_TRAINING.txt", buildReadme(samples.size, useValidationSplit))

            val evaluationLog = File(
                context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: context.filesDir,
                "BacterialScope_Learning/post_evaluations.csv"
            )
            if (evaluationLog.exists()) {
                addFile(zip, evaluationLog, "post_evaluations.csv")
            }
        }

        return ExportSummary(samples.size, positiveSamples, negativeSamples)
    }

    private fun trainingSamples(): List<File> {
        return imagesDirectory().listFiles { file ->
            file.isFile && file.extension.equals("jpg", ignoreCase = true) &&
                File(labelsDirectory(), "${file.nameWithoutExtension}.txt").exists()
        }?.sortedBy(File::getName).orEmpty()
    }

    private fun buildDataYaml(labels: List<String>, useValidationSplit: Boolean): String {
        val validationPath = if (useValidationSplit) "images/val" else "images/train"
        return buildString {
            appendLine("path: .")
            appendLine("train: images/train")
            appendLine("val: $validationPath")
            appendLine("nc: ${labels.size}")
            appendLine("names:")
            labels.forEachIndexed { index, label ->
                appendLine("  $index: '${label.replace("'", "''")}'")
            }
        }
    }

    private fun buildManifest(samples: List<File>): String = buildString {
        appendLine("sample,source_image,captured_at,predicted_label,correct_label,confidence,negative_sample")
        samples.forEach { imageFile ->
            val metadataFile = File(metadataDirectory(), "${imageFile.nameWithoutExtension}.json")
            val metadata = runCatching { JSONObject(metadataFile.readText()) }.getOrNull()
            val values = listOf(
                imageFile.nameWithoutExtension,
                metadata?.optString("source_image").orEmpty(),
                metadata?.optString("captured_at").orEmpty(),
                metadata?.optString("predicted_label").orEmpty(),
                metadata?.optString("correct_label").orEmpty(),
                metadata?.optDouble("confidence")?.toString().orEmpty(),
                metadata?.optBoolean("negative_sample")?.toString().orEmpty()
            )
            appendLine(values.joinToString(",") { csvEscape(it) })
        }
    }

    private fun buildReadme(sampleCount: Int, useValidationSplit: Boolean): String = """
        BACTERIALSCOPE YOLO TRAINING EXPORT

        Confirmed samples: $sampleCount

        CONTENTS
        - images/train and labels/train: new YOLO training samples
        - images/val and labels/val: automatic 20% validation split when at least $MINIMUM_SPLIT_SIZE samples exist
        - data.yaml: Ultralytics dataset configuration with the app's exact class order
        - manifest.csv: prediction and user-correction audit
        - metadata: one JSON record per sample
        - post_evaluations.csv: complete learning audit, when available

        USE IN VISUAL STUDIO CODE
        1. Extract this ZIP into your YOLO project.
        2. Review every crop and bounding box before training. The class was confirmed by the user,
           but the starting box still came from the current model.
        3. Merge images/ and labels/ into your main dataset, or train this dataset with:
              yolo detect train model=best.pt data=data.yaml epochs=100 imgsz=640
        4. Export the improved model with:
              yolo export model=runs/detect/train/weights/best.pt format=tflite imgsz=640

        Empty label files are intentional negative samples marked as "No bacteria / false detection".
        ${if (useValidationSplit) "A separate validation split was included." else "There were fewer than $MINIMUM_SPLIT_SIZE samples, so data.yaml temporarily uses the training set for validation. Gather more samples and create a separate validation split before judging model accuracy."}
    """.trimIndent() + "\n"

    private fun addFile(zip: ZipOutputStream, file: File, path: String) {
        zip.putNextEntry(ZipEntry(path))
        file.inputStream().buffered().use { input -> input.copyTo(zip) }
        zip.closeEntry()
    }

    private fun addText(zip: ZipOutputStream, path: String, text: String) {
        zip.putNextEntry(ZipEntry(path))
        zip.write(text.toByteArray(Charsets.UTF_8))
        zip.closeEntry()
    }

    private fun buildSampleId(sourceImageName: String): String {
        val safeSource = sourceImageName.substringBeforeLast('.').replace(Regex("[^A-Za-z0-9_-]"), "_")
        return "${safeSource}_${System.currentTimeMillis()}"
    }

    private fun csvEscape(value: String): String = "\"${value.replace("\"", "\"\"")}\""

    private fun rootDirectory(): File = File(
        context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: context.filesDir,
        "BacterialScope_Training"
    )

    private fun imagesDirectory(): File = File(rootDirectory(), "images")
    private fun labelsDirectory(): File = File(rootDirectory(), "labels")
    private fun metadataDirectory(): File = File(rootDirectory(), "metadata")

    companion object {
        private const val CROP_PADDING = 0.20f
        private const val MINIMUM_SPLIT_SIZE = 5
        private const val VALIDATION_INTERVAL = 5
    }
}
