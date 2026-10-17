package com.fungiscope.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class Detector(
    context: Context,
    modelAssetName: String,
    labelAssetName: String,
    private var scoreThreshold: Float = 0.40f,
    private var iouThreshold: Float = 0.45f
) {
    private val interpreter: Interpreter
    private val labels: List<String>
    private val inputWidth: Int
    private val inputHeight: Int
    private val isNchw: Boolean
    private val inputType: DataType
    private val inputScale: Float
    private val inputZeroPoint: Int

    init {
        val options = Interpreter.Options().apply {
            setNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(4, 6))
        }
        interpreter = Interpreter(loadModelFile(context, modelAssetName), options)
        labels = loadLabels(context, labelAssetName)

        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        require(inputShape.size == 4) {
            "Model input must be 4D; found ${inputShape.joinToString("×")}."
        }

        inputType = inputTensor.dataType()
        val quantization = inputTensor.quantizationParams()
        inputScale = quantization.scale
        inputZeroPoint = quantization.zeroPoint

        isNchw = inputShape[1] == 3
        if (isNchw) {
            inputHeight = inputShape[2]
            inputWidth = inputShape[3]
        } else {
            inputHeight = inputShape[1]
            inputWidth = inputShape[2]
        }

        val channels = if (isNchw) inputShape[1] else inputShape[3]
        require(channels == 3) { "The detector must accept 3-channel RGB input." }
        require(labels.isNotEmpty()) { "bacterial_labels.txt is empty." }
    }

    fun summary(): String {
        val input = interpreter.getInputTensor(0)
        val outputs = (0 until interpreter.outputTensorCount).joinToString(" • ") { index ->
            "out$index ${interpreter.getOutputTensor(index).shape().joinToString("×")}" 
        }
        return "${input.shape().joinToString("×")} ${input.dataType()} • ${labels.size} classes • $outputs"
    }

    @Synchronized
    fun detect(source: Bitmap): List<Detection> {
        val input = bitmapToInput(source)
        return if (isDetectionPostProcessModel()) {
            detectPostProcess(input, source.width, source.height)
        } else {
            detectYolo(input, source.width, source.height)
        }
    }

    fun setThreshold(score: Float) {
        scoreThreshold = score.coerceIn(0.05f, 0.95f)
    }

    fun close() = interpreter.close()

    private fun loadModelFile(context: Context, assetName: String): MappedByteBuffer {
        val descriptor = context.assets.openFd(assetName)
        FileInputStream(descriptor.fileDescriptor).use { stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY,
                descriptor.startOffset,
                descriptor.declaredLength
            )
        }
    }

    private fun loadLabels(context: Context, assetName: String): List<String> {
        return context.assets.open(assetName).bufferedReader().useLines { lines ->
            lines.map(String::trim).filter(String::isNotEmpty).toList()
        }
    }

    private fun bitmapToInput(source: Bitmap): ByteBuffer {
        val bytesPerChannel = when (inputType) {
            DataType.FLOAT32 -> 4
            DataType.UINT8, DataType.INT8 -> 1
            else -> error("Unsupported input type: $inputType")
        }
        val buffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * bytesPerChannel)
            .order(ByteOrder.nativeOrder())
        val resized = Bitmap.createScaledBitmap(source, inputWidth, inputHeight, true)
        val pixels = IntArray(inputWidth * inputHeight)
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        if (isNchw) {
            for (channel in 0..2) {
                pixels.forEach { pixel ->
                    val value = when (channel) {
                        0 -> pixel shr 16 and 0xFF
                        1 -> pixel shr 8 and 0xFF
                        else -> pixel and 0xFF
                    }
                    putChannel(buffer, value)
                }
            }
        } else {
            pixels.forEach { pixel ->
                putChannel(buffer, pixel shr 16 and 0xFF)
                putChannel(buffer, pixel shr 8 and 0xFF)
                putChannel(buffer, pixel and 0xFF)
            }
        }
        if (resized !== source) resized.recycle()
        buffer.rewind()
        return buffer
    }

    private fun putChannel(buffer: ByteBuffer, value: Int) {
        when (inputType) {
            DataType.FLOAT32 -> buffer.putFloat(value / 255f)
            DataType.UINT8 -> {
                val quantized = if (inputScale in 0.000001f..0.999999f) {
                    ((value / 255f) / inputScale + inputZeroPoint).roundToInt()
                } else value
                buffer.put(quantized.coerceIn(0, 255).toByte())
            }
            DataType.INT8 -> {
                val quantized = if (inputScale > 0f) {
                    ((value / 255f) / inputScale + inputZeroPoint).roundToInt()
                } else value - 128
                buffer.put(quantized.coerceIn(-128, 127).toByte())
            }
            else -> error("Unsupported input type: $inputType")
        }
    }

    private fun isDetectionPostProcessModel(): Boolean {
        if (interpreter.outputTensorCount < 4) return false
        val shape = interpreter.getOutputTensor(0).shape()
        return shape.size == 3 && shape.last() == 4
    }

    private fun detectPostProcess(input: ByteBuffer, sourceWidth: Int, sourceHeight: Int): List<Detection> {
        val maxDetections = interpreter.getOutputTensor(0).shape()[1]
        val boxes = Array(1) { Array(maxDetections) { FloatArray(4) } }
        val classes = Array(1) { FloatArray(maxDetections) }
        val scores = Array(1) { FloatArray(maxDetections) }
        val count = FloatArray(1)

        interpreter.runForMultipleInputsOutputs(
            arrayOf(input),
            mapOf(0 to boxes, 1 to classes, 2 to scores, 3 to count)
        )

        val detections = mutableListOf<Detection>()
        val resultCount = count[0].roundToInt().coerceIn(0, maxDetections)
        repeat(resultCount) { index ->
            val score = scores[0][index]
            if (score < scoreThreshold) return@repeat
            val classId = classes[0][index].roundToInt().coerceAtLeast(0)
            val top = boxes[0][index][0].coerceIn(0f, 1f) * sourceHeight
            val left = boxes[0][index][1].coerceIn(0f, 1f) * sourceWidth
            val bottom = boxes[0][index][2].coerceIn(0f, 1f) * sourceHeight
            val right = boxes[0][index][3].coerceIn(0f, 1f) * sourceWidth
            if (right > left && bottom > top) {
                detections += Detection(
                    label = labels.getOrElse(classId) { "Class ${classId + 1}" },
                    confidence = score,
                    box = RectF(left, top, right, bottom)
                )
            }
        }
        return nonMaxSuppression(detections)
    }

    private fun detectYolo(input: ByteBuffer, sourceWidth: Int, sourceHeight: Int): List<Detection> {
        val matrix = readYoloOutput(input)
        if (matrix.isEmpty() || matrix[0].isEmpty()) return emptyList()

        val channelsFirst = matrix.size <= matrix[0].size
        val boxCount = if (channelsFirst) matrix[0].size else matrix.size
        val channelCount = if (channelsFirst) matrix.size else matrix[0].size
        require(channelCount >= 5) { "YOLO output needs at least five channels." }

        val detections = mutableListOf<Detection>()
        repeat(boxCount) { boxIndex ->
            fun value(channel: Int): Float =
                if (channelsFirst) matrix[channel][boxIndex] else matrix[boxIndex][channel]

            val (classId, score) = bestClass(channelCount, ::value)
            if (score < scoreThreshold) return@repeat

            val rawX = value(0)
            val rawY = value(1)
            val rawW = value(2)
            val rawH = value(3)
            val normalized = rawX in -0.2f..1.2f && rawY in -0.2f..1.2f &&
                rawW in 0f..1.2f && rawH in 0f..1.2f
            val centerX = if (normalized) rawX * inputWidth else rawX
            val centerY = if (normalized) rawY * inputHeight else rawY
            val width = if (normalized) rawW * inputWidth else rawW
            val height = if (normalized) rawH * inputHeight else rawH

            val left = ((centerX - width / 2f) / inputWidth * sourceWidth).coerceIn(0f, sourceWidth.toFloat())
            val top = ((centerY - height / 2f) / inputHeight * sourceHeight).coerceIn(0f, sourceHeight.toFloat())
            val right = ((centerX + width / 2f) / inputWidth * sourceWidth).coerceIn(0f, sourceWidth.toFloat())
            val bottom = ((centerY + height / 2f) / inputHeight * sourceHeight).coerceIn(0f, sourceHeight.toFloat())
            if (right > left && bottom > top) {
                detections += Detection(
                    labels.getOrElse(classId) { "Class ${classId + 1}" },
                    score,
                    RectF(left, top, right, bottom)
                )
            }
        }
        return nonMaxSuppression(detections)
    }

    private fun bestClass(channelCount: Int, value: (Int) -> Float): Pair<Int, Float> {
        if (channelCount == 5) return 0 to value(4)

        // Prefer the layout whose class count matches bacterial_labels.txt.
        if (channelCount - 4 == labels.size) {
            return (4 until channelCount)
                .map { it - 4 to value(it) }
                .maxBy { it.second }
        }
        if (channelCount - 5 == labels.size) {
            val objectness = value(4)
            return (5 until channelCount)
                .map { it - 5 to objectness * value(it) }
                .maxBy { it.second }
        }

        return (4 until channelCount)
            .map { it - 4 to value(it) }
            .maxBy { it.second }
    }

    private fun readYoloOutput(input: ByteBuffer): Array<FloatArray> {
        val tensor = interpreter.getOutputTensor(0)
        val shape = tensor.shape()
        require(shape.size == 3) {
            "Unsupported output ${shape.joinToString("×")}; expected a YOLO or Detection_PostProcess model."
        }
        val rows = shape[1]
        val columns = shape[2]
        val matrix = Array(rows) { FloatArray(columns) }

        when (tensor.dataType()) {
            DataType.FLOAT32 -> {
                val output = Array(1) { Array(rows) { FloatArray(columns) } }
                interpreter.run(input, output)
                repeat(rows) { row ->
                    repeat(columns) { column -> matrix[row][column] = output[0][row][column] }
                }
            }
            DataType.UINT8, DataType.INT8 -> {
                val output = Array(1) { Array(rows) { ByteArray(columns) } }
                interpreter.run(input, output)
                val quantization = tensor.quantizationParams()
                repeat(rows) { row ->
                    repeat(columns) { column ->
                        val raw = if (tensor.dataType() == DataType.UINT8) {
                            output[0][row][column].toInt() and 0xFF
                        } else output[0][row][column].toInt()
                        matrix[row][column] = (raw - quantization.zeroPoint) * quantization.scale
                    }
                }
            }
            else -> error("Unsupported output type: ${tensor.dataType()}")
        }
        return matrix
    }

    private fun nonMaxSuppression(input: List<Detection>): List<Detection> {
        val remaining = input.sortedByDescending(Detection::confidence).toMutableList()
        val selected = mutableListOf<Detection>()
        while (remaining.isNotEmpty() && selected.size < 100) {
            val best = remaining.removeAt(0)
            selected += best
            remaining.removeAll { other ->
                intersectionOverUnion(best.box, other.box) > iouThreshold
            }
        }
        return selected
    }

    private fun intersectionOverUnion(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union <= 0f) 0f else intersection / union
    }
}
