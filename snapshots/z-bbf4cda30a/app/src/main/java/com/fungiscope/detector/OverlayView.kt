package com.fungiscope.detector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {
    private var detections: List<Detection> = emptyList()
    private var imageWidth = 1
    private var imageHeight = 1
    private val density = resources.displayMetrics.density

    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.5f * density
        strokeCap = Paint.Cap.ROUND
    }
    private val wash = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val chip = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 12f * density
        typeface = android.graphics.Typeface.create("sans-serif-medium", android.graphics.Typeface.NORMAL)
    }

    fun setResults(results: List<Detection>, sourceWidth: Int, sourceHeight: Int) {
        detections = results
        imageWidth = sourceWidth.coerceAtLeast(1)
        imageHeight = sourceHeight.coerceAtLeast(1)
        invalidate()
    }

    fun clear() {
        detections = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty()) return

        val scale = max(width / imageWidth.toFloat(), height / imageHeight.toFloat())
        val offsetX = (width - imageWidth * scale) / 2f
        val offsetY = (height - imageHeight * scale) / 2f

        detections.forEachIndexed { index, detection ->
            val box = RectF(
                detection.box.left * scale + offsetX,
                detection.box.top * scale + offsetY,
                detection.box.right * scale + offsetX,
                detection.box.bottom * scale + offsetY
            )
            val color = if (detection.reviewRecommended) {
                Color.rgb(255, 190, 70)
            } else {
                colorFor(detection.label, index)
            }
            wash.color = Color.argb(25, Color.red(color), Color.green(color), Color.blue(color))
            stroke.color = color
            canvas.drawRoundRect(box, 16f * density, 16f * density, wash)
            drawCorners(canvas, box)
            drawLabel(canvas, box, detection, color)
        }
    }

    private fun drawCorners(canvas: Canvas, box: RectF) {
        val length = (box.width().coerceAtMost(box.height()) * 0.20f).coerceIn(16f * density, 38f * density)
        val radius = 10f * density
        val path = Path().apply {
            moveTo(box.left, box.top + length)
            lineTo(box.left, box.top + radius)
            quadTo(box.left, box.top, box.left + radius, box.top)
            lineTo(box.left + length, box.top)
            moveTo(box.right - length, box.top)
            lineTo(box.right - radius, box.top)
            quadTo(box.right, box.top, box.right, box.top + radius)
            lineTo(box.right, box.top + length)
            moveTo(box.right, box.bottom - length)
            lineTo(box.right, box.bottom - radius)
            quadTo(box.right, box.bottom, box.right - radius, box.bottom)
            lineTo(box.right - length, box.bottom)
            moveTo(box.left + length, box.bottom)
            lineTo(box.left + radius, box.bottom)
            quadTo(box.left, box.bottom, box.left, box.bottom - radius)
            lineTo(box.left, box.bottom - length)
        }
        canvas.drawPath(path, stroke)
    }

    private fun drawLabel(canvas: Canvas, box: RectF, detection: Detection, color: Int) {
        val status = when {
            detection.locallyCorrected -> "  • learned"
            detection.reviewRecommended -> "  • review"
            else -> ""
        }
        val text = "${detection.label}  ${(detection.confidence * 100).toInt()}%$status"
        val horizontal = 10f * density
        val chipHeight = 28f * density
        val chipWidth = label.measureText(text) + horizontal * 2
        val left = box.left.coerceIn(8f * density, width - chipWidth - 8f * density)
        val top = if (box.top > chipHeight + 10f * density) box.top - chipHeight - 5f * density else box.top + 5f * density
        val rect = RectF(left, top, left + chipWidth, top + chipHeight)
        chip.color = Color.argb(225, 12, 28, 20)
        canvas.drawRoundRect(rect, 14f * density, 14f * density, chip)
        chip.color = if (detection.reviewRecommended) Color.rgb(255, 190, 70) else color
        canvas.drawCircle(left + 9f * density, rect.centerY(), 2.5f * density, chip)
        canvas.drawText(text, left + 16f * density, top + 18.5f * density, label)
    }

    private fun colorFor(text: String, index: Int): Int {
        if (text.equals("SRA-EB", true)) return Color.rgb(85, 221, 246)
        if (text.equals("SRA-SB", true)) return Color.rgb(167, 123, 255)
        val palette = intArrayOf(
            Color.rgb(85, 221, 246),
            Color.rgb(167, 123, 255),
            Color.rgb(95, 166, 255),
            Color.rgb(198, 247, 255)
        )
        return palette[(text.hashCode() + index).and(Int.MAX_VALUE) % palette.size]
    }
}
