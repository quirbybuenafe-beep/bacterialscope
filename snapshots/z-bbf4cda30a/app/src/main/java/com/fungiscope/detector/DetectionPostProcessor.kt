package com.fungiscope.detector

import android.graphics.RectF
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Combines detections from several lighting/contrast passes of the same image.
 * Geometry is matched independently of the predicted class, then the class is
 * selected by confidence-weighted consensus. This is intentionally used only
 * for full-resolution captures; live preview uses [DetectionStabilizer].
 */
class DetectionPostProcessor {
    private data class Candidate(val passIndex: Int, val detection: Detection)

    private data class Cluster(
        val candidates: MutableList<Candidate> = mutableListOf()
    ) {
        fun containsPass(passIndex: Int): Boolean = candidates.any { it.passIndex == passIndex }

        fun representativeBox(): RectF {
            var weightSum = 0f
            var left = 0f
            var top = 0f
            var right = 0f
            var bottom = 0f
            candidates.forEach { candidate ->
                val weight = candidate.detection.confidence.coerceAtLeast(0.05f).pow(2)
                weightSum += weight
                left += candidate.detection.box.left * weight
                top += candidate.detection.box.top * weight
                right += candidate.detection.box.right * weight
                bottom += candidate.detection.box.bottom * weight
            }
            return RectF(left / weightSum, top / weightSum, right / weightSum, bottom / weightSum)
        }
    }

    fun fusePasses(
        passes: List<List<Detection>>,
        imageWidth: Int,
        imageHeight: Int
    ): List<Detection> {
        if (passes.isEmpty()) return emptyList()
        val diagonal = hypot(imageWidth.toFloat(), imageHeight.toFloat()).coerceAtLeast(1f)
        val candidates = passes.flatMapIndexed { passIndex, detections ->
            detections.map { Candidate(passIndex, it) }
        }.sortedByDescending { it.detection.confidence }
        if (candidates.isEmpty()) return emptyList()

        val clusters = mutableListOf<Cluster>()
        candidates.forEach { candidate ->
            val bestCluster = clusters
                .asSequence()
                .filterNot { it.containsPass(candidate.passIndex) }
                .map { it to matchScore(it.representativeBox(), candidate.detection.box, diagonal) }
                .filter { it.second >= 0f }
                .maxByOrNull { it.second }
                ?.first

            if (bestCluster == null) {
                clusters += Cluster(mutableListOf(candidate))
            } else {
                bestCluster.candidates += candidate
            }
        }

        return clusters.mapNotNull { cluster ->
            clusterToDetection(cluster, passes.size)
        }.sortedByDescending(Detection::confidence).take(MAX_DETECTIONS)
    }

    private fun clusterToDetection(cluster: Cluster, totalPasses: Int): Detection? {
        val support = cluster.candidates.map(Candidate::passIndex).distinct().size
        val strongest = cluster.candidates.maxOf { it.detection.confidence }
        if (support == 1 && strongest < STRONG_SINGLE_PASS_CONFIDENCE) return null

        val labelScores = linkedMapOf<String, Float>()
        var confidenceWeight = 0f
        var weightedConfidence = 0f
        cluster.candidates.forEach { candidate ->
            val confidence = candidate.detection.confidence.coerceIn(0f, 1f)
            val weight = confidence.coerceAtLeast(0.05f).pow(2)
            labelScores[candidate.detection.label] =
                labelScores.getOrDefault(candidate.detection.label, 0f) + weight
            confidenceWeight += weight
            weightedConfidence += confidence * weight
        }

        val chosenLabel = labelScores.maxByOrNull { it.value }?.key ?: return null
        val labelAgreement = labelScores.getValue(chosenLabel) /
            labelScores.values.sum().coerceAtLeast(0.0001f)
        val passAgreement = support.toFloat() / totalPasses.coerceAtLeast(1)
        val consensusConfidence = (weightedConfidence / confidenceWeight)
            .times(0.92f + 0.08f * passAgreement)
            .times(0.92f + 0.08f * labelAgreement)
            .coerceIn(0f, 1f)

        return Detection(
            label = chosenLabel,
            confidence = consensusConfidence,
            box = cluster.representativeBox(),
            reviewRecommended = support == 1 || labelAgreement < MIN_LABEL_AGREEMENT
        )
    }

    private fun matchScore(a: RectF, b: RectF, diagonal: Float): Float {
        val iou = intersectionOverUnion(a, b)
        val centerDistance = hypot(a.centerX() - b.centerX(), a.centerY() - b.centerY()) / diagonal
        val areaA = max(1f, a.width() * a.height())
        val areaB = max(1f, b.width() * b.height())
        val sizeSimilarity = min(areaA, areaB) / max(areaA, areaB)
        val centerMatch = centerDistance <= MAX_CENTER_DISTANCE_RATIO &&
            sizeSimilarity >= MIN_SIZE_SIMILARITY
        if (iou < MIN_MATCH_IOU && !centerMatch) return -1f
        val proximity = (1f - centerDistance / MAX_CENTER_DISTANCE_RATIO).coerceIn(0f, 1f)
        return iou + proximity * 0.25f + sizeSimilarity * 0.10f
    }

    private fun intersectionOverUnion(a: RectF, b: RectF): Float {
        val intersectionWidth = max(0f, min(a.right, b.right) - max(a.left, b.left))
        val intersectionHeight = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val intersection = intersectionWidth * intersectionHeight
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private companion object {
        const val MIN_MATCH_IOU = 0.28f
        const val MAX_CENTER_DISTANCE_RATIO = 0.065f
        const val MIN_SIZE_SIMILARITY = 0.48f
        const val STRONG_SINGLE_PASS_CONFIDENCE = 0.72f
        const val MIN_LABEL_AGREEMENT = 0.66f
        const val MAX_DETECTIONS = 100
    }
}
