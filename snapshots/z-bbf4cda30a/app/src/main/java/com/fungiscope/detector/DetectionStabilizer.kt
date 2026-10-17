package com.fungiscope.detector

import android.graphics.RectF
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Tracks live detections across frames. A result must be seen twice before it
 * is shown, and label votes decay over time so isolated class spikes do not
 * replace a stable prediction.
 */
class DetectionStabilizer {
    private class Track(
        var box: RectF,
        var confidence: Float,
        var hits: Int,
        var missingFrames: Int,
        val labelScores: MutableMap<String, Float>,
        val learnedScores: MutableMap<String, Float>,
        val reviewScores: MutableMap<String, Float>
    )

    private val tracks = mutableListOf<Track>()

    @Synchronized
    fun update(detections: List<Detection>, imageWidth: Int, imageHeight: Int): List<Detection> {
        val diagonal = hypot(imageWidth.toFloat(), imageHeight.toFloat()).coerceAtLeast(1f)
        tracks.forEach { it.missingFrames++ }
        val usedTracks = mutableSetOf<Track>()

        detections.sortedByDescending(Detection::confidence).forEach { detection ->
            val match = tracks.asSequence()
                .filterNot { it in usedTracks }
                .map {
                    it to matchScore(
                        it.box,
                        detection.box,
                        diagonal,
                        dominantLabel(it),
                        detection.label
                    )
                }
                .filter { it.second >= 0f }
                .maxByOrNull { it.second }
                ?.first

            if (match == null) {
                val newTrack = newTrack(detection)
                tracks += newTrack
                usedTracks += newTrack
            } else {
                updateTrack(match, detection)
                usedTracks += match
            }
        }

        tracks.removeAll { it.missingFrames > MAX_MISSING_FRAMES }
        return tracks.asSequence()
            .filter { it.missingFrames == 0 && it.hits >= CONFIRM_FRAMES }
            .map { trackToDetection(it) }
            .sortedByDescending(Detection::confidence)
            .take(MAX_DETECTIONS)
            .toList()
    }

    @Synchronized
    fun clear() = tracks.clear()

    private fun newTrack(detection: Detection): Track {
        val reliability = voteReliability(detection)
        return Track(
            box = RectF(detection.box),
            confidence = detection.confidence,
            hits = 1,
            missingFrames = 0,
            labelScores = mutableMapOf(detection.label to detection.confidence * reliability),
            learnedScores = mutableMapOf(
                detection.label to if (detection.locallyCorrected) detection.confidence else 0f
            ),
            reviewScores = mutableMapOf(
                detection.label to if (detection.reviewRecommended) detection.confidence else 0f
            )
        )
    }

    private fun updateTrack(track: Track, detection: Detection) {
        val wasConsecutive = track.missingFrames == 1
        track.box = RectF(
            track.box.left * OLD_BOX_WEIGHT + detection.box.left * NEW_BOX_WEIGHT,
            track.box.top * OLD_BOX_WEIGHT + detection.box.top * NEW_BOX_WEIGHT,
            track.box.right * OLD_BOX_WEIGHT + detection.box.right * NEW_BOX_WEIGHT,
            track.box.bottom * OLD_BOX_WEIGHT + detection.box.bottom * NEW_BOX_WEIGHT
        )
        track.confidence = track.confidence * OLD_CONFIDENCE_WEIGHT +
            detection.confidence * NEW_CONFIDENCE_WEIGHT
        track.hits = if (wasConsecutive) track.hits + 1 else 1
        track.missingFrames = 0
        decay(track.labelScores)
        decay(track.learnedScores)
        decay(track.reviewScores)
        track.labelScores[detection.label] = track.labelScores.getOrDefault(detection.label, 0f) +
            detection.confidence * voteReliability(detection)
        if (detection.locallyCorrected) {
            track.learnedScores[detection.label] =
                track.learnedScores.getOrDefault(detection.label, 0f) + detection.confidence
        }
        if (detection.reviewRecommended) {
            track.reviewScores[detection.label] =
                track.reviewScores.getOrDefault(detection.label, 0f) + detection.confidence
        }
    }

    private fun trackToDetection(track: Track): Detection {
        val rankedLabels = track.labelScores.entries.sortedByDescending { it.value }
        val label = rankedLabels.first().key
        val topScore = rankedLabels.first().value
        val secondScore = rankedLabels.getOrNull(1)?.value ?: 0f
        val uncertain = secondScore > 0f && secondScore / topScore.coerceAtLeast(0.0001f) > 0.72f
        val learnedScore = track.learnedScores.getOrDefault(label, 0f)
        val reviewScore = track.reviewScores.getOrDefault(label, 0f)
        val learned = learnedScore > reviewScore && learnedScore > 0.20f
        return Detection(
            label = label,
            confidence = track.confidence.coerceIn(0f, 1f),
            box = RectF(track.box),
            locallyCorrected = learned,
            reviewRecommended = !learned && (uncertain || reviewScore > 0.20f)
        )
    }

    private fun dominantLabel(track: Track): String =
        track.labelScores.maxByOrNull { it.value }?.key.orEmpty()

    private fun voteReliability(detection: Detection): Float = when {
        detection.locallyCorrected -> 1.35f
        detection.reviewRecommended -> 0.55f
        else -> 1f
    }

    private fun decay(scores: MutableMap<String, Float>) {
        scores.keys.toList().forEach { label ->
            val decayed = scores.getValue(label) * LABEL_VOTE_DECAY
            if (decayed < 0.01f) scores.remove(label) else scores[label] = decayed
        }
    }

    private fun matchScore(
        a: RectF,
        b: RectF,
        diagonal: Float,
        trackLabel: String,
        detectionLabel: String
    ): Float {
        val iou = intersectionOverUnion(a, b)
        val centerDistance = hypot(a.centerX() - b.centerX(), a.centerY() - b.centerY()) / diagonal
        val areaA = max(1f, a.width() * a.height())
        val areaB = max(1f, b.width() * b.height())
        val sizeSimilarity = min(areaA, areaB) / max(areaA, areaB)
        if (iou < MIN_MATCH_IOU &&
            (centerDistance > MAX_CENTER_DISTANCE_RATIO || sizeSimilarity < MIN_SIZE_SIMILARITY)
        ) return -1f
        val proximity = (1f - centerDistance / MAX_CENTER_DISTANCE_RATIO).coerceIn(0f, 1f)
        return iou * 0.70f + proximity * 0.20f + sizeSimilarity * 0.10f +
            if (trackLabel == detectionLabel) 0.05f else 0f
    }

    private fun intersectionOverUnion(a: RectF, b: RectF): Float {
        val intersectionWidth = max(0f, min(a.right, b.right) - max(a.left, b.left))
        val intersectionHeight = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val intersection = intersectionWidth * intersectionHeight
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private companion object {
        const val CONFIRM_FRAMES = 2
        const val MAX_MISSING_FRAMES = 2
        const val MIN_MATCH_IOU = 0.12f
        const val MAX_CENTER_DISTANCE_RATIO = 0.10f
        const val MIN_SIZE_SIMILARITY = 0.42f
        const val OLD_BOX_WEIGHT = 0.55f
        const val NEW_BOX_WEIGHT = 0.45f
        const val OLD_CONFIDENCE_WEIGHT = 0.55f
        const val NEW_CONFIDENCE_WEIGHT = 0.45f
        const val LABEL_VOTE_DECAY = 0.70f
        const val MAX_DETECTIONS = 60
    }
}
