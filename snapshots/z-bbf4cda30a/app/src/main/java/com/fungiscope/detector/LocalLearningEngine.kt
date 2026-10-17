package com.fungiscope.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.RectF
import android.os.Environment
import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Durable post-inference learning for an immutable TFLite detector.
 *
 * Confirmed object crops are matched with rotation-tolerant weighted nearest
 * neighbors. Broad class remapping is intentionally never used for inference:
 * it can collapse visually different objects into one frequently predicted class.
 */
class LocalLearningEngine(
    private val context: Context
) {
    data class Stats(
        val evaluations: Int,
        val learnedRules: Int,
        val visualExamples: Int,
        val bundledSeedExamples: Int,
        val unreliableLabels: Int
    )

    private data class VisualMatch(
        val label: String,
        val similarity: Float,
        val support: Int
    )

    private data class VisualExample(
        val label: String,
        val feature: ByteArray,
        val voteWeight: Float
    )

    private data class RuleSummary(
        val suggestedLabel: String,
        val total: Int,
        val agreement: Float,
        val rawAccuracy: Float
    )

    private val preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
    private var cachedVisualExamples: List<VisualExample>? = null
    private var cachedRuleSummaries: Map<String, RuleSummary>? = null
    private val bundledSeedExamples: JSONArray by lazy {
        try {
            context.assets.open(BUNDLED_SEED_ASSET).bufferedReader().use { reader ->
                JSONObject(reader.readText()).optJSONArray("examples") ?: JSONArray()
            }
        } catch (_: Exception) {
            JSONArray()
        }
    }

    init {
        reconcileVisualMemoryFromTrainingData()
    }

    @Synchronized
    fun applyCorrections(sourceBitmap: Bitmap, detections: List<Detection>): List<Detection> {
        val examples = decodedVisualExamples()
        val unreliableLabels = unreliableRawLabels()
        return detections.mapNotNull { detection ->
            val visualMatch = findVisualMatch(sourceBitmap, detection, examples)
            when {
                visualMatch == null -> detection.copy(
                    reviewRecommended = detection.reviewRecommended || detection.label in unreliableLabels
                )
                visualMatch.label == NO_BACTERIA -> null
                else -> detection.copy(
                    label = visualMatch.label,
                    locallyCorrected = true,
                    reviewRecommended = false
                )
            }
        }
    }

    @Synchronized
    fun recordEvaluation(
        predictedLabel: String,
        correctLabel: String,
        imageName: String,
        sourceBitmap: Bitmap,
        detection: Detection
    ) {
        val rules = loadRules()
        val visualExamples = loadVisualExamples()

        extractFeature(sourceBitmap, detection)?.let { feature ->
            for (index in visualExamples.length() - 1 downTo 0) {
                val existing = visualExamples.optJSONObject(index) ?: continue
                val encoded = existing.optString("feature")
                val previousFeature = runCatching { Base64.decode(encoded, Base64.NO_WRAP) }.getOrNull()
                    ?: continue
                if (colorsAreClose(feature, previousFeature) &&
                    featureSimilarityWithTransforms(feature, previousFeature) >= REPLACEMENT_SIMILARITY
                ) {
                    decrementVote(
                        root = rules,
                        predictedLabel = existing.optString("predicted"),
                        correctLabel = existing.optString("correct")
                    )
                    visualExamples.remove(index)
                }
            }
            visualExamples.put(
                JSONObject()
                    .put("predicted", predictedLabel)
                    .put("correct", correctLabel)
                    .put("feature", Base64.encodeToString(feature, Base64.NO_WRAP))
                    .put("created", System.currentTimeMillis())
            )
            while (visualExamples.length() > MAX_VISUAL_EXAMPLES) {
                val pruneIndex = indexToPrune(visualExamples)
                val removed = visualExamples.optJSONObject(pruneIndex)
                if (removed != null) {
                    decrementVote(
                        root = rules,
                        predictedLabel = removed.optString("predicted"),
                        correctLabel = removed.optString("correct")
                    )
                }
                visualExamples.remove(pruneIndex)
            }
        }
        incrementVote(rules, predictedLabel, correctLabel)

        val saved = preferences.edit()
            .putString(KEY_RULES, rules.toString())
            .putString(KEY_VISUAL_EXAMPLES, visualExamples.toString())
            .putInt(KEY_EVALUATIONS, preferences.getInt(KEY_EVALUATIONS, 0) + 1)
            .commit()
        if (!saved) {
            // The clean training crop is still saved by TrainingDataManager and
            // will rebuild this compact cache on the next app start.
        }

        cachedVisualExamples = null
        cachedRuleSummaries = null
        appendEvaluationLog(predictedLabel, correctLabel, imageName)
    }

    @Synchronized
    fun learnedLabelFor(predictedLabel: String): String? {
        val summary = ruleSummaries()[predictedLabel] ?: return null
        return if (summary.total >= MINIMUM_RULE_SAMPLES &&
            summary.agreement >= MINIMUM_RULE_SUGGESTION_AGREEMENT
        ) summary.suggestedLabel else null
    }

    @Synchronized
    fun unreliableLabelNames(): List<String> = unreliableRawLabels().sorted()

    @Synchronized
    fun stats(): Stats {
        val summaries = ruleSummaries()
        return Stats(
            evaluations = preferences.getInt(KEY_EVALUATIONS, 0),
            learnedRules = summaries.values.count {
                it.total >= MINIMUM_RULE_SAMPLES && it.agreement >= MINIMUM_RULE_SUGGESTION_AGREEMENT
            },
            visualExamples = loadVisualExamples().length(),
            bundledSeedExamples = bundledSeedExamples.length(),
            unreliableLabels = summaries.count { (_, summary) ->
                summary.total >= MINIMUM_RELIABILITY_SAMPLES && summary.rawAccuracy < MINIMUM_RAW_ACCURACY
            }
        )
    }

    @Synchronized
    fun reset() {
        val resetAt = System.currentTimeMillis()
        val saved = preferences.edit()
            .clear()
            .putLong(KEY_LEARNING_RESET_EPOCH, resetAt)
            .putString(KEY_DATASET_SIGNATURE, "reset:$resetAt")
            .commit()
        if (!saved) return
        cachedVisualExamples = null
        cachedRuleSummaries = null
        appendEvaluationLog("SYSTEM", "LEARNING_RESET", "")
    }

    private fun findVisualMatch(
        sourceBitmap: Bitmap,
        detection: Detection,
        examples: List<VisualExample>
    ): VisualMatch? {
        if (examples.isEmpty()) return null
        val currentFeature = extractFeature(sourceBitmap, detection) ?: return null
        val nearest = examples.asSequence()
            .map { example -> example to featureSimilarityWithTransforms(currentFeature, example.feature) }
            .filter { (_, similarity) -> similarity >= MINIMUM_NEIGHBOR_SIMILARITY }
            .sortedByDescending { (_, similarity) -> similarity }
            .take(NEIGHBOR_COUNT)
            .toList()
        if (nearest.isEmpty()) return null

        val scores = mutableMapOf<String, Float>()
        val support = mutableMapOf<String, Int>()
        val bestSimilarity = mutableMapOf<String, Float>()
        nearest.forEach { (example, similarity) ->
            val normalized = ((similarity - MINIMUM_NEIGHBOR_SIMILARITY) /
                (1f - MINIMUM_NEIGHBOR_SIMILARITY)).coerceIn(0f, 1f)
            val weight = (0.05f + normalized * normalized) * example.voteWeight
            scores[example.label] = (scores[example.label] ?: 0f) + weight
            support[example.label] = (support[example.label] ?: 0) + 1
            bestSimilarity[example.label] = maxOf(bestSimilarity[example.label] ?: 0f, similarity)
        }

        val ranked = scores.entries.sortedByDescending { it.value }
        val winner = ranked.first()
        val runnerUpScore = ranked.getOrNull(1)?.value ?: 0f
        val winnerSupport = support[winner.key] ?: 0
        val winnerSimilarity = bestSimilarity[winner.key] ?: 0f
        val requiredSimilarity = if (winnerSupport >= 2) {
            MINIMUM_MULTI_EXAMPLE_MATCH
        } else {
            MINIMUM_SINGLE_EXAMPLE_MATCH
        }
        if (winnerSimilarity < requiredSimilarity) return null
        if (runnerUpScore > 0f && winner.value < runnerUpScore * MINIMUM_SCORE_LEAD) return null
        return VisualMatch(winner.key, winnerSimilarity, winnerSupport)
    }

    private fun decodedVisualExamples(): List<VisualExample> {
        cachedVisualExamples?.let { return it }
        val allExamples = decodeExamples(loadVisualExamples(), LOCAL_EXAMPLE_WEIGHT) +
            decodeExamples(bundledSeedExamples, BUNDLED_SEED_WEIGHT)
        val representatives = allExamples
            .groupBy(VisualExample::label)
            .values
            .flatMap(::selectRepresentativeExamples)
        cachedVisualExamples = representatives
        return representatives
    }

    private fun decodeExamples(json: JSONArray, voteWeight: Float): List<VisualExample> = buildList {
        repeat(json.length()) { index ->
            val example = json.optJSONObject(index) ?: return@repeat
            val label = example.optString("correct")
            val encoded = example.optString("feature")
            if (label.isBlank() || encoded.isBlank()) return@repeat
            val feature = runCatching { Base64.decode(encoded, Base64.NO_WRAP) }.getOrNull()
                ?: return@repeat
            add(VisualExample(label, feature, voteWeight))
        }
    }

    private fun selectRepresentativeExamples(examples: List<VisualExample>): List<VisualExample> {
        if (examples.size <= MAX_MATCH_EXAMPLES_PER_LABEL) return examples
        val local = examples.filter { it.voteWeight > BUNDLED_SEED_WEIGHT }
        val seed = examples.filter { it.voteWeight <= BUNDLED_SEED_WEIGHT }
        if (local.size >= MAX_MATCH_EXAMPLES_PER_LABEL) {
            return evenlySelect(local, MAX_MATCH_EXAMPLES_PER_LABEL)
        }
        return local + evenlySelect(seed, MAX_MATCH_EXAMPLES_PER_LABEL - local.size)
    }

    private fun evenlySelect(examples: List<VisualExample>, count: Int): List<VisualExample> {
        if (count <= 0 || examples.isEmpty()) return emptyList()
        if (examples.size <= count) return examples
        if (count == 1) return listOf(examples.last())
        val lastIndex = examples.lastIndex
        return List(count) { position ->
            examples[position * lastIndex / (count - 1)]
        }.distinct()
    }

    private fun indexToPrune(examples: JSONArray): Int {
        val counts = mutableMapOf<String, Int>()
        repeat(examples.length()) { index ->
            val label = examples.optJSONObject(index)?.optString("correct").orEmpty()
            counts[label] = (counts[label] ?: 0) + 1
        }
        val mostRepresented = counts.maxByOrNull { it.value }?.key.orEmpty()
        repeat(examples.length()) { index ->
            if (examples.optJSONObject(index)?.optString("correct") == mostRepresented) return index
        }
        return 0
    }

    /**
     * The clean training crops are the durable source of truth. A compact
     * feature cache is rebuilt whenever the on-disk sample set changes.
     */
    private fun reconcileVisualMemoryFromTrainingData() {
        val documents = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: context.filesDir
        val root = File(documents, "BacterialScope_Training")
        val images = File(root, "images")
        val metadata = File(root, "metadata")
        val resetEpoch = preferences.getLong(KEY_LEARNING_RESET_EPOCH, 0L)
        val candidates = images.listFiles { file ->
            file.isFile && file.extension.equals("jpg", ignoreCase = true) && file.lastModified() > resetEpoch
        }?.sortedByDescending(File::lastModified)?.take(MAX_VISUAL_EXAMPLES).orEmpty()

        val signature = buildString {
            append(resetEpoch)
            append(':')
            append(candidates.size)
            append(':')
            append(candidates.maxOfOrNull(File::lastModified) ?: 0L)
            append(':')
            append(candidates.sumOf(File::length))
        }
        if (preferences.getString(KEY_DATASET_SIGNATURE, "") == signature &&
            loadVisualExamples().length() > 0
        ) return
        if (candidates.isEmpty()) {
            if (resetEpoch > 0L) {
                preferences.edit()
                    .putString(KEY_VISUAL_EXAMPLES, "[]")
                    .putString(KEY_DATASET_SIGNATURE, signature)
                    .commit()
            }
            return
        }

        val rebuilt = JSONArray()
        val rebuiltRules = JSONObject()
        val rebuiltFeatureBuckets = mutableMapOf<Int, MutableList<ByteArray>>()
        candidates.forEach { imageFile ->
            val metadataFile = File(metadata, "${imageFile.nameWithoutExtension}.json")
            val details = runCatching { JSONObject(metadataFile.readText()) }.getOrNull()
                ?: return@forEach
            val created = details.optLong("created_epoch", imageFile.lastModified())
            if (created <= resetEpoch) return@forEach
            val predicted = details.optString("predicted_label")
            val correct = details.optString("correct_label")
            if (predicted.isBlank() || correct.isBlank()) return@forEach
            val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath) ?: return@forEach
            try {
                val wholeCrop = Detection(
                    label = predicted,
                    confidence = details.optDouble("confidence", 1.0).toFloat(),
                    box = RectF(0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat())
                )
                val feature = extractFeature(bitmap, wholeCrop) ?: return@forEach
                val bucket = featureBucket(feature)
                val nearbyFeatures = rebuiltFeatureBuckets.getOrPut(bucket) { mutableListOf() }
                if (nearbyFeatures.any { existing ->
                        featureSimilarityWithTransforms(feature, existing) >= REPLACEMENT_SIMILARITY
                    }
                ) return@forEach
                nearbyFeatures += feature
                incrementVote(rebuiltRules, predicted, correct)
                rebuilt.put(
                    JSONObject()
                        .put("predicted", predicted)
                        .put("correct", correct)
                        .put("feature", Base64.encodeToString(feature, Base64.NO_WRAP))
                        .put("created", created)
                )
            } finally {
                if (!bitmap.isRecycled) bitmap.recycle()
            }
        }
        preferences.edit()
            .putString(KEY_VISUAL_EXAMPLES, rebuilt.toString())
            .putString(KEY_RULES, rebuiltRules.toString())
            .putString(KEY_DATASET_SIGNATURE, signature)
            .commit()
        cachedVisualExamples = null
        cachedRuleSummaries = null
    }

    private fun ruleSummaries(): Map<String, RuleSummary> {
        cachedRuleSummaries?.let { return it }
        val root = JSONObject(loadRules().toString())
        repeat(bundledSeedExamples.length()) { index ->
            val example = bundledSeedExamples.optJSONObject(index) ?: return@repeat
            val predicted = example.optString("predicted")
            val correct = example.optString("correct")
            if (predicted.isNotBlank() && correct.isNotBlank()) {
                incrementVote(root, predicted, correct)
            }
        }
        val summaries = buildMap {
            val predictedLabels = root.keys()
            while (predictedLabels.hasNext()) {
                val predicted = predictedLabels.next()
                val votes = root.optJSONObject(predicted) ?: continue
                var total = 0
                var winningLabel = ""
                var winningVotes = 0
                val labels = votes.keys()
                while (labels.hasNext()) {
                    val label = labels.next()
                    val count = votes.optInt(label, 0)
                    total += count
                    if (count > winningVotes) {
                        winningVotes = count
                        winningLabel = label
                    }
                }
                if (total > 0 && winningLabel.isNotBlank()) {
                    put(
                        predicted,
                        RuleSummary(
                            suggestedLabel = winningLabel,
                            total = total,
                            agreement = winningVotes.toFloat() / total,
                            rawAccuracy = votes.optInt(predicted, 0).toFloat() / total
                        )
                    )
                }
            }
        }
        cachedRuleSummaries = summaries
        return summaries
    }

    private fun unreliableRawLabels(): Set<String> = ruleSummaries()
        .filterValues { summary ->
            summary.total >= MINIMUM_RELIABILITY_SAMPLES && summary.rawAccuracy < MINIMUM_RAW_ACCURACY
        }
        .keys

    private fun incrementVote(root: JSONObject, predictedLabel: String, correctLabel: String) {
        val votes = root.optJSONObject(predictedLabel) ?: JSONObject()
        votes.put(correctLabel, votes.optInt(correctLabel, 0) + 1)
        root.put(predictedLabel, votes)
    }

    private fun decrementVote(root: JSONObject, predictedLabel: String, correctLabel: String) {
        if (predictedLabel.isBlank() || correctLabel.isBlank()) return
        val votes = root.optJSONObject(predictedLabel) ?: return
        val count = votes.optInt(correctLabel, 0)
        if (count <= 1) votes.remove(correctLabel) else votes.put(correctLabel, count - 1)
        if (!votes.keys().hasNext()) root.remove(predictedLabel) else root.put(predictedLabel, votes)
    }

    private fun extractFeature(sourceBitmap: Bitmap, detection: Detection): ByteArray? {
        if (sourceBitmap.width < 2 || sourceBitmap.height < 2) return null
        val box = detection.box
        val paddingX = box.width() * FEATURE_PADDING
        val paddingY = box.height() * FEATURE_PADDING
        val left = floor(box.left - paddingX).toInt().coerceIn(0, sourceBitmap.width - 1)
        val top = floor(box.top - paddingY).toInt().coerceIn(0, sourceBitmap.height - 1)
        val right = ceil(box.right + paddingX).toInt().coerceIn(left + 1, sourceBitmap.width)
        val bottom = ceil(box.bottom + paddingY).toInt().coerceIn(top + 1, sourceBitmap.height)
        if (right - left < 2 || bottom - top < 2) return null

        val crop = Bitmap.createBitmap(sourceBitmap, left, top, right - left, bottom - top)
        val scaled = Bitmap.createScaledBitmap(crop, FEATURE_GRID_SIZE, FEATURE_GRID_SIZE, true)
        try {
            val pixels = IntArray(FEATURE_GRID_SIZE * FEATURE_GRID_SIZE)
            scaled.getPixels(pixels, 0, FEATURE_GRID_SIZE, 0, 0, FEATURE_GRID_SIZE, FEATURE_GRID_SIZE)
            val luminance = FloatArray(pixels.size)
            var total = 0f
            var totalRed = 0f
            var totalGreen = 0f
            var totalBlue = 0f
            pixels.forEachIndexed { index, pixel ->
                val red = Color.red(pixel).toFloat()
                val green = Color.green(pixel).toFloat()
                val blue = Color.blue(pixel).toFloat()
                val value = red * 0.299f + green * 0.587f + blue * 0.114f
                luminance[index] = value
                total += value
                totalRed += red
                totalGreen += green
                totalBlue += blue
            }

            val mean = total / luminance.size
            var variance = 0f
            luminance.forEach { value ->
                val difference = value - mean
                variance += difference * difference
            }
            val standardDeviation = sqrt(variance / luminance.size).coerceAtLeast(10f)
            val feature = ByteArray(luminance.size + COLOR_FEATURES)
            luminance.forEachIndexed { index, value ->
                val normalized = ((value - mean) / standardDeviation).coerceIn(-2.5f, 2.5f)
                feature[index] = (((normalized + 2.5f) / 5f) * 255f).toInt().coerceIn(0, 255).toByte()
            }

            val colorTotal = (totalRed + totalGreen + totalBlue).coerceAtLeast(1f)
            feature[luminance.size] = (totalRed / colorTotal * 255f).toInt().coerceIn(0, 255).toByte()
            feature[luminance.size + 1] = (totalGreen / colorTotal * 255f).toInt().coerceIn(0, 255).toByte()
            feature[luminance.size + 2] = (totalBlue / colorTotal * 255f).toInt().coerceIn(0, 255).toByte()
            return feature
        } finally {
            if (scaled !== crop && !scaled.isRecycled) scaled.recycle()
            if (crop !== sourceBitmap && !crop.isRecycled) crop.recycle()
        }
    }

    private fun featureSimilarityWithTransforms(first: ByteArray, second: ByteArray): Float {
        val gridPixels = FEATURE_GRID_SIZE * FEATURE_GRID_SIZE
        if (first.size != gridPixels + COLOR_FEATURES || second.size != first.size) return 0f
        var bestDifference = Long.MAX_VALUE
        repeat(8) { transform ->
            var difference = 0L
            repeat(FEATURE_GRID_SIZE) { y ->
                repeat(FEATURE_GRID_SIZE) { x ->
                    val sourceIndex = transformedIndex(x, y, transform)
                    val firstValue = first[sourceIndex].toInt() and 0xFF
                    val secondValue = second[y * FEATURE_GRID_SIZE + x].toInt() and 0xFF
                    difference += kotlin.math.abs(firstValue - secondValue)
                }
            }
            repeat(COLOR_FEATURES) { offset ->
                val index = gridPixels + offset
                difference += kotlin.math.abs(
                    (first[index].toInt() and 0xFF) - (second[index].toInt() and 0xFF)
                ) * COLOR_WEIGHT
            }
            bestDifference = minOf(bestDifference, difference)
        }
        val weightedSize = gridPixels + COLOR_FEATURES * COLOR_WEIGHT
        return 1f - bestDifference.toFloat() / (weightedSize * 255f)
    }

    private fun colorsAreClose(first: ByteArray, second: ByteArray): Boolean {
        val gridPixels = FEATURE_GRID_SIZE * FEATURE_GRID_SIZE
        if (first.size != gridPixels + COLOR_FEATURES || second.size != first.size) return false
        var difference = 0
        repeat(COLOR_FEATURES) { offset ->
            val index = gridPixels + offset
            difference += kotlin.math.abs(
                (first[index].toInt() and 0xFF) - (second[index].toInt() and 0xFF)
            )
        }
        return difference <= MAX_REPLACEMENT_COLOR_DIFFERENCE
    }

    private fun featureBucket(feature: ByteArray): Int {
        val gridPixels = FEATURE_GRID_SIZE * FEATURE_GRID_SIZE
        if (feature.size < gridPixels + COLOR_FEATURES) return 0
        var key = 0
        repeat(COLOR_FEATURES) { offset ->
            key = key * COLOR_BUCKETS_PER_CHANNEL +
                ((feature[gridPixels + offset].toInt() and 0xFF) / COLOR_BUCKET_SIZE)
                    .coerceAtMost(COLOR_BUCKETS_PER_CHANNEL - 1)
        }
        return key
    }

    private fun transformedIndex(x: Int, y: Int, transform: Int): Int {
        val last = FEATURE_GRID_SIZE - 1
        val mirroredX = if (transform >= 4) last - x else x
        return when (transform % 4) {
            0 -> y * FEATURE_GRID_SIZE + mirroredX
            1 -> (last - mirroredX) * FEATURE_GRID_SIZE + y
            2 -> (last - y) * FEATURE_GRID_SIZE + (last - mirroredX)
            else -> mirroredX * FEATURE_GRID_SIZE + (last - y)
        }
    }

    private fun loadRules(): JSONObject = try {
        JSONObject(preferences.getString(KEY_RULES, "{}") ?: "{}")
    } catch (_: Exception) {
        JSONObject()
    }

    private fun loadVisualExamples(): JSONArray = try {
        JSONArray(preferences.getString(KEY_VISUAL_EXAMPLES, "[]") ?: "[]")
    } catch (_: Exception) {
        JSONArray()
    }

    private fun appendEvaluationLog(predicted: String, correct: String, imageName: String) {
        try {
            val directory = File(
                context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: context.filesDir,
                "BacterialScope_Learning"
            ).apply { mkdirs() }
            val file = File(directory, "post_evaluations.csv")
            if (!file.exists()) file.writeText("timestamp,image,predicted,correct\n")
            val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())
            file.appendText(
                listOf(timestamp, imageName, predicted, correct)
                    .joinToString(",") { csvEscape(it) } + "\n"
            )
        } catch (_: Exception) {
            // The durable training crop remains available even if the audit log cannot be written.
        }
    }

    private fun csvEscape(value: String): String = "\"${value.replace("\"", "\"\"")}\""

    companion object {
        const val NO_BACTERIA = "No bacteria / false detection"
        private const val PREFERENCES = "bacterialscope_local_learning_v1"
        private const val KEY_RULES = "correction_votes"
        private const val KEY_VISUAL_EXAMPLES = "visual_examples"
        private const val KEY_EVALUATIONS = "evaluation_count"
        private const val KEY_LEARNING_RESET_EPOCH = "learning_reset_epoch"
        private const val KEY_DATASET_SIGNATURE = "training_dataset_signature"
        private const val BUNDLED_SEED_ASSET = "bacterial_learning_seed.json"
        private const val LOCAL_EXAMPLE_WEIGHT = 1.35f
        private const val BUNDLED_SEED_WEIGHT = 1.0f
        private const val FEATURE_GRID_SIZE = 10
        private const val COLOR_FEATURES = 3
        private const val COLOR_WEIGHT = 3
        private const val COLOR_BUCKET_SIZE = 64
        private const val COLOR_BUCKETS_PER_CHANNEL = 4
        private const val MAX_REPLACEMENT_COLOR_DIFFERENCE = 18
        private const val FEATURE_PADDING = 0.20f
        private const val MINIMUM_NEIGHBOR_SIMILARITY = 0.80f
        private const val MINIMUM_SINGLE_EXAMPLE_MATCH = 0.88f
        private const val MINIMUM_MULTI_EXAMPLE_MATCH = 0.84f
        private const val MINIMUM_SCORE_LEAD = 1.12f
        private const val REPLACEMENT_SIMILARITY = 0.985f
        private const val NEIGHBOR_COUNT = 7
        private const val MAX_MATCH_EXAMPLES_PER_LABEL = 12
        private const val MAX_VISUAL_EXAMPLES = 2400
        private const val MINIMUM_RULE_SAMPLES = 2
        private const val MINIMUM_RULE_SUGGESTION_AGREEMENT = 0.65f
        private const val MINIMUM_RELIABILITY_SAMPLES = 3
        private const val MINIMUM_RAW_ACCURACY = 0.45f
    }
}
