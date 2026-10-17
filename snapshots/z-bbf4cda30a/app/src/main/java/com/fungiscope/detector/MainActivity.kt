package com.fungiscope.detector

import android.Manifest
import android.app.AlertDialog
import android.app.Dialog
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.ColorDrawable
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.MediaStore
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.addCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max

class MainActivity : ComponentActivity() {
    private lateinit var cameraScreen: FrameLayout
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var txtModelState: TextView
    private lateinit var txtLiveChip: TextView
    private lateinit var txtCount: TextView
    private lateinit var txtCategories: TextView
    private lateinit var txtLatency: TextView
    private lateinit var txtThreshold: TextView
    private lateinit var seekThreshold: SeekBar
    private lateinit var btnLive: Button
    private lateinit var btnCapture: Button
    private lateinit var btnFlashDrawerToggle: Button
    private lateinit var flashDrawer: View
    private lateinit var btnFlashOn: Button
    private lateinit var btnFlashOff: Button
    private lateinit var btnZoom: Button
    private lateinit var btnGallery: Button
    private lateinit var resultCard: View
    private lateinit var imgLastCapture: ImageView
    private lateinit var txtLastResult: TextView

    private lateinit var galleryScreen: View
    private lateinit var btnGalleryBack: Button
    private lateinit var btnGalleryInfo: Button
    private lateinit var galleryInfoPanel: View
    private lateinit var txtGallerySubtitle: TextView
    private lateinit var txtGalleryInfo: TextView
    private lateinit var txtGalleryEmpty: TextView
    private lateinit var btnSaveToPhotos: Button
    private lateinit var btnDeletePhoto: Button
    private lateinit var btnExportCsv: Button
    private lateinit var btnExportTraining: Button
    private lateinit var btnLearningMemory: Button
    private lateinit var imgGallery: ImageView
    private lateinit var galleryScroll: HorizontalScrollView
    private lateinit var galleryStrip: LinearLayout

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var galleryExecutor: ExecutorService
    private lateinit var galleryScaleDetector: ScaleGestureDetector
    private lateinit var cameraScaleDetector: ScaleGestureDetector
    private lateinit var learningEngine: LocalLearningEngine
    private lateinit var trainingDataManager: TrainingDataManager
    private val detectionPostProcessor = DetectionPostProcessor()
    private val detectionStabilizer = DetectionStabilizer()
    private var activeEvaluationDialog: Dialog? = null
    private var detector: Detector? = null
    private var camera: Camera? = null
    private var imageCapture: ImageCapture? = null
    private var isLive = true
    private var torchEnabled = false
    private var lastAnalyzedAt = 0L
    private var threshold = 0.40f
    private var zoomIndex = 0
    private var cameraDidPinch = false
    private val categoryLabels: List<String> by lazy {
        assets.open(LABEL_ASSET).bufferedReader().useLines { lines ->
            lines.map(String::trim).filter(String::isNotEmpty).toList()
        }
    }
    private val mainHandler = Handler(Looper.getMainLooper())
    private val dismissCaptureNotice = Runnable {
        if (resultCard.visibility == View.VISIBLE) {
            resultCard.animate().cancel()
            resultCard.animate()
                .alpha(0f)
                .translationY(dp(110).toFloat())
                .setDuration(320L)
                .withEndAction {
                    resultCard.visibility = View.GONE
                    resultCard.alpha = 1f
                    resultCard.translationY = 0f
                }
                .start()
        }
    }

    private var galleryFiles: List<File> = emptyList()
    private var galleryIndex = -1
    private var galleryLoadRequest = 0
    private var galleryScale = 1f
    private var galleryTranslationX = 0f
    private var galleryTranslationY = 0f
    private var galleryDownX = 0f
    private var galleryLastX = 0f
    private var galleryLastY = 0f

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else {
            txtModelState.text = "Camera permission is required"
            Toast.makeText(this, "Allow camera access to scan bacterial samples.", Toast.LENGTH_LONG).show()
        }
    }

    private val createTrainingExport = registerForActivityResult(
        ActivityResultContracts.CreateDocument("application/zip")
    ) { destination ->
        if (destination == null) return@registerForActivityResult
        btnExportTraining.isEnabled = false
        btnExportTraining.alpha = 0.55f
        galleryExecutor.execute {
            try {
                val summary = contentResolver.openOutputStream(destination)?.use { output ->
                    trainingDataManager.exportZip(output, categoryLabels)
                } ?: error("Could not open the selected destination")
                runOnUiThread {
                    btnExportTraining.isEnabled = true
                    btnExportTraining.alpha = 1f
                    Toast.makeText(
                        this,
                        "Training ZIP saved • ${summary.samples} samples (${summary.positiveSamples} positive, ${summary.negativeSamples} negative)",
                        Toast.LENGTH_LONG
                    ).show()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    btnExportTraining.isEnabled = true
                    btnExportTraining.alpha = 1f
                    Toast.makeText(this, "Training export failed: ${error.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
        setContentView(R.layout.activity_main)

        cameraExecutor = Executors.newSingleThreadExecutor()
        galleryExecutor = Executors.newSingleThreadExecutor()
        bindViews()
        learningEngine = LocalLearningEngine(this)
        trainingDataManager = TrainingDataManager(this)
        setupControls()
        setupCameraGestures()
        setupGalleryGestures()
        setupDetector()
        updateLearningButton()
        updateTrainingExportButton()

        onBackPressedDispatcher.addCallback(this) {
            if (galleryScreen.visibility == View.VISIBLE) showCameraScreen() else finish()
        }

        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            previewView.post { startCamera() }
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun bindViews() {
        cameraScreen = findViewById(R.id.cameraScreen)
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        txtModelState = findViewById(R.id.txtModelState)
        txtLiveChip = findViewById(R.id.txtLiveChip)
        txtCount = findViewById(R.id.txtCount)
        txtCategories = findViewById(R.id.txtCategories)
        txtLatency = findViewById(R.id.txtLatency)
        txtThreshold = findViewById(R.id.txtThreshold)
        seekThreshold = findViewById(R.id.seekThreshold)
        btnLive = findViewById(R.id.btnLive)
        btnCapture = findViewById(R.id.btnCapture)
        btnFlashDrawerToggle = findViewById(R.id.btnFlashDrawerToggle)
        flashDrawer = findViewById(R.id.flashDrawer)
        btnFlashOn = findViewById(R.id.btnFlashOn)
        btnFlashOff = findViewById(R.id.btnFlashOff)
        btnZoom = findViewById(R.id.btnZoom)
        btnGallery = findViewById(R.id.btnGallery)
        resultCard = findViewById(R.id.resultCard)
        imgLastCapture = findViewById(R.id.imgLastCapture)
        txtLastResult = findViewById(R.id.txtLastResult)

        galleryScreen = findViewById(R.id.galleryScreen)
        btnGalleryBack = findViewById(R.id.btnGalleryBack)
        btnGalleryInfo = findViewById(R.id.btnGalleryInfo)
        galleryInfoPanel = findViewById(R.id.galleryInfoPanel)
        txtGallerySubtitle = findViewById(R.id.txtGallerySubtitle)
        txtGalleryInfo = findViewById(R.id.txtGalleryInfo)
        txtGalleryEmpty = findViewById(R.id.txtGalleryEmpty)
        btnSaveToPhotos = findViewById(R.id.btnSaveToPhotos)
        btnDeletePhoto = findViewById(R.id.btnDeletePhoto)
        btnExportCsv = findViewById(R.id.btnExportCsv)
        btnExportTraining = findViewById(R.id.btnExportTraining)
        btnLearningMemory = findViewById(R.id.btnLearningMemory)
        imgGallery = findViewById(R.id.imgGallery)
        galleryScroll = findViewById(R.id.galleryScroll)
        galleryStrip = findViewById(R.id.galleryStrip)
    }

    private fun setupControls() {
        listOf(
            btnLive, btnCapture, btnFlashDrawerToggle, btnFlashOn, btnFlashOff, btnZoom, btnGallery,
            btnGalleryBack, btnGalleryInfo, btnSaveToPhotos, btnDeletePhoto, btnExportCsv,
            btnExportTraining, btnLearningMemory
        ).forEach { it.applySoftPress() }

        btnLive.setOnClickListener {
            isLive = !isLive
            btnLive.text = if (isLive) "Pause" else "Resume"
            refreshLiveChip()
            if (!isLive) {
                detectionStabilizer.clear()
                overlayView.clear()
                txtCount.text = "0"
                txtLatency.text = "paused"
            }
            it.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
        }

        btnFlashDrawerToggle.setOnClickListener { toggleFlashDrawer() }
        btnFlashOn.setOnClickListener { setFlashEnabled(true) }
        btnFlashOff.setOnClickListener { setFlashEnabled(false) }

        btnZoom.setOnClickListener {
            val activeCamera = camera ?: return@setOnClickListener
            val zoomState = activeCamera.cameraInfo.zoomState.value ?: return@setOnClickListener
            val presets = listOf(1f, 2f, 3f).filter { ratio -> ratio <= zoomState.maxZoomRatio + 0.01f }
            if (presets.isEmpty()) return@setOnClickListener
            zoomIndex = (zoomIndex + 1) % presets.size
            val ratio = presets[zoomIndex].coerceAtLeast(zoomState.minZoomRatio)
            activeCamera.cameraControl.setZoomRatio(ratio)
            btnZoom.text = "${ratio.toInt()}×"
            it.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK)
        }

        btnCapture.setOnClickListener {
            captureScan()
            it.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
        }
        btnGallery.setOnClickListener { showGalleryScreen() }
        btnGalleryBack.setOnClickListener { showCameraScreen() }
        btnGalleryInfo.setOnClickListener { toggleGalleryInfo() }
        btnSaveToPhotos.setOnClickListener { saveSelectedToPhotos() }
        btnDeletePhoto.setOnClickListener { confirmDeleteSelected() }
        btnExportCsv.setOnClickListener { exportGalleryCsv() }
        btnExportTraining.setOnClickListener { exportTrainingData() }
        btnLearningMemory.setOnClickListener { showLearningMemoryDialog() }

        seekThreshold.max = 80
        seekThreshold.progress = 30
        seekThreshold.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                threshold = (progress + 10) / 100f
                detector?.setThreshold(threshold)
                detectionStabilizer.clear()
                txtThreshold.text = "Confidence ${(threshold * 100).toInt()}%"
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                seekBar?.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK)
            }
        })
    }

    private fun refreshLiveChip() {
        txtLiveChip.text = when {
            !isLive -> "Ⅱ  PAUSED"
            detector == null -> "●  CAMERA"
            else -> "●  LIVE"
        }
    }

    private fun setupDetector() {
        detectionStabilizer.clear()
        val hasModel = assets.list("")?.contains(MODEL_ASSET) == true
        if (!hasModel) {
            txtModelState.text = "Add bacterial_model.tflite • camera ready"
            txtCategories.text = "Bacterial model needed"
            refreshLiveChip()
            return
        }

        try {
            detector = Detector(
                context = this,
                modelAssetName = MODEL_ASSET,
                labelAssetName = LABEL_ASSET,
                scoreThreshold = threshold
            )
            txtModelState.text = detector?.summary() ?: "Detector ready"
            txtCategories.text = "Ready to scan"
            refreshLiveChip()
        } catch (error: Exception) {
            detector = null
            txtModelState.text = "Model error • ${error.message}"
            txtCategories.text = "Check model export format"
            txtLiveChip.text = "!  MODEL"
        }
    }

    private fun startCamera() {
        detectionStabilizer.clear()
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val rotation = previewView.display.rotation
                val preview = Preview.Builder()
                    .setTargetRotation(rotation)
                    .build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                    .setTargetRotation(rotation)
                    .build()

                val analysis = ImageAnalysis.Builder()
                    .setTargetRotation(rotation)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(cameraExecutor, ::analyzeFrame) }

                provider.unbindAll()
                camera = provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture,
                    analysis
                )
                torchEnabled = false
                btnFlashDrawerToggle.text = "Flash"
                flashDrawer.visibility = View.GONE
            } catch (error: Exception) {
                txtModelState.text = "Camera error • ${error.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(image: ImageProxy) {
        try {
            val activeDetector = detector ?: return
            if (!isLive || galleryScreen.visibility == View.VISIBLE) return
            val now = SystemClock.uptimeMillis()
            if (now - lastAnalyzedAt < ANALYSIS_INTERVAL_MS) return
            lastAnalyzedAt = now

            val started = SystemClock.uptimeMillis()
            val bitmap = image.toBitmapSmart().rotate(image.imageInfo.rotationDegrees)
            val rawDetections = activeDetector.detect(bitmap)
            val correctedDetections = learningEngine.applyCorrections(bitmap, rawDetections)
            val detections = detectionStabilizer.update(
                correctedDetections,
                bitmap.width,
                bitmap.height
            )
            val elapsed = SystemClock.uptimeMillis() - started
            val sourceWidth = bitmap.width
            val sourceHeight = bitmap.height
            bitmap.recycle()

            runOnUiThread {
                if (!isLive || galleryScreen.visibility == View.VISIBLE) return@runOnUiThread
                overlayView.setResults(detections, sourceWidth, sourceHeight)
                updateResultText(detections, elapsed)
            }
        } catch (error: Exception) {
            runOnUiThread {
                overlayView.clear()
                txtModelState.text = "Detection error • ${error.message}"
            }
        } finally {
            image.close()
        }
    }

    private fun updateResultText(detections: List<Detection>, elapsedMs: Long) {
        txtCount.text = detections.size.toString()
        txtLatency.text = "$elapsedMs ms • on device"
        txtCategories.text = if (detections.isEmpty()) {
            "No colonies above threshold"
        } else {
            val summary = detections.groupingBy(Detection::label).eachCount()
                .entries.sortedByDescending { it.value }
                .joinToString("  •  ") { "${it.key} ${it.value}" }
            val reviewCount = detections.count(Detection::reviewRecommended)
            if (reviewCount > 0) "$summary  •  REVIEW $reviewCount" else summary
        }
    }

    private fun captureScan() {
        val capture = imageCapture ?: return
        detectionStabilizer.clear()
        btnCapture.isEnabled = false
        btnCapture.alpha = 0.55f
        capture.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                try {
                    val cleanBitmap = image.toBitmapSmart().rotate(image.imageInfo.rotationDegrees)
                    val started = SystemClock.uptimeMillis()
                    val rawDetections = detector?.let { activeDetector ->
                        runAccurateCaptureDetection(activeDetector, cleanBitmap)
                    }.orEmpty()
                    val detections = learningEngine.applyCorrections(cleanBitmap, rawDetections)
                    val latency = if (detector == null) null else SystemClock.uptimeMillis() - started
                    val annotatedBitmap = drawDetections(
                        checkNotNull(cleanBitmap.copy(Bitmap.Config.ARGB_8888, true)),
                        detections
                    )
                    val savedFile = saveBitmap(annotatedBitmap)
                    saveCaptureMetadata(savedFile, detections, latency)
                    val resultText = when {
                        detector == null -> "Photo saved • add bacterial_model.tflite to analyze"
                        rawDetections.isEmpty() ->
                            "No model box found • lower confidence to teach this sample"
                        detections.isEmpty() -> "Detection suppressed by learning memory"
                        else -> detections.groupingBy(Detection::label).eachCount()
                            .entries.joinToString(" • ") { "${it.key}: ${it.value}" }
                    }
                    runOnUiThread {
                        showCaptureNotice(annotatedBitmap, resultText)
                        btnCapture.isEnabled = true
                        btnCapture.alpha = 1f
                        Toast.makeText(this@MainActivity, "Saved ${savedFile.name}", Toast.LENGTH_SHORT).show()
                        if (rawDetections.isNotEmpty()) {
                            mainHandler.postDelayed({
                                if (!isFinishing && cameraScreen.visibility == View.VISIBLE) {
                                    showPostEvaluation(
                                        cleanBitmap = cleanBitmap,
                                        rawDetections = rawDetections,
                                        savedFile = savedFile,
                                        latencyMs = latency
                                    )
                                } else if (!cleanBitmap.isRecycled) {
                                    cleanBitmap.recycle()
                                }
                            }, 520L)
                        } else if (!cleanBitmap.isRecycled) {
                            cleanBitmap.recycle()
                        }
                    }
                } catch (error: Exception) {
                    showCaptureError(error)
                } finally {
                    image.close()
                }
            }

            override fun onError(exception: ImageCaptureException) = showCaptureError(exception)
        })
    }

    private fun runAccurateCaptureDetection(
        activeDetector: Detector,
        sourceBitmap: Bitmap
    ): List<Detection> {
        val passes = mutableListOf<List<Detection>>()
        passes += activeDetector.detect(sourceBitmap)

        val enhancementSettings = listOf(
            Triple(1.10f, 4f, 1.12f),
            Triple(1.18f, 6f, 1.18f)
        )
        enhancementSettings.forEach { (contrast, brightness, saturation) ->
            val enhanced = enhanceBitmapForDetection(
                sourceBitmap = sourceBitmap,
                contrast = contrast,
                brightness = brightness,
                saturation = saturation
            )
            try {
                passes += activeDetector.detect(enhanced)
            } finally {
                enhanced.recycle()
            }
        }

        return detectionPostProcessor.fusePasses(
            passes = passes,
            imageWidth = sourceBitmap.width,
            imageHeight = sourceBitmap.height
        )
    }

    private fun enhanceBitmapForDetection(
        sourceBitmap: Bitmap,
        contrast: Float,
        brightness: Float,
        saturation: Float
    ): Bitmap {
        val output = Bitmap.createBitmap(
            sourceBitmap.width,
            sourceBitmap.height,
            Bitmap.Config.ARGB_8888
        )
        val saturationMatrix = ColorMatrix().apply { setSaturation(saturation) }
        val contrastMatrix = ColorMatrix(
            floatArrayOf(
                contrast, 0f, 0f, 0f, brightness,
                0f, contrast, 0f, 0f, brightness,
                0f, 0f, contrast, 0f, brightness,
                0f, 0f, 0f, 1f, 0f
            )
        )
        saturationMatrix.postConcat(contrastMatrix)
        val paint = Paint().apply {
            isAntiAlias = true
            colorFilter = ColorMatrixColorFilter(saturationMatrix)
        }
        Canvas(output).drawBitmap(sourceBitmap, 0f, 0f, paint)
        return output
    }

    private fun showCaptureError(error: Exception) {
        runOnUiThread {
            btnCapture.isEnabled = true
            btnCapture.alpha = 1f
            Toast.makeText(this, "Capture failed: ${error.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun showCaptureNotice(bitmap: Bitmap, message: String) {
        mainHandler.removeCallbacks(dismissCaptureNotice)
        resultCard.animate().cancel()
        imgLastCapture.setImageBitmap(bitmap)
        txtLastResult.text = message
        resultCard.visibility = View.VISIBLE
        resultCard.alpha = 0f
        resultCard.translationY = dp(24).toFloat()
        resultCard.animate()
            .alpha(1f)
            .translationY(0f)
            .setDuration(220L)
            .start()
        mainHandler.postDelayed(dismissCaptureNotice, 1900L)
    }

    private fun showPostEvaluation(
        cleanBitmap: Bitmap,
        rawDetections: List<Detection>,
        savedFile: File,
        latencyMs: Long?
    ) {
        val predictions = rawDetections
            .sortedByDescending(Detection::confidence)
            .take(MAX_EVALUATIONS_PER_CAPTURE)
        if (predictions.isEmpty()) return
        activeEvaluationDialog?.dismiss()

        val dialog = Dialog(this)
        dialog.setContentView(R.layout.dialog_post_evaluation)
        dialog.window?.let { window ->
            window.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
            window.addFlags(WindowManager.LayoutParams.FLAG_DIM_BEHIND)
            window.attributes = window.attributes.apply { dimAmount = 0.72f }
        }

        val image = dialog.findViewById<ImageView>(R.id.imgEvaluation)
        val prediction = dialog.findViewById<TextView>(R.id.txtEvaluationPrediction)
        val choices = dialog.findViewById<RadioGroup>(R.id.evaluationChoices)
        val skip = dialog.findViewById<Button>(R.id.btnSkipEvaluation)
        val teach = dialog.findViewById<Button>(R.id.btnTeachApp)
        var currentIndex = 0
        var savedExamples = 0
        var displayedCrop: Bitmap? = null
        var trainingError: String? = null
        val unreliableLabels = learningEngine.unreliableLabelNames().toSet()

        fun showCurrentPrediction() {
            val current = predictions[currentIndex]
            val nextCrop = cropForEvaluation(cleanBitmap, current)
            val previousCrop = displayedCrop
            image.setImageBitmap(nextCrop)
            displayedCrop = nextCrop
            if (previousCrop != null && previousCrop !== cleanBitmap && previousCrop !== nextCrop &&
                !previousCrop.isRecycled
            ) {
                previousCrop.recycle()
            }

            val learnedSuggestion = learningEngine.learnedLabelFor(current.label)
            prediction.text = buildString {
                append("Object ${currentIndex + 1} of ${predictions.size}\n")
                append("Model predicted: ${current.label}  ${(current.confidence * 100).toInt()}%")
                learnedSuggestion?.let { append("\nMost confirmed correction: $it") }
                if (current.reviewRecommended || current.label in unreliableLabels) {
                    append("\nThis prediction is uncertain. Choose the label manually.")
                }
            }

            choices.removeAllViews()
            val choiceLabels = categoryLabels + LocalLearningEngine.NO_BACTERIA
            val initiallySelected = if (
                current.reviewRecommended || current.label in unreliableLabels
            ) {
                null
            } else {
                learnedSuggestion ?: current.label
            }
            choiceLabels.forEach { label ->
                val option = RadioButton(this).apply {
                    id = View.generateViewId()
                    tag = label
                    text = label
                    textSize = 13f
                    setTextColor(ContextCompat.getColor(this@MainActivity, R.color.mist))
                    buttonTintList = android.content.res.ColorStateList.valueOf(
                        ContextCompat.getColor(this@MainActivity, R.color.plasma)
                    )
                    setPadding(dp(4), dp(4), dp(4), dp(4))
                    isChecked = label == initiallySelected
                }
                choices.addView(option)
            }
            skip.text = if (currentIndex < predictions.lastIndex) "Skip object" else "Finish"
            teach.text = if (currentIndex < predictions.lastIndex) "Teach & next" else "Teach & finish"
        }

        fun advanceOrFinish() {
            if (currentIndex < predictions.lastIndex) {
                currentIndex++
                showCurrentPrediction()
            } else {
                dialog.dismiss()
            }
        }

        listOf(skip, teach).forEach { it.applySoftPress() }
        skip.setOnClickListener { advanceOrFinish() }
        teach.setOnClickListener {
            val selected = dialog.findViewById<RadioButton>(choices.checkedRadioButtonId)
                ?.tag as? String
            if (selected == null) {
                Toast.makeText(this, "Choose the correct label first.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val currentPrediction = predictions[currentIndex]

            learningEngine.recordEvaluation(
                predictedLabel = currentPrediction.label,
                correctLabel = selected,
                imageName = savedFile.name,
                sourceBitmap = cleanBitmap,
                detection = currentPrediction
            )
            savedExamples++
            try {
                trainingDataManager.saveConfirmedExample(
                    sourceBitmap = cleanBitmap,
                    prediction = currentPrediction,
                    correctLabel = selected,
                    labels = categoryLabels,
                    sourceImageName = savedFile.name
                )
            } catch (error: Exception) {
                trainingError = error.message ?: "unknown storage error"
            }
            advanceOrFinish()
        }

        dialog.setOnDismissListener {
            val reevaluated = learningEngine.applyCorrections(cleanBitmap, rawDetections)
            saveCaptureMetadata(savedFile, reevaluated, latencyMs)
            updateLearningButton()
            updateTrainingExportButton()
            if (savedExamples > 0) {
                Toast.makeText(
                    this,
                    if (trainingError == null) {
                        "$savedExamples visual example${if (savedExamples == 1) "" else "s"} learned and added to training data."
                    } else {
                        "Corrections learned, but training export storage failed: $trainingError"
                    },
                    Toast.LENGTH_LONG
                ).show()
            }
            displayedCrop?.let { crop ->
                if (crop !== cleanBitmap && !crop.isRecycled) crop.recycle()
            }
            if (!cleanBitmap.isRecycled) cleanBitmap.recycle()
            if (activeEvaluationDialog === dialog) activeEvaluationDialog = null
        }
        activeEvaluationDialog = dialog
        showCurrentPrediction()
        dialog.show()
        dialog.window?.setLayout(
            (resources.displayMetrics.widthPixels * 0.92f).toInt(),
            WindowManager.LayoutParams.WRAP_CONTENT
        )
    }

    private fun cropForEvaluation(source: Bitmap, detection: Detection): Bitmap {
        val paddingX = detection.box.width() * 0.20f
        val paddingY = detection.box.height() * 0.20f
        val left = (detection.box.left - paddingX).toInt().coerceIn(0, source.width - 1)
        val top = (detection.box.top - paddingY).toInt().coerceIn(0, source.height - 1)
        val right = (detection.box.right + paddingX).toInt().coerceIn(left + 1, source.width)
        val bottom = (detection.box.bottom + paddingY).toInt().coerceIn(top + 1, source.height)
        return Bitmap.createBitmap(source, left, top, right - left, bottom - top)
    }

    private fun updateLearningButton() {
        val stats = learningEngine.stats()
        btnLearningMemory.text =
            "Learning memory • ${stats.visualExamples} phone • ${stats.bundledSeedExamples} built-in"
    }

    private fun updateTrainingExportButton() {
        val samples = trainingDataManager.sampleCount()
        btnExportTraining.text = "Export Training ZIP • $samples confirmed samples"
    }

    private fun exportTrainingData() {
        val samples = trainingDataManager.sampleCount()
        if (samples == 0) {
            Toast.makeText(
                this,
                "No confirmed samples yet. Capture a bacterium and tap Teach app first.",
                Toast.LENGTH_LONG
            ).show()
            return
        }
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        createTrainingExport.launch("BACTERIALSCOPE_Training_$timestamp.zip")
    }

    private fun showLearningMemoryDialog() {
        val stats = learningEngine.stats()
        val reviewLabels = learningEngine.unreliableLabelNames()
        val trainingSamples = trainingDataManager.sampleCount()
        val reviewText = if (reviewLabels.isNotEmpty()) {
            "Review: ${reviewLabels.joinToString()}\n"
        } else {
            ""
        }
        AlertDialog.Builder(this)
            .setTitle("Local learning memory")
            .setMessage(
                    "${stats.evaluations} post-evaluations saved\n" +
                    "${stats.visualExamples} learned examples on this phone\n" +
                    "${stats.bundledSeedExamples} built-in examples from the imported ZIP\n" +
                    "${stats.learnedRules} label suggestions\n" +
                    "${stats.unreliableLabels} raw labels flagged for review\n" +
                    reviewText +
                    "$trainingSamples exportable YOLO training samples\n\n" +
                    "Visual corrections are applied only when nearby learned examples agree. " +
                    "Frequently corrected raw labels are marked REVIEW instead of being forced onto other samples. " +
                    "Corrections remain on this device. " +
                    "The original TFLite weights are not modified. Resetting learning rules does not delete " +
                    "the confirmed training samples."
            )
            .setNegativeButton("Close", null)
            .setPositiveButton("Reset learning") { _, _ ->
                learningEngine.reset()
                overlayView.clear()
                updateLearningButton()
                Toast.makeText(this, "Local learning memory reset.", Toast.LENGTH_SHORT).show()
            }
            .show()
    }

    private fun toggleFlashDrawer() {
        if (flashDrawer.visibility == View.VISIBLE) {
            hideFlashDrawer()
        } else {
            flashDrawer.visibility = View.VISIBLE
            flashDrawer.alpha = 0f
            flashDrawer.translationY = -dp(12).toFloat()
            flashDrawer.animate().alpha(1f).translationY(0f).setDuration(180L).start()
        }
        btnFlashDrawerToggle.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
    }

    private fun hideFlashDrawer() {
        if (flashDrawer.visibility != View.VISIBLE) return
        flashDrawer.animate().cancel()
        flashDrawer.animate()
            .alpha(0f)
            .translationY(-dp(10).toFloat())
            .setDuration(140L)
            .withEndAction {
                flashDrawer.visibility = View.GONE
                flashDrawer.alpha = 1f
                flashDrawer.translationY = 0f
            }
            .start()
    }

    private fun setFlashEnabled(enabled: Boolean) {
        val activeCamera = camera
        if (activeCamera == null || !activeCamera.cameraInfo.hasFlashUnit()) {
            Toast.makeText(this, "Flash is not available on this camera.", Toast.LENGTH_SHORT).show()
            hideFlashDrawer()
            return
        }
        torchEnabled = enabled
        activeCamera.cameraControl.enableTorch(enabled)
        btnFlashDrawerToggle.text = if (enabled) "Flash on" else "Flash"
        btnFlashDrawerToggle.performHapticFeedback(HapticFeedbackConstants.CONTEXT_CLICK)
        hideFlashDrawer()
    }

    private fun setupCameraGestures() {
        cameraScaleDetector = ScaleGestureDetector(
            this,
            object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScaleBegin(detector: ScaleGestureDetector): Boolean {
                    cameraDidPinch = true
                    return true
                }

                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    val activeCamera = camera ?: return false
                    val state = activeCamera.cameraInfo.zoomState.value ?: return false
                    val ratio = (state.zoomRatio * detector.scaleFactor)
                        .coerceIn(state.minZoomRatio, state.maxZoomRatio)
                    activeCamera.cameraControl.setZoomRatio(ratio)
                    btnZoom.text = String.format(Locale.US, "%.1f×", ratio)
                    return true
                }
            }
        )

        previewView.setOnTouchListener { _, event ->
            if (galleryScreen.visibility == View.VISIBLE) return@setOnTouchListener false
            cameraScaleDetector.onTouchEvent(event)
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    cameraDidPinch = false
                    true
                }
                MotionEvent.ACTION_POINTER_DOWN -> {
                    cameraDidPinch = true
                    true
                }
                MotionEvent.ACTION_UP -> {
                    if (!cameraDidPinch && event.pointerCount == 1) {
                        focusAt(event.x, event.y)
                    }
                    cameraDidPinch = false
                    true
                }
                MotionEvent.ACTION_CANCEL -> {
                    cameraDidPinch = false
                    true
                }
                else -> true
            }
        }
    }

    private fun focusAt(x: Float, y: Float) {
        val activeCamera = camera ?: return
        val point = previewView.meteringPointFactory.createPoint(x, y)
        val action = FocusMeteringAction.Builder(
            point,
            FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE
        ).setAutoCancelDuration(3, TimeUnit.SECONDS).build()
        activeCamera.cameraControl.startFocusAndMetering(action)
        showFocusPulse(x, y)
        previewView.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK)
    }

    private fun showFocusPulse(x: Float, y: Float) {
        val size = dp(74)
        val pulse = View(this).apply {
            layoutParams = FrameLayout.LayoutParams(size, size).apply {
                leftMargin = (x - size / 2f).toInt().coerceIn(0, (cameraScreen.width - size).coerceAtLeast(0))
                topMargin = (y - size / 2f).toInt().coerceIn(0, (cameraScreen.height - size).coerceAtLeast(0))
            }
            background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_focus_ring)
            alpha = 0f
            scaleX = 1.35f
            scaleY = 1.35f
            elevation = dp(20).toFloat()
        }
        cameraScreen.addView(pulse)
        pulse.animate()
            .alpha(1f)
            .scaleX(1f)
            .scaleY(1f)
            .setDuration(150L)
            .withEndAction {
                pulse.animate()
                    .alpha(0f)
                    .scaleX(0.82f)
                    .scaleY(0.82f)
                    .setStartDelay(480L)
                    .setDuration(220L)
                    .withEndAction { cameraScreen.removeView(pulse) }
                    .start()
            }
            .start()
    }

    private fun showGalleryScreen() {
        isLive = false
        detectionStabilizer.clear()
        hideFlashDrawer()
        mainHandler.removeCallbacks(dismissCaptureNotice)
        resultCard.visibility = View.GONE
        overlayView.clear()
        cameraScreen.visibility = View.GONE
        galleryScreen.visibility = View.VISIBLE
        galleryInfoPanel.visibility = View.GONE
        galleryScreen.alpha = 0f
        galleryScreen.translationY = dp(18).toFloat()
        galleryScreen.animate().alpha(1f).translationY(0f).setDuration(220L).start()
        loadGallery()
    }

    private fun showCameraScreen() {
        galleryLoadRequest++
        galleryScreen.visibility = View.GONE
        cameraScreen.visibility = View.VISIBLE
        isLive = true
        detectionStabilizer.clear()
        btnLive.text = "Pause"
        refreshLiveChip()
        resetGalleryTransform()
    }

    private fun toggleGalleryInfo() {
        if (galleryInfoPanel.visibility == View.VISIBLE) {
            galleryInfoPanel.animate().alpha(0f).translationY(-dp(10).toFloat()).setDuration(140L)
                .withEndAction {
                    galleryInfoPanel.visibility = View.GONE
                    galleryInfoPanel.alpha = 1f
                    galleryInfoPanel.translationY = 0f
                }.start()
        } else {
            galleryInfoPanel.visibility = View.VISIBLE
            galleryInfoPanel.alpha = 0f
            galleryInfoPanel.translationY = -dp(10).toFloat()
            galleryInfoPanel.animate().alpha(1f).translationY(0f).setDuration(180L).start()
        }
    }

    private fun loadGallery() {
        updateTrainingExportButton()
        galleryFiles = captureFiles()
        galleryStrip.removeAllViews()
        galleryIndex = -1
        txtGalleryEmpty.visibility = if (galleryFiles.isEmpty()) View.VISIBLE else View.GONE
        imgGallery.visibility = if (galleryFiles.isEmpty()) View.GONE else View.VISIBLE
        txtGallerySubtitle.text = if (galleryFiles.isEmpty()) "No saved scans" else "${galleryFiles.size} saved scans"
        txtGalleryInfo.text = "No scan selected"

        galleryFiles.forEachIndexed { index, file ->
            val thumbnail = ImageView(this).apply {
                layoutParams = LinearLayout.LayoutParams(dp(70), dp(64)).apply {
                    marginStart = dp(4)
                    marginEnd = dp(4)
                }
                scaleType = ImageView.ScaleType.CENTER_CROP
                setPadding(dp(3), dp(3), dp(3), dp(3))
                background = ContextCompat.getDrawable(this@MainActivity, R.drawable.bg_thumb)
                setOnClickListener { selectGalleryImage(index) }
            }
            galleryStrip.addView(thumbnail)
            galleryExecutor.execute {
                val bitmap = decodeSampledBitmap(file, dp(160), dp(160))
                runOnUiThread { if (thumbnail.parent != null && bitmap != null) thumbnail.setImageBitmap(bitmap) }
            }
        }

        if (galleryFiles.isNotEmpty()) selectGalleryImage(0)
    }

    private fun selectGalleryImage(index: Int) {
        if (index !in galleryFiles.indices) return
        galleryIndex = index
        val file = galleryFiles[index]
        resetGalleryTransform()

        repeat(galleryStrip.childCount) { childIndex ->
            galleryStrip.getChildAt(childIndex).background = ContextCompat.getDrawable(
                this,
                if (childIndex == index) R.drawable.bg_thumb_selected else R.drawable.bg_thumb
            )
        }
        val selectedThumb = galleryStrip.getChildAt(index)
        galleryScroll.post {
            galleryScroll.smoothScrollTo(
                (selectedThumb.left - (galleryScroll.width - selectedThumb.width) / 2).coerceAtLeast(0),
                0
            )
        }

        val metadata = readCaptureMetadata(file)
        val bestTotal = galleryFiles.mapNotNull { readCaptureMetadata(it)?.total }.maxOrNull() ?: 0
        val best = metadata != null && metadata.total > 0 && metadata.total == bestTotal
        txtGallerySubtitle.text = "Scan ${index + 1} of ${galleryFiles.size}${if (best) " • Best count" else ""}"
        txtGalleryInfo.text = galleryInfoText(file, metadata)
        txtGalleryEmpty.visibility = View.GONE
        imgGallery.visibility = View.VISIBLE

        val request = ++galleryLoadRequest
        val targetWidth = resources.displayMetrics.widthPixels
        val targetHeight = resources.displayMetrics.heightPixels
        galleryExecutor.execute {
            val bitmap = decodeSampledBitmap(file, targetWidth, targetHeight)
            runOnUiThread {
                if (request != galleryLoadRequest || galleryIndex != index || bitmap == null) return@runOnUiThread
                val old = (imgGallery.drawable as? BitmapDrawable)?.bitmap
                imgGallery.alpha = 0f
                imgGallery.setImageBitmap(bitmap)
                imgGallery.animate().alpha(1f).setDuration(150L).start()
                if (old != null && old !== bitmap && !old.isRecycled) old.recycle()
            }
        }
    }

    private fun galleryInfoText(file: File, metadata: CaptureMetadata?): String {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        val taken = SimpleDateFormat("MMM d, yyyy • h:mm a", Locale.US).format(Date(file.lastModified()))
        return buildString {
            appendLine(file.name)
            appendLine(taken)
            appendLine("${bounds.outWidth} × ${bounds.outHeight} • ${formatBytes(file.length())}")
            appendLine()
            appendLine("Total detections: ${metadata?.total ?: "—"}")
            val detectedClasses = categoryLabels.mapNotNull { label ->
                val count = metadata?.counts?.get(label) ?: 0
                if (count > 0) "$label: $count" else null
            }
            if (detectedClasses.isEmpty()) {
                appendLine("Classes: none recorded")
            } else {
                detectedClasses.forEach(::appendLine)
            }
            metadata?.threshold?.let { appendLine("Confidence: ${(it * 100).toInt()}%") }
            metadata?.latencyMs?.let { append("Inference: $it ms") }
        }.trim()
    }

    private fun setupGalleryGestures() {
        galleryScaleDetector = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(detector: ScaleGestureDetector): Boolean {
                galleryScale = (galleryScale * detector.scaleFactor).coerceIn(1f, 5f)
                if (galleryScale == 1f) {
                    galleryTranslationX = 0f
                    galleryTranslationY = 0f
                }
                applyGalleryTransform()
                return true
            }
        })

        imgGallery.setOnTouchListener { _, event ->
            galleryScaleDetector.onTouchEvent(event)
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    galleryDownX = event.x
                    galleryLastX = event.x
                    galleryLastY = event.y
                }
                MotionEvent.ACTION_MOVE -> if (!galleryScaleDetector.isInProgress && galleryScale > 1f) {
                    galleryTranslationX += event.x - galleryLastX
                    galleryTranslationY += event.y - galleryLastY
                    galleryLastX = event.x
                    galleryLastY = event.y
                    clampGalleryTranslation()
                    applyGalleryTransform()
                }
                MotionEvent.ACTION_UP -> if (galleryScale <= 1.02f) {
                    val distance = event.x - galleryDownX
                    if (abs(distance) > dp(70)) {
                        if (distance < 0) selectGalleryImage((galleryIndex + 1).coerceAtMost(galleryFiles.lastIndex))
                        else selectGalleryImage((galleryIndex - 1).coerceAtLeast(0))
                    }
                }
            }
            true
        }
    }

    private fun clampGalleryTranslation() {
        val maxX = imgGallery.width * (galleryScale - 1f) / 2f
        val maxY = imgGallery.height * (galleryScale - 1f) / 2f
        galleryTranslationX = galleryTranslationX.coerceIn(-maxX, maxX)
        galleryTranslationY = galleryTranslationY.coerceIn(-maxY, maxY)
    }

    private fun applyGalleryTransform() {
        imgGallery.scaleX = galleryScale
        imgGallery.scaleY = galleryScale
        imgGallery.translationX = galleryTranslationX
        imgGallery.translationY = galleryTranslationY
    }

    private fun resetGalleryTransform() {
        galleryScale = 1f
        galleryTranslationX = 0f
        galleryTranslationY = 0f
        applyGalleryTransform()
    }

    private fun saveSelectedToPhotos() {
        val file = galleryFiles.getOrNull(galleryIndex) ?: run {
            Toast.makeText(this, "No scan selected.", Toast.LENGTH_SHORT).show()
            return
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), STORAGE_PERMISSION_REQUEST)
            Toast.makeText(this, "Allow storage access, then tap Save to Photos again.", Toast.LENGTH_LONG).show()
            return
        }

        try {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, file.name)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/BacterialScope")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: error("Could not create Photos entry")
            contentResolver.openOutputStream(uri)?.use { output -> file.inputStream().use { it.copyTo(output) } }
                ?: error("Could not open Photos output")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                contentResolver.update(uri, values, null, null)
            }
            Toast.makeText(this, "Saved to Pictures/BacterialScope.", Toast.LENGTH_SHORT).show()
        } catch (error: Exception) {
            Toast.makeText(this, "Save failed: ${error.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun confirmDeleteSelected() {
        val file = galleryFiles.getOrNull(galleryIndex) ?: return
        AlertDialog.Builder(this)
            .setTitle("Delete this scan?")
            .setMessage("This removes ${file.name} and its detection record.")
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Delete") { _, _ ->
                val deleted = file.delete()
                File(file.absolutePath + ".txt").delete()
                Toast.makeText(this, if (deleted) "Scan deleted." else "Delete failed.", Toast.LENGTH_SHORT).show()
                if (deleted) loadGallery()
            }
            .show()
    }

    private fun exportGalleryCsv() {
        val files = captureFiles()
        if (files.isEmpty()) {
            Toast.makeText(this, "No scans to export.", Toast.LENGTH_SHORT).show()
            return
        }
        try {
            val directory = File(
                getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: filesDir,
                "BacterialScope_Exports"
            ).apply { mkdirs() }
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val csv = File(directory, "BACTERIALSCOPE_Summary_$timestamp.csv")
            csv.writeText(buildString {
                appendCsvRow(*(
                    listOf("file", "taken", "total") +
                        categoryLabels +
                        listOf("confidence", "latency_ms")
                    ).toTypedArray())
                files.forEach { file ->
                    val data = readCaptureMetadata(file)
                    appendCsvRow(*(
                        listOf(
                            file.name,
                            SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date(file.lastModified())),
                            data?.total?.toString().orEmpty()
                        ) + categoryLabels.map { label ->
                            (data?.counts?.get(label) ?: 0).toString()
                        } + listOf(
                            data?.threshold?.toString().orEmpty(),
                            data?.latencyMs?.toString().orEmpty()
                        )
                    ).toTypedArray())
                }
            })
            Toast.makeText(this, "CSV exported: ${csv.name}", Toast.LENGTH_LONG).show()
        } catch (error: Exception) {
            Toast.makeText(this, "CSV export failed: ${error.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun StringBuilder.appendCsvRow(vararg values: String) {
        appendLine(values.joinToString(",") { "\"${it.replace("\"", "\"\"")}\"" })
    }

    private fun drawDetections(source: Bitmap, detections: List<Detection>): Bitmap {
        if (detections.isEmpty()) return source
        val bitmap = if (source.isMutable) source else source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(bitmap)
        val scale = bitmap.width / 1080f
        val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(4f, 5f * scale)
        }
        val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
        val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = max(26f, 30f * scale)
            typeface = android.graphics.Typeface.DEFAULT_BOLD
        }

        detections.forEach { detection ->
            val accent = if (detection.reviewRecommended) {
                Color.rgb(255, 190, 70)
            } else {
                accentForLabel(detection.label)
            }
            stroke.color = accent
            fill.color = Color.argb(235, 6, 16, 31)
            canvas.drawRoundRect(detection.box, 18f * scale, 18f * scale, stroke)
            val statusText = when {
                detection.locallyCorrected -> " • LEARNED"
                detection.reviewRecommended -> " • REVIEW"
                else -> ""
            }
            val label = "${detection.label} ${(detection.confidence * 100).toInt()}%$statusText"
            val padding = 12f * scale
            val height = text.textSize + padding * 1.5f
            val width = text.measureText(label) + padding * 2
            val top = (detection.box.top - height).coerceAtLeast(0f)
            val chip = RectF(detection.box.left, top, detection.box.left + width, top + height)
            canvas.drawRoundRect(chip, height / 2, height / 2, fill)
            canvas.drawCircle(chip.left + padding, chip.centerY(), 3.5f * scale, Paint(Paint.ANTI_ALIAS_FLAG).apply { color = accent })
            canvas.drawText(label, chip.left + padding * 1.8f, chip.bottom - padding * 0.65f, text)
        }
        if (bitmap !== source) source.recycle()
        return bitmap
    }

    private fun saveBitmap(bitmap: Bitmap): File {
        val directory = currentCaptureDirectory().apply { mkdirs() }
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val file = File(directory, "BACTERIAL_$timestamp.jpg")
        FileOutputStream(file).use { stream ->
            check(bitmap.compress(Bitmap.CompressFormat.JPEG, 96, stream)) { "JPEG encoding failed" }
        }
        return file
    }

    private fun saveCaptureMetadata(file: File, detections: List<Detection>, latencyMs: Long?) {
        val counts = detections.groupingBy(Detection::label).eachCount()
        File(file.absolutePath + ".txt").writeText(buildString {
            appendLine("total=${detections.size}")
            categoryLabels.forEach { label ->
                appendLine("$label=${counts[label] ?: 0}")
            }
            appendLine("threshold=$threshold")
            latencyMs?.let { appendLine("latency_ms=$it") }
            appendLine("taken_epoch=${System.currentTimeMillis()}")
        })
    }

    private data class CaptureMetadata(
        val total: Int,
        val counts: Map<String, Int>,
        val threshold: Float?,
        val latencyMs: Long?
    )

    private fun readCaptureMetadata(file: File): CaptureMetadata? {
        return try {
            val metadata = File(file.absolutePath + ".txt")
            if (!metadata.exists()) return null
            val values = metadata.readLines().mapNotNull { line ->
                val parts = line.split("=", limit = 2)
                if (parts.size == 2) parts[0].trim() to parts[1].trim() else null
            }.toMap()
            CaptureMetadata(
                total = values["total"]?.toIntOrNull() ?: 0,
                counts = categoryLabels.associateWith { label ->
                    values[label]?.toIntOrNull() ?: 0
                },
                threshold = values["threshold"]?.toFloatOrNull(),
                latencyMs = values["latency_ms"]?.toLongOrNull()
            )
        } catch (_: Exception) {
            null
        }
    }

    private fun currentCaptureDirectory(): File {
        return File(getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: filesDir, "BacterialScope")
    }

    private fun captureFiles(): List<File> {
        val base = getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: filesDir
        val directories = listOf(File(base, "BacterialScope"), File(base, "FungiScope"))
        return directories.flatMap { directory ->
            directory.listFiles { file -> file.isFile && file.extension.lowercase() in listOf("jpg", "jpeg", "png") }
                ?.toList().orEmpty()
        }.distinctBy(File::getAbsolutePath).sortedByDescending(File::lastModified)
    }

    private fun decodeSampledBitmap(file: File, targetWidth: Int, targetHeight: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        var sample = 1
        while (bounds.outWidth / sample > targetWidth * 2 || bounds.outHeight / sample > targetHeight * 2) sample *= 2
        return BitmapFactory.decodeFile(file.absolutePath, BitmapFactory.Options().apply { inSampleSize = sample })
    }

    private fun formatBytes(bytes: Long): String {
        return when {
            bytes >= 1_048_576 -> String.format(Locale.US, "%.1f MB", bytes / 1_048_576f)
            bytes >= 1024 -> String.format(Locale.US, "%.0f KB", bytes / 1024f)
            else -> "$bytes B"
        }
    }

    private fun accentForLabel(label: String): Int {
        val palette = intArrayOf(
            Color.rgb(85, 221, 246),
            Color.rgb(167, 123, 255),
            Color.rgb(95, 166, 255),
            Color.rgb(198, 247, 255),
            Color.rgb(255, 119, 190),
            Color.rgb(91, 239, 190)
        )
        return palette[label.hashCode().and(Int.MAX_VALUE) % palette.size]
    }

    private fun ImageProxy.toBitmapSmart(): Bitmap {
        return when (format) {
            ImageFormat.JPEG -> {
                val buffer = planes[0].buffer.apply { rewind() }
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                requireNotNull(BitmapFactory.decodeByteArray(bytes, 0, bytes.size))
            }
            ImageFormat.YUV_420_888 -> {
                val yuv = YuvImage(yuv420ToNv21(), ImageFormat.NV21, width, height, null)
                val output = ByteArrayOutputStream()
                yuv.compressToJpeg(Rect(0, 0, width, height), 96, output)
                val bytes = output.toByteArray()
                requireNotNull(BitmapFactory.decodeByteArray(bytes, 0, bytes.size))
            }
            PixelFormat.RGBA_8888 -> {
                val plane = planes[0]
                val buffer = plane.buffer.apply { rewind() }
                val paddedWidth = width + (plane.rowStride - plane.pixelStride * width) / plane.pixelStride
                val padded = Bitmap.createBitmap(paddedWidth, height, Bitmap.Config.ARGB_8888)
                padded.copyPixelsFromBuffer(buffer)
                if (paddedWidth == width) padded else {
                    Bitmap.createBitmap(padded, 0, 0, width, height).also { padded.recycle() }
                }
            }
            else -> error("Unsupported camera format: $format")
        }
    }

    private fun ImageProxy.yuv420ToNv21(): ByteArray {
        val crop = cropRect
        val outputWidth = crop.width()
        val outputHeight = crop.height()
        val data = ByteArray(outputWidth * outputHeight * ImageFormat.getBitsPerPixel(format) / 8)
        val rowData = ByteArray(planes[0].rowStride)

        planes.forEachIndexed { planeIndex, plane ->
            var channelOffset = when (planeIndex) {
                0 -> 0
                1 -> outputWidth * outputHeight + 1
                2 -> outputWidth * outputHeight
                else -> return@forEachIndexed
            }
            val outputStride = if (planeIndex == 0) 1 else 2
            val shift = if (planeIndex == 0) 0 else 1
            val planeWidth = outputWidth shr shift
            val planeHeight = outputHeight shr shift
            val buffer = plane.buffer
            buffer.position(plane.rowStride * (crop.top shr shift) + plane.pixelStride * (crop.left shr shift))

            repeat(planeHeight) { row ->
                val length: Int
                if (plane.pixelStride == 1 && outputStride == 1) {
                    length = planeWidth
                    buffer.get(data, channelOffset, length)
                    channelOffset += length
                } else {
                    length = (planeWidth - 1) * plane.pixelStride + 1
                    buffer.get(rowData, 0, length)
                    repeat(planeWidth) { column ->
                        data[channelOffset] = rowData[column * plane.pixelStride]
                        channelOffset += outputStride
                    }
                }
                if (row < planeHeight - 1) buffer.position(buffer.position() + plane.rowStride - length)
            }
        }
        return data
    }

    private fun Bitmap.rotate(degrees: Int): Bitmap {
        if (degrees == 0) return this
        val rotated = Bitmap.createBitmap(
            this, 0, 0, width, height,
            Matrix().apply { postRotate(degrees.toFloat()) }, true
        )
        if (rotated !== this) recycle()
        return rotated
    }

    private fun View.applySoftPress() {
        setOnTouchListener { view, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> view.animate()
                    .scaleX(0.96f).scaleY(0.96f).alpha(0.84f).setDuration(80L).start()
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> view.animate()
                    .scaleX(1f).scaleY(1f).alpha(1f).setDuration(130L).start()
            }
            false
        }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    override fun onDestroy() {
        mainHandler.removeCallbacksAndMessages(null)
        activeEvaluationDialog?.dismiss()
        detector?.close()
        cameraExecutor.shutdown()
        galleryExecutor.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val MODEL_ASSET = "bacterial_model.tflite"
        private const val LABEL_ASSET = "bacterial_labels.txt"
        private const val ANALYSIS_INTERVAL_MS = 180L
        private const val MAX_EVALUATIONS_PER_CAPTURE = 6
        private const val STORAGE_PERMISSION_REQUEST = 220
    }
}
