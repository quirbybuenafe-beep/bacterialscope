package com.fungiscope.detector

import android.graphics.RectF

data class Detection(
    val label: String,
    val confidence: Float,
    val box: RectF,
    val locallyCorrected: Boolean = false,
    val reviewRecommended: Boolean = false
)
