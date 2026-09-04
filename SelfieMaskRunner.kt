package com.example.blurslider

import android.graphics.Bitmap
import android.graphics.Color
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.SegmentationMask
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions

class SelfieMaskRunner {
    private val segmenter = Segmentation.getClient(
        SelfieSegmenterOptions.Builder()
            .setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE)
            .enableRawSizeMask()
            .build()
    )

    fun process(
        bitmap: Bitmap,
        onSuccess: (Bitmap) -> Unit,
        onFailure: (Exception) -> Unit
    ) {
        val input = InputImage.fromBitmap(bitmap, 0)
        segmenter.process(input)
            .addOnSuccessListener { mask ->
                try {
                    onSuccess(toGrayscaleMask(mask))
                } catch (error: Exception) {
                    onFailure(error)
                }
            }
            .addOnFailureListener { error -> onFailure(error) }
    }

    private fun toGrayscaleMask(mask: SegmentationMask): Bitmap {
        val width = mask.width
        val height = mask.height
        val output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val buffer = mask.buffer
        buffer.rewind()
        val values = FloatArray(width * height)
        buffer.asFloatBuffer().get(values)
        val pixels = IntArray(values.size)
        for (i in values.indices) {
            val confidence = values[i].coerceIn(0f, 1f)
            val softened = ((confidence - 0.12f) / 0.76f).coerceIn(0f, 1f)
            val gray = (softened * 255f).toInt()
            pixels[i] = Color.rgb(gray, gray, gray)
        }
        output.setPixels(pixels, 0, width, 0, 0, width, height)
        return smoothMask(output)
    }

    private fun smoothMask(source: Bitmap): Bitmap {
        val width = source.width
        val height = source.height
        val input = IntArray(width * height)
        val output = IntArray(width * height)
        source.getPixels(input, 0, width, 0, 0, width, height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var sum = 0
                var weight = 0
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        val xx = (x + dx).coerceIn(0, width - 1)
                        val yy = (y + dy).coerceIn(0, height - 1)
                        val w = if (dx == 0 && dy == 0) 4 else 1
                        sum += Color.red(input[yy * width + xx]) * w
                        weight += w
                    }
                }
                val value = (sum / weight).coerceIn(0, 255)
                output[y * width + x] = Color.rgb(value, value, value)
            }
        }
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            setPixels(output, 0, width, 0, 0, width, height)
        }
    }

    fun close() {
        segmenter.close()
    }
}
