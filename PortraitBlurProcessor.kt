package com.example.blurslider

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.max

object PortraitBlurProcessor {
    private const val PREVIEW_MAX_EDGE = 640
    private const val DEPTH_SIZE = 518

    class Prepared(
        private val preview: Bitmap,
        private val blurred: Bitmap,
        private val depthMap: Bitmap,
        private val personMask: Bitmap
    ) {
        private val focusDepth: Float = estimateFocusDepth(depthMap)

        fun render(strength: Int): Bitmap {
            val output = Bitmap.createBitmap(preview.width, preview.height, Bitmap.Config.ARGB_8888)
            val maxBlur = (strength / 25f).coerceIn(0f, 1f)
            val maskWidth = personMask.width
            val maskHeight = personMask.height

            for (y in 0 until preview.height) {
                val depthY = (y * DEPTH_SIZE / preview.height).coerceIn(0, DEPTH_SIZE - 1)
                val maskY = (y * maskHeight / preview.height).coerceIn(0, maskHeight - 1)
                for (x in 0 until preview.width) {
                    val depthX = (x * DEPTH_SIZE / preview.width).coerceIn(0, DEPTH_SIZE - 1)
                    val maskX = (x * maskWidth / preview.width).coerceIn(0, maskWidth - 1)
                    val depth = Color.red(depthMap.getPixel(depthX, depthY)) / 255f
                    val personConfidence = Color.red(personMask.getPixel(maskX, maskY)) / 255f
                    val relativeBackground = ((focusDepth - depth + 0.08f) / 0.35f).coerceIn(0f, 1f)
                    val depthBackground = smoothStep(0.05f, 0.85f, relativeBackground)
                    val maskBackground = 1f - smoothStep(0.18f, 0.72f, personConfidence)
                    val background = max(depthBackground * 0.65f, maskBackground)
                    val blurWeight = max(0.015f, background) * maxBlur
                    output.setPixel(
                        x,
                        y,
                        blend(preview.getPixel(x, y), blurred.getPixel(x, y), blurWeight)
                    )
                }
            }
            return output
        }
    }

    fun prepare(source: Bitmap, depthMap: Bitmap, personMask: Bitmap): Prepared {
        val preview = scaleDown(source, PREVIEW_MAX_EDGE)
        val blurred = makeFastBlur(preview)
        return Prepared(preview, blurred, depthMap, personMask)
    }

    private fun estimateFocusDepth(depthMap: Bitmap): Float {
        var total = 0f
        var count = 0
        val left = DEPTH_SIZE / 4
        val right = DEPTH_SIZE * 3 / 4
        val top = DEPTH_SIZE / 8
        val bottom = DEPTH_SIZE * 3 / 4
        for (y in top until bottom) {
            for (x in left until right) {
                total += Color.red(depthMap.getPixel(x, y)) / 255f
                count++
            }
        }
        return if (count == 0) 0.5f else total / count
    }

    private fun scaleDown(source: Bitmap, maxEdge: Int): Bitmap {
        val largest = max(source.width, source.height)
        if (largest <= maxEdge) return source
        val scale = maxEdge.toFloat() / largest.toFloat()
        return Bitmap.createScaledBitmap(
            source,
            (source.width * scale).toInt().coerceAtLeast(1),
            (source.height * scale).toInt().coerceAtLeast(1),
            true
        )
    }

    private fun makeFastBlur(source: Bitmap): Bitmap {
        val factor = 0.18f
        val smallWidth = max(1, (source.width * factor).toInt())
        val smallHeight = max(1, (source.height * factor).toInt())
        val small = Bitmap.createScaledBitmap(source, smallWidth, smallHeight, true)
        return Bitmap.createScaledBitmap(small, source.width, source.height, true)
    }

    private fun smoothStep(edge0: Float, edge1: Float, value: Float): Float {
        val t = ((value - edge0) / (edge1 - edge0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private fun blend(first: Int, second: Int, amount: Float): Int {
        val t = amount.coerceIn(0f, 1f)
        val r = (Color.red(first) * (1f - t) + Color.red(second) * t).toInt()
        val g = (Color.green(first) * (1f - t) + Color.green(second) * t).toInt()
        val b = (Color.blue(first) * (1f - t) + Color.blue(second) * t).toInt()
        return Color.rgb(r, g, b)
    }
}
