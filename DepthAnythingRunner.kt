package com.example.blurslider

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class DepthAnythingRunner(private val context: Context) {
    companion object {
        private const val MODEL_NAME = "model_fp16.onnx"
        private const val INPUT_SIZE = 518
        private const val TAG = "DepthRunner"

        private const val MEAN_R = 0.485f
        private const val MEAN_G = 0.456f
        private const val MEAN_B = 0.406f

        private const val STD_R = 0.229f
        private const val STD_G = 0.224f
        private const val STD_B = 0.225f
    }

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    @Synchronized
    private fun getSession(): OrtSession {
        if (ortSession == null) {
            ortEnv = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
            }
            context.assets.open(MODEL_NAME).use { asset ->
                val bytes = asset.readBytes()
                Log.d(TAG, "Loaded ONNX model: ${bytes.size} bytes")
                ortSession = ortEnv!!.createSession(bytes, options)
            }
        }
        return ortSession!!
    }

    fun run(input: Bitmap): Bitmap {
        val session = getSession()
        val prepared = resizeWithLetterbox(input, INPUT_SIZE, INPUT_SIZE)

        val area = INPUT_SIZE * INPUT_SIZE
        val inputFloatArray = FloatArray(1 * 3 * area)
        val pixels = IntArray(area)

        prepared.bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (i in 0 until area) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF

            inputFloatArray[i] = (r / 255f - MEAN_R) / STD_R
            inputFloatArray[area + i] = (g / 255f - MEAN_G) / STD_G
            inputFloatArray[2 * area + i] = (b / 255f - MEAN_B) / STD_B
        }

        val env = ortEnv ?: OrtEnvironment.getEnvironment()
        val inputShape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputFloatArray), inputShape)

        val inputName = session.inputNames.iterator().next()
        val results = session.run(mapOf(inputName to inputTensor))

        val outputTensor = results[0] as OnnxTensor
        val outputBuffer = outputTensor.floatBuffer
        val depth = FloatArray(outputBuffer.remaining())
        outputBuffer.get(depth)

        inputTensor.close()
        results.close()

        // العينات تحسب حصراً من المنطقة الفعلية للصورة بدون الهوامش السوداء
        val sampleSize = 2000
        val contentW = prepared.contentWidth
        val contentH = prepared.contentHeight
        val left = prepared.left
        val top = prepared.top

        val totalValidPixels = contentW * contentH
        val step = max(1, totalValidPixels / sampleSize)
        val sampled = FloatArray(sampleSize)
        var sampledCount = 0

        for (y in top until (top + contentH) step step) {
            val rowOffset = y * INPUT_SIZE
            for (x in left until (left + contentW) step step) {
                val idx = rowOffset + x
                if (sampledCount < sampleSize && idx < depth.size) {
                    sampled[sampledCount++] = depth[idx]
                }
            }
        }

        sampled.sort(0, sampledCount)
        val p5 = sampled[(sampledCount * 0.05f).toInt()]
        val p95 = sampled[(sampledCount * 0.95f).toInt()]
        val range = max(0.0001f, p95 - p5)

        val depthPixels = IntArray(area)
        for (i in 0 until area) {
            val n = ((depth[i] - p5) / range).coerceIn(0f, 1f)
            val g = (n * 255f).toInt()
            depthPixels[i] = (0xFF shl 24) or (g shl 16) or (g shl 8) or g
        }

        val depthSquare = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        depthSquare.setPixels(depthPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val cropped = Bitmap.createBitmap(
            depthSquare,
            prepared.left,
            prepared.top,
            prepared.contentWidth,
            prepared.contentHeight
        )

        val finalBitmap = Bitmap.createScaledBitmap(cropped, input.width, input.height, true)

        if (!prepared.bitmap.isRecycled) prepared.bitmap.recycle()
        if (!depthSquare.isRecycled) depthSquare.recycle()
        if (cropped != finalBitmap && !cropped.isRecycled) cropped.recycle()

        return finalBitmap
    }

    private data class PreparedBitmap(
        val bitmap: Bitmap,
        val left: Int,
        val top: Int,
        val contentWidth: Int,
        val contentHeight: Int
    )

    private fun resizeWithLetterbox(src: Bitmap, dstW: Int, dstH: Int): PreparedBitmap {
        val result = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888)
        val scale = min(dstW / src.width.toFloat(), dstH / src.height.toFloat())
        val w = (src.width * scale).toInt().coerceIn(1, dstW)
        val h = (src.height * scale).toInt().coerceIn(1, dstH)
        val left = (dstW - w) / 2
        val top = (dstH - h) / 2
        
        val scaled = Bitmap.createScaledBitmap(src, w, h, true)
        
        Canvas(result).apply {
            drawColor(Color.BLACK)
            drawBitmap(scaled, left.toFloat(), top.toFloat(), null)
        }

        if (scaled != src && !scaled.isRecycled) {
            scaled.recycle()
        }

        return PreparedBitmap(result, left, top, w, h)
    }

    fun close() {
        try {
            ortSession?.close()
            ortEnv?.close()
        } catch (_: Exception) { }
        ortSession = null
        ortEnv = null
    }
}
