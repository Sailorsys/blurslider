package com.example.blurslider

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlin.math.max

class DepthAnythingRunner(private val context: Context) {

    companion object {
        private const val MODEL_NAME = "Depth-Anything-V2.tflite"   // أو الاسم اللي حطيته
        private const val INPUT_SIZE = 518
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    fun run(input: Bitmap): Bitmap {
        val model = CompiledModel.create(
            context.assets,
            MODEL_NAME,
            CompiledModel.Options(Accelerator.CPU),   // جرب GPU أولاً، لو فشل غيّره لـ CPU
            null
        )

        val resized = Bitmap.createScaledBitmap(input, INPUT_SIZE, INPUT_SIZE, true)

        // جرب أولاً NHWC (زي ما كان عندك في MiDaS)
        // NCHW layout: [1, 3, 518, 518]
val inputData = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
var idx = 0

// Channel R
for (y in 0 until INPUT_SIZE) {
    for (x in 0 until INPUT_SIZE) {
        val pixel = resized.getPixel(x, y)
        val r = Color.red(pixel) / 255f
        inputData[idx++] = (r - MEAN[0]) / STD[0]
    }
}
// Channel G
for (y in 0 until INPUT_SIZE) {
    for (x in 0 until INPUT_SIZE) {
        val pixel = resized.getPixel(x, y)
        val g = Color.green(pixel) / 255f
        inputData[idx++] = (g - MEAN[1]) / STD[1]
    }
}
// Channel B
for (y in 0 until INPUT_SIZE) {
    for (x in 0 until INPUT_SIZE) {
        val pixel = resized.getPixel(x, y)
        val b = Color.blue(pixel) / 255f
        inputData[idx++] = (b - MEAN[2]) / STD[2]
    }
}

        val inputs = model.createInputBuffers()
        val outputs = model.createOutputBuffers()
        inputs[0].writeFloat(inputData)
        model.run(inputs, outputs)

        val depth = outputs[0].readFloat()

        // ===== تحسين التطبيع + زيادة التباين =====
        val sorted = depth.sorted()
        val p5 = sorted[(sorted.size * 0.05).toInt()]
        val p95 = sorted[(sorted.size * 0.95).toInt()]
        val range = max(0.0001f, p95 - p5)

        val result = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        for (i in depth.indices) {
            val normalized = ((depth[i] - p5) / range).coerceIn(0f, 1f)
            // زيادة التباين
            val contrast = ((normalized - 0.5f) * 1.6f + 0.5f).coerceIn(0f, 1f)
            val gray = (contrast * 255f).toInt()
            result.setPixel(i % INPUT_SIZE, i / INPUT_SIZE, Color.rgb(gray, gray, gray))
        }

        return result
    }
}