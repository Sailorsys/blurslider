package com.example.blurslider

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import com.google.android.material.slider.Slider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.max

class MainActivity : AppCompatActivity() {
    private var originalBitmap: Bitmap? = null
    private var depthBitmap: Bitmap? = null
    private var depthForGpu: Bitmap? = null
    private var maskForGpu: Bitmap? = null
    private var sourceForGpu: Bitmap? = null
    private var focusDepth: Float = 0.5f

    private lateinit var depthRunner: DepthAnythingRunner
    private lateinit var selfieRunner: SelfieMaskRunner

    private lateinit var imageView: ImageView
    private lateinit var depthView: ImageView
    private lateinit var depthBlurView: DepthBlurGLView
    private lateinit var btnPick: Button
    private lateinit var btnIsolate: Button

    private lateinit var blurSeekBar: Slider
    private lateinit var blurValueText: TextView

    private lateinit var bokehSizeSeekBar: Slider
    private lateinit var bokehSizeValueText: TextView

    private lateinit var lightThresholdSeekBar: Slider
    private lateinit var lightThresholdValueText: TextView

    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { loadImage(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        depthView = findViewById(R.id.depthView)
        depthBlurView = findViewById(R.id.depthBlurView)

        btnPick = findViewById(R.id.btnPick)
        btnIsolate = findViewById(R.id.btnIsolate)

        blurSeekBar = findViewById(R.id.blurSeekBar)
        blurValueText = findViewById(R.id.blurValueText)

        bokehSizeSeekBar = findViewById(R.id.bokehSizeSeekBar)
        bokehSizeValueText = findViewById(R.id.bokehSizeValueText)

        lightThresholdSeekBar = findViewById(R.id.lightThresholdSeekBar)
        lightThresholdValueText = findViewById(R.id.lightThresholdValueText)

        depthRunner = DepthAnythingRunner(this)
        selfieRunner = SelfieMaskRunner()

        btnPick.setOnClickListener { pickImageLauncher.launch("image/*") }
        btnIsolate.setOnClickListener { runIsolation() }

        setupSliders()

        depthBlurView.visibility = View.GONE
        imageView.visibility = View.VISIBLE
    }

    private fun setupSliders() {
        // 1. التحكم بقوة العزل (الأقصى عند 100% أصبح 0.50f بدلاً من 1.0f)
        blurSeekBar.addOnChangeListener { _, value, _ ->
            blurValueText.text = "${value.toInt()}%"

            // قوة البلور
            val mappedBlur = (value / 100f) * 0.75f
            depthBlurView.setBlurStrength(mappedBlur)

            // حجم البوكيه مربوط بالبلور (Power Curve)
            val t = value / 100f
            val curved = Math.pow(t.toDouble(), 0.85).toFloat()   // الأس 0.85 هو المنحنى المتفق عليه
            val mappedBokehSize = 20f + curved * 35f               // من حوالي 18 إلى 35

            depthBlurView.setMaxBokehSize(mappedBokehSize)
        }

        // 2. التحكم في تعزيز ألوان وتشبع البوكيه (باستخدام نفس السلايدر القديم bokehSizeSeekBar)
        bokehSizeSeekBar.addOnChangeListener { _, value, _ ->
            bokehSizeValueText.text = "${value.toInt()}%"
            // تحويل القيمة من 0.0f إلى 1.0f للتحكم في ألوان البوكيه
            val mappedColorBoost = value / 100f
            depthBlurView.setBokehColorBoost(mappedColorBoost)
        }

        // 3. التحكم بحساسية إضاءة البوكيه
        lightThresholdSeekBar.addOnChangeListener { _, value, _ ->
            lightThresholdValueText.text = "${value.toInt()}%"
            val mappedThreshold = 1.0f - (value / 100f) * 0.35f
            depthBlurView.setLightThreshold(mappedThreshold)
        }
    }

    private fun applyCurrentSliderValuesToGL() {
        val blurVal = blurSeekBar.value

        // البلور وحجم البوكيه
        depthBlurView.setBlurStrength((blurVal / 100f) * 0.75f)

        val t = blurVal / 100f
        val curved = Math.pow(t.toDouble(), 0.85).toFloat()
        val mappedBokehSize = 20f + curved * 35f
        depthBlurView.setMaxBokehSize(mappedBokehSize)

        // تطبيق لون البوكيه من السلايدر
        val bokehColorVal = bokehSizeSeekBar.value
        depthBlurView.setBokehColorBoost(bokehColorVal / 100f)

        // الـ Light Threshold
        val lightVal = lightThresholdSeekBar.value
        depthBlurView.setLightThreshold(1.0f - (lightVal / 100f) * 0.35f)
    }

    private fun loadImage(uri: Uri) {
        try {
            recycleAllBitmaps()

            contentResolver.openInputStream(uri)?.use { stream ->
                val decoded = BitmapFactory.decodeStream(stream) ?: throw IllegalArgumentException("Cannot decode image")

                val orientation = contentResolver.openInputStream(uri)?.use { exifStream ->
                    ExifInterface(exifStream).getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL
                    )
                } ?: ExifInterface.ORIENTATION_NORMAL

                val bmp = applyExifOrientation(decoded, orientation)
                if (bmp !== decoded && !decoded.isRecycled) decoded.recycle()

                originalBitmap = bmp
                Log.d("Main", "Image normalized: ${bmp.width}x${bmp.height}, exif=$orientation")
                imageView.setImageBitmap(bmp)
                imageView.visibility = View.VISIBLE
                depthBlurView.visibility = View.GONE
                depthView.setImageBitmap(null)

                blurSeekBar.value = 0f
            }
        } catch (e: Exception) {
            Log.e("Main", "loadImage failed", e)
            Toast.makeText(this, "فشل تحميل الصورة", Toast.LENGTH_SHORT).show()
        }
    }

    private fun applyExifOrientation(source: Bitmap, orientation: Int): Bitmap {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.setScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.setRotate(90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.setRotate(-90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(270f)
            else -> return source
        }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    private fun runIsolation() {
        val bitmap = originalBitmap ?: run {
            Toast.makeText(this, "اختار صورة الأول", Toast.LENGTH_SHORT).show()
            return
        }

        btnIsolate.isEnabled = false
        btnIsolate.text = "جاري العزل..."

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val depth518 = depthRunner.run(bitmap)
                val personMaskFull = selfieRunner.process(this@MainActivity, bitmap)

                withContext(Dispatchers.Main) {
                    recycleGpuBitmaps()

                    depthView.setImageBitmap(depth518)
                    depthBitmap = depth518

                    val gpuSize = scaleForGpu(bitmap)
                    sourceForGpu = Bitmap.createScaledBitmap(bitmap, gpuSize.first, gpuSize.second, true)
                    depthForGpu = Bitmap.createScaledBitmap(depth518, gpuSize.first, gpuSize.second, true)
                    maskForGpu = Bitmap.createScaledBitmap(personMaskFull, gpuSize.first, gpuSize.second, true)

                    focusDepth = computeFocusDepthFast(depthForGpu!!, maskForGpu!!)
                    Log.d("Main", "FocusDepth=$focusDepth")

                    depthBlurView.setFocusDepth(focusDepth)
                    depthBlurView.setBitmaps(
                        sourceForGpu!!,
                        depthForGpu!!,
                        maskForGpu!!
                    )

                    // ضبط القيم الأولية
                    if (blurSeekBar.value == 0f) {
                        blurSeekBar.value = 40f
                    }
                    applyCurrentSliderValuesToGL()

                    imageView.visibility = View.GONE
                    depthBlurView.visibility = View.VISIBLE
                }
            } catch (e: Exception) {
                Log.e("Main", "Isolation failed", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "فشل: ${e.message}", Toast.LENGTH_LONG).show()
                }
            } finally {
                withContext(Dispatchers.Main) {
                    btnIsolate.isEnabled = true
                    btnIsolate.text = "عزل"
                }
            }
        }
    }

    private fun scaleForGpu(bitmap: Bitmap): Pair<Int, Int> {
        val maxEdge = 1280
        val w = bitmap.width
        val h = bitmap.height
        return if (max(w, h) <= maxEdge) w to h
        else if (w > h) maxEdge to (h * maxEdge / w)
        else (w * maxEdge / h) to maxEdge
    }

    private fun computeFocusDepthFast(depth: Bitmap, mask: Bitmap): Float {
        val width = depth.width
        val height = depth.height
        val depthPixels = IntArray(width * height)
        val maskPixels = IntArray(width * height)

        depth.getPixels(depthPixels, 0, width, 0, 0, width, height)
        mask.getPixels(maskPixels, 0, width, 0, 0, width, height)

        val values = ArrayList<Float>()
        val step = 4

        for (y in 0 until height step step) {
            val rowOffset = y * width
            for (x in 0 until width step step) {
                val index = rowOffset + x
                if (Color.red(maskPixels[index]) > 170) {
                    values.add(Color.red(depthPixels[index]) / 255f)
                }
            }
        }

        if (values.isEmpty()) return 0.5f
        values.sort()
        return values[values.size / 2]
    }

    private fun recycleGpuBitmaps() {
        sourceForGpu?.takeIf { !it.isRecycled }?.recycle()
        depthForGpu?.takeIf { !it.isRecycled }?.recycle()
        maskForGpu?.takeIf { !it.isRecycled }?.recycle()
        sourceForGpu = null
        depthForGpu = null
        maskForGpu = null
    }

    private fun recycleAllBitmaps() {
        recycleGpuBitmaps()
        originalBitmap?.takeIf { !it.isRecycled }?.recycle()
        depthBitmap?.takeIf { !it.isRecycled }?.recycle()
        originalBitmap = null
        depthBitmap = null
    }

    override fun onPause() {
        super.onPause()
        depthBlurView.onPause()
    }

    override fun onResume() {
        super.onResume()
        depthBlurView.onResume()
    }

    override fun onDestroy() {
        super.onDestroy()
        recycleAllBitmaps()
        try { depthRunner.close() } catch (e: Exception) {}
        try { selfieRunner.close() } catch (e: Exception) {}
    }
}
