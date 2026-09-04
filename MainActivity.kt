package com.example.blurslider

import android.graphics.Bitmap
import android.graphics.RenderEffect
import android.graphics.Shader
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.drawable.toBitmap
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var depthView: ImageView
    private lateinit var emptyText: TextView
    private lateinit var blurValueText: TextView
    private lateinit var depthBlurView: DepthBlurGLView
    private var selectedImageUri: Uri? = null
    private var sourceBitmap: Bitmap? = null
    private var depthBitmap: Bitmap? = null
    private var preparedPortrait: PortraitBlurProcessor.Prepared? = null
    private var selfieMaskRunner: SelfieMaskRunner? = null
    private val inferenceExecutor = Executors.newSingleThreadExecutor()

    private val imagePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            selectedImageUri = uri
            imageView.setImageURI(uri)
            depthBlurView.visibility = View.GONE
            imageView.visibility = View.VISIBLE
            emptyText.visibility = View.GONE
            depthView.visibility = View.GONE
            sourceBitmap = null
            depthBitmap = null
            preparedPortrait = null
            applyGlobalBlur(0f)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        depthView = findViewById(R.id.depthView)
        depthBlurView = findViewById(R.id.depthBlurView)
        emptyText = findViewById(R.id.emptyText)
        blurValueText = findViewById(R.id.blurValueText)
        val selectImageButton: Button = findViewById(R.id.selectImageButton)
        val depthButton: Button = findViewById(R.id.depthButton)
        val blurSeekBar: SeekBar = findViewById(R.id.blurSeekBar)

        selfieMaskRunner = SelfieMaskRunner()
        selectImageButton.setOnClickListener { imagePicker.launch("image/*") }
        depthButton.setOnClickListener { runPortraitBlur() }

        blurSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                blurValueText.text = "قوة عزل الخلفية: $progress"
                if (preparedPortrait != null) {
                    depthBlurView.setStrength(progress)
                } else if (selectedImageUri != null) {
                    applyGlobalBlur(progress.toFloat())
                }
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        })
    }

    private fun runPortraitBlur() {
        if (selectedImageUri == null || imageView.drawable == null) {
            Toast.makeText(this, "اختر صورة أولًا", Toast.LENGTH_SHORT).show()
            return
        }

        val bitmap = imageView.drawable.toBitmap()
        sourceBitmap = bitmap
        Toast.makeText(this, "جاري استخراج القناع والعمق...", Toast.LENGTH_SHORT).show()

        selfieMaskRunner?.process(
            bitmap,
            onSuccess = { mask ->
                runMidasAndPrepare(bitmap, mask)
            },
            onFailure = { error ->
                Toast.makeText(this, "فشل قناع الشخص: ${error.message}", Toast.LENGTH_LONG).show()
            }
        )
    }

    private fun runMidasAndPrepare(bitmap: Bitmap, personMask: Bitmap) {
        inferenceExecutor.execute {
            try {
              //  val generatedDepth = MidasDepthRunner(this).run(bitmap)
                val generatedDepth = DepthAnythingRunner(this).run(bitmap)
                val prepared = PortraitBlurProcessor.prepare(bitmap, generatedDepth, personMask)
                depthBitmap = generatedDepth
                preparedPortrait = prepared
                val gpuSource = scaleForGpu(bitmap)
                runOnUiThread {
                    depthBlurView.setBitmaps(gpuSource, generatedDepth, personMask)
                    depthBlurView.setStrength(findViewById<SeekBar>(R.id.blurSeekBar).progress)
                    imageView.visibility = View.GONE
                    depthBlurView.visibility = View.VISIBLE
                    depthView.setImageBitmap(generatedDepth)
                    depthView.visibility = View.VISIBLE
                    Toast.makeText(this, "تم تجهيز عزل الشخص والخلفية", Toast.LENGTH_SHORT).show()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    Toast.makeText(this, "فشل تشغيل MiDaS: ${error.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun renderPortraitPreview(strength: Int) {
        val prepared = preparedPortrait ?: return
        inferenceExecutor.execute {
            try {
                val result = prepared.render(strength)
                runOnUiThread { imageView.setImageBitmap(result) }
            } catch (error: Exception) {
                runOnUiThread {
                    Toast.makeText(this, "فشل تطبيق البلور: ${error.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun scaleForGpu(source: Bitmap, maxEdge: Int = 1280): Bitmap {
        val edge = maxOf(source.width, source.height)
        if (edge <= maxEdge) return source
        val ratio = maxEdge.toFloat() / edge.toFloat()
        return Bitmap.createScaledBitmap(
            source,
            (source.width * ratio).toInt().coerceAtLeast(1),
            (source.height * ratio).toInt().coerceAtLeast(1),
            true
        )
    }

    private fun applyGlobalBlur(radius: Float) {
        if (radius <= 0f) {
            imageView.setRenderEffect(null)
        } else {
            imageView.setRenderEffect(
                RenderEffect.createBlurEffect(radius, radius, Shader.TileMode.CLAMP)
            )
        }
    }

    override fun onDestroy() {
        selfieMaskRunner?.close()
        inferenceExecutor.shutdownNow()
        super.onDestroy()
    }
}
