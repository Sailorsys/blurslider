package com.example.blurslider

import android.content.Context
import android.graphics.Bitmap
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.util.AttributeSet
import android.os.Handler
import android.os.Looper
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.max

class DepthBlurGLView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : GLSurfaceView(context, attrs) {
    private val rendererImpl = Renderer()

    init {
        setEGLContextClientVersion(2)
        setRenderer(rendererImpl)
        renderMode = RENDERMODE_WHEN_DIRTY
    }

    fun setBitmaps(source: Bitmap, depth: Bitmap, mask: Bitmap) {
        rendererImpl.setBitmaps(source, depth, mask)
    }

    fun setStrength(value: Int) {
        rendererImpl.strength = value / 15f
        requestRender()
    }

    private inner class Renderer : GLSurfaceView.Renderer {
        private val vertexBuffer: FloatBuffer = floatBuffer(floatArrayOf(
                -1f, -1f, 1f, -1f, -1f, 1f,
                1f, -1f, 1f, 1f, -1f, 1f
        ))
        private val texBuffer: FloatBuffer = floatBuffer(floatArrayOf(
                0f, 1f, 1f, 1f, 0f, 0f,
                1f, 1f, 1f, 0f, 0f, 0f
        ))
        private var program = 0
        private var imageTexture = 0
        private var depthTexture = 0
        private var maskTexture = 0
        private var imageWidth = 1
        private var imageHeight = 1
        @Volatile var strength = 0f
        private var source: Bitmap? = null
        private var depth: Bitmap? = null
        private var mask: Bitmap? = null
        private val hasData = AtomicBoolean(false)

        fun setBitmaps(source: Bitmap, depth: Bitmap, mask: Bitmap) {
            post {
                if (imageTexture != 0) {
                    GLES20.glDeleteTextures(3, intArrayOf(imageTexture, depthTexture, maskTexture), 0)
                    imageTexture = 0
                    depthTexture = 0
                    maskTexture = 0
                }
                this.source = source
                this.depth = depth
                this.mask = mask
                imageWidth = source.width
                imageHeight = source.height
                hasData.set(true)
                requestRender()
            }
        }

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            GLES20.glClearColor(0.06f, 0.07f, 0.08f, 1f)
            program = buildProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            this.width = width
            this.height = height
            GLES20.glViewport(0, 0, width, height)
        }

        override fun onDrawFrame(gl: GL10?) {
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
            if (!hasData.get()) return
            if (imageTexture == 0) {
                imageTexture = createTexture(source)
                depthTexture = createTexture(depth)
                maskTexture = createTexture(mask)
            }
            GLES20.glUseProgram(program)
            val position = GLES20.glGetAttribLocation(program, "aPosition")
            val texCoord = GLES20.glGetAttribLocation(program, "aTexCoord")
            val image = GLES20.glGetUniformLocation(program, "uImage")
            val depthLoc = GLES20.glGetUniformLocation(program, "uDepth")
            val maskLoc = GLES20.glGetUniformLocation(program, "uMask")
            val strengthLoc = GLES20.glGetUniformLocation(program, "uStrength")
            val texelLoc = GLES20.glGetUniformLocation(program, "uTexel")

            vertexBuffer.position(0)
            GLES20.glEnableVertexAttribArray(position)
            GLES20.glVertexAttribPointer(position, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer)
            texBuffer.position(0)
            GLES20.glEnableVertexAttribArray(texCoord)
            GLES20.glVertexAttribPointer(texCoord, 2, GLES20.GL_FLOAT, false, 0, texBuffer)

            bindTexture(0, imageTexture, image)
            bindTexture(1, depthTexture, depthLoc)
            bindTexture(2, maskTexture, maskLoc)
            GLES20.glUniform1f(strengthLoc, strength)
            GLES20.glUniform2f(texelLoc, 1f / max(1, imageWidth), 1f / max(1, imageHeight))
            GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6)
            GLES20.glDisableVertexAttribArray(position)
            GLES20.glDisableVertexAttribArray(texCoord)
        }

        private var width = 1
        private var height = 1

        private fun bindTexture(unit: Int, texture: Int, location: Int) {
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0 + unit)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture)
            GLES20.glUniform1i(location, unit)
        }

        private fun createTexture(bitmap: Bitmap?): Int {
            val texture = IntArray(1)
            GLES20.glGenTextures(1, texture, 0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture[0])
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
            if (bitmap != null) GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0)
            return texture[0]
        }

        private fun buildProgram(vertex: String, fragment: String): Int {
            val v = compile(GLES20.GL_VERTEX_SHADER, vertex)
            val f = compile(GLES20.GL_FRAGMENT_SHADER, fragment)
            return GLES20.glCreateProgram().also {
                GLES20.glAttachShader(it, v)
                GLES20.glAttachShader(it, f)
                GLES20.glLinkProgram(it)
            }
        }

        private fun compile(type: Int, source: String): Int {
            return GLES20.glCreateShader(type).also {
                GLES20.glShaderSource(it, source)
                GLES20.glCompileShader(it)
            }
        }
    }

    private fun floatBuffer(values: FloatArray): FloatBuffer = ByteBuffer
    .allocateDirect(values.size * 4)
    .order(ByteOrder.nativeOrder())
    .asFloatBuffer()
    .apply { put(values); position(0) }

    companion object {
        private const val VERTEX_SHADER = """
            attribute vec2 aPosition;
            attribute vec2 aTexCoord;
            varying vec2 vTexCoord;
            void main() {
                gl_Position = vec4(aPosition, 0.0, 1.0);
                vTexCoord = aTexCoord;
            }
        """

        private const val FRAGMENT_SHADER = """
            precision mediump float;
            varying vec2 vTexCoord;
            uniform sampler2D uImage;
            uniform sampler2D uDepth;
            uniform sampler2D uMask;
            uniform float uStrength;
            uniform vec2 uTexel;

            vec4 blurSample(vec2 uv, float radius) {
                vec4 sum = texture2D(uImage, uv) * 0.227027;
                sum += texture2D(uImage, uv + vec2(1.384615, 0.0) * uTexel * radius) * 0.316216;
                sum += texture2D(uImage, uv - vec2(1.384615, 0.0) * uTexel * radius) * 0.316216;
                sum += texture2D(uImage, uv + vec2(3.230769, 0.0) * uTexel * radius) * 0.070270;
                sum += texture2D(uImage, uv - vec2(3.230769, 0.0) * uTexel * radius) * 0.070270;
                sum += texture2D(uImage, uv + vec2(0.0, 1.384615) * uTexel * radius) * 0.316216;
                sum += texture2D(uImage, uv - vec2(0.0, 1.384615) * uTexel * radius) * 0.316216;
                sum += texture2D(uImage, uv + vec2(0.0, 3.230769) * uTexel * radius) * 0.070270;
                sum += texture2D(uImage, uv - vec2(0.0, 3.230769) * uTexel * radius) * 0.070270;
                return sum / 1.8;
            }

            void main() {
                vec4 original = texture2D(uImage, vTexCoord);
                float depth = texture2D(uDepth, vTexCoord).r;
                float person = texture2D(uMask, vTexCoord).r;
                float depthNorm = depth; // لو العمق قريب = فاتح
                float backgroundByDepth = smoothstep(0.15, 0.65, 0.55 - depthNorm);
                float personProtection = smoothstep(0.25, 0.70, person);
                float background = max(backgroundByDepth * 0.85, 1.0 - personProtection);
                float amount = pow(background, 0.85) * uStrength;

                float radius = 2.5 + 18.0 * uStrength;
                vec4 blurred = blurSample(vTexCoord, radius);
                gl_FragColor = mix(original, blurred, clamp(amount, 0.0, 1.0));
            }
        """
    }
}
