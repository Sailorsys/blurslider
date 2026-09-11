package com.example.blurslider

import android.content.Context
import android.graphics.Bitmap
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.util.AttributeSet
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class DepthBlurGLView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : GLSurfaceView(context, attrs) {

    private val renderer = DepthBlurRenderer()

    init {
        setEGLContextClientVersion(3)
        setRenderer(renderer)
        renderMode = RENDERMODE_WHEN_DIRTY
    }

    fun setBitmaps(main: Bitmap, depth: Bitmap, mask: Bitmap) {
        queueEvent {
            renderer.updateTextures(main, depth, mask)
            requestRender()
        }
    }

    fun setBlurStrength(strength: Float) {
        queueEvent {
            renderer.blurStrength = strength.coerceIn(0f, 1f)
            requestRender()
        }
    }

    fun setFocusDepth(focus: Float) {
        queueEvent {
            renderer.focusDepth = focus.coerceIn(0f, 1f)
            requestRender()
        }
    }

    fun setLightThreshold(threshold: Float) {
        queueEvent {
            renderer.lightThreshold = threshold.coerceIn(0f, 1f)
            requestRender()
        }
    }

    fun setMaxBokehSize(size: Float) {
        queueEvent {
            renderer.maxBokehSize = size.coerceIn(0f, 90f)
            requestRender()
        }
    }

    fun setBokehColorBoost(boost: Float) {
        queueEvent {
            renderer.bokehColorBoost = boost.coerceIn(0f, 1f)
            requestRender()
        }
    }

    private inner class DepthBlurRenderer : Renderer {

        var blurStrength: Float = 0.40f
        var focusDepth: Float = 0.5f
        var lightThreshold: Float = 0.30f
        var maxBokehSize: Float = 40f
        var bokehColorBoost: Float = 0.5f

        private var viewportWidth = 0
        private var viewportHeight = 0
        private var imageWidth = 1
        private var imageHeight = 1

        private var scaleX = 1.0f
        private var scaleY = 1.0f

        private var mainTextureId = 0
        private var depthTextureId = 0
        private var maskTextureId = 0

        private var fboId = 0
        private var fboTextureId = 0

        private var pendingMain: Bitmap? = null
        private var pendingDepth: Bitmap? = null
        private var pendingMask: Bitmap? = null
        private var hasNewBitmaps = false

        private var pass1Program = 0
        private var pass2QuadProgram = 0
        private var pass2PointsProgram = 0

        private lateinit var quadBuffer: FloatBuffer
        private lateinit var pointGridBuffer: FloatBuffer
        private var pointCount = 0

        private val gridResolutionX = 250
        private val gridResolutionY = 250

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            GLES30.glClearColor(0f, 0f, 0f, 1f)

            initBuffers()
            initPass1Shader()
            initPass2QuadShader()
            initPass2PointsShader()
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            viewportWidth = width
            viewportHeight = height
            GLES30.glViewport(0, 0, width, height)
            setupFBO(width, height)
            updateAspectScale()
        }

        override fun onDrawFrame(gl: GL10?) {
            if (hasNewBitmaps) {
                uploadBitmapsGL()
                hasNewBitmaps = false
            }

            if (mainTextureId == 0) return

            // Pass 1: بلور الخلفية الأساسي
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
            GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

            GLES30.glUseProgram(pass1Program)
            bindTexture(mainTextureId, GLES30.GL_TEXTURE0, "u_mainTex", pass1Program)
            bindTexture(depthTextureId, GLES30.GL_TEXTURE1, "u_depthTex", pass1Program)
            bindTexture(maskTextureId, GLES30.GL_TEXTURE2, "u_maskTex", pass1Program)

            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_blurStrength"), blurStrength)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass1Program, "u_focusDepth"), focusDepth)
            GLES30.glUniform2f(GLES30.glGetUniformLocation(pass1Program, "u_scale"), scaleX, scaleY)

            drawQuad(pass1Program)

            // Pass 2: رسم الصورة ورسم دوائر البوكيه الناعمة فوقها
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
            GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

            GLES30.glUseProgram(pass2QuadProgram)
            bindTexture(fboTextureId, GLES30.GL_TEXTURE0, "u_fboTex", pass2QuadProgram)
            drawQuad(pass2QuadProgram)

            // دمج شفافية دوائر البوكيه
            GLES30.glEnable(GLES30.GL_BLEND)
            GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)

            GLES30.glUseProgram(pass2PointsProgram)

            bindTexture(mainTextureId, GLES30.GL_TEXTURE0, "u_mainTex", pass2PointsProgram)
            bindTexture(depthTextureId, GLES30.GL_TEXTURE1, "u_depthTex", pass2PointsProgram)
            bindTexture(maskTextureId, GLES30.GL_TEXTURE2, "u_maskTex", pass2PointsProgram)

            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass2PointsProgram, "u_focusDepth"), focusDepth)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass2PointsProgram, "u_maxBokehSize"), maxBokehSize)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass2PointsProgram, "u_lightThreshold"), lightThreshold)
            GLES30.glUniform1f(GLES30.glGetUniformLocation(pass2PointsProgram, "u_bokehColorBoost"), bokehColorBoost)
            GLES30.glUniform2f(GLES30.glGetUniformLocation(pass2PointsProgram, "u_scale"), scaleX, scaleY)
            GLES30.glUniform2f(
                GLES30.glGetUniformLocation(pass2PointsProgram, "u_resolution"),
                viewportWidth.toFloat(),
                viewportHeight.toFloat()
            )

            val aPosLoc = GLES30.glGetAttribLocation(pass2PointsProgram, "a_position")
            GLES30.glEnableVertexAttribArray(aPosLoc)
            GLES30.glVertexAttribPointer(aPosLoc, 2, GLES30.GL_FLOAT, false, 0, pointGridBuffer)

            GLES30.glDrawArrays(GLES30.GL_POINTS, 0, pointCount)

            GLES30.glDisableVertexAttribArray(aPosLoc)
            GLES30.glDisable(GLES30.GL_BLEND)
        }

        fun updateTextures(main: Bitmap, depth: Bitmap, mask: Bitmap) {
            pendingMain = main
            pendingDepth = depth
            pendingMask = mask
            hasNewBitmaps = true
        }

        private fun updateAspectScale() {
            if (viewportWidth <= 0 || viewportHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) return

            val imageAspect = imageWidth.toFloat() / imageHeight.toFloat()
            val viewAspect = viewportWidth.toFloat() / viewportHeight.toFloat()

            if (imageAspect > viewAspect) {
                scaleX = 1.0f
                scaleY = viewAspect / imageAspect
            } else {
                scaleX = imageAspect / viewAspect
                scaleY = 1.0f
            }
        }

        private fun initBuffers() {
            val quadCoords = floatArrayOf(
                -1.0f, 1.0f,
                -1.0f, -1.0f,
                1.0f, 1.0f,
                1.0f, -1.0f
            )
            quadBuffer = ByteBuffer.allocateDirect(quadCoords.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(quadCoords)
            quadBuffer.position(0)

            val points = FloatArray(gridResolutionX * gridResolutionY * 2)
            var index = 0
            for (y in 0 until gridResolutionY) {
                val ny = (y.toFloat() / (gridResolutionY - 1)) * 2.0f - 1.0f
                for (x in 0 until gridResolutionX) {
                    val nx = (x.toFloat() / (gridResolutionX - 1)) * 2.0f - 1.0f
                    points[index++] = nx
                    points[index++] = ny
                }
            }
            pointCount = gridResolutionX * gridResolutionY

            pointGridBuffer = ByteBuffer.allocateDirect(points.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(points)
            pointGridBuffer.position(0)
        }

        private fun setupFBO(width: Int, height: Int) {
            if (fboId != 0) {
                GLES30.glDeleteFramebuffers(1, intArrayOf(fboId), 0)
                GLES30.glDeleteTextures(1, intArrayOf(fboTextureId), 0)
            }

            val fboArr = IntArray(1)
            val texArr = IntArray(1)

            GLES30.glGenFramebuffers(1, fboArr, 0)
            GLES30.glGenTextures(1, texArr, 0)

            fboId = fboArr[0]
            fboTextureId = texArr[0]

            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, fboTextureId)
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, width, height, 0,
                GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
            )

            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fboId)
            GLES30.glFramebufferTexture2D(
                GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D, fboTextureId, 0
            )

            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        }

        private fun uploadBitmapsGL() {
            if (pendingMain != null && !pendingMain!!.isRecycled) {
                imageWidth = pendingMain!!.width
                imageHeight = pendingMain!!.height
                updateAspectScale()
            }
            mainTextureId = loadTexture(pendingMain, mainTextureId)
            depthTextureId = loadTexture(pendingDepth, depthTextureId)
            maskTextureId = loadTexture(pendingMask, maskTextureId)
        }

        private fun loadTexture(bitmap: Bitmap?, oldId: Int): Int {
            if (bitmap == null || bitmap.isRecycled) return oldId
            var id = oldId
            if (id == 0) {
                val tex = IntArray(1)
                GLES30.glGenTextures(1, tex, 0)
                id = tex[0]
            }

            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, id)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

            GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
            return id
        }

        private fun bindTexture(textureId: Int, textureUnit: Int, uniformName: String, program: Int) {
            GLES30.glActiveTexture(textureUnit)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
            GLES30.glUniform1i(GLES30.glGetUniformLocation(program, uniformName), textureUnit - GLES30.GL_TEXTURE0)
        }

        private fun drawQuad(program: Int) {
            val aPosLoc = GLES30.glGetAttribLocation(program, "a_position")
            GLES30.glEnableVertexAttribArray(aPosLoc)
            GLES30.glVertexAttribPointer(aPosLoc, 2, GLES30.GL_FLOAT, false, 0, quadBuffer)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
            GLES30.glDisableVertexAttribArray(aPosLoc)
        }

        private fun initPass1Shader() {
            val vs = """#version 300 es
                in vec2 a_position;
                uniform vec2 u_scale;
                out vec2 v_texCoord;
                void main() {
                    v_texCoord = vec2((a_position.x + 1.0) * 0.5, (1.0 - a_position.y) * 0.5);
                    gl_Position = vec4(a_position * u_scale, 0.0, 1.0);
                }
            """.trimIndent()

            val fs = """#version 300 es
                precision mediump float;
                in vec2 v_texCoord;
                
                uniform sampler2D u_mainTex;
                uniform sampler2D u_depthTex;
                uniform sampler2D u_maskTex;
                
                uniform float u_blurStrength;
                uniform float u_focusDepth;
                
                out vec4 fragColor;
                
                const float GOLDEN_ANGLE = 2.39996323;
                
                void main() {
                    vec4 color = texture(u_mainTex, v_texCoord);
                    float depth = texture(u_depthTex, v_texCoord).r;
                    float mask = texture(u_maskTex, v_texCoord).r;
                    
                    float coc = abs(depth - u_focusDepth);
                    float factor = coc * u_blurStrength * (1.0 - mask);
                    
                    if (factor < 0.01) {
                        fragColor = color;
                        return;
                    }
                    
                    vec4 accumColor = vec4(0.0);
                    float totalWeight = 0.0;
                    float radius = factor * 0.035;
                    
                    for (int i = 0; i < 128; i++) {
                        float fi = float(i);
                        float r = sqrt(fi / 128.0) * radius;
                        float theta = fi * GOLDEN_ANGLE;
                        vec2 off = vec2(cos(theta), sin(theta)) * r;
                        vec2 sampleUV = v_texCoord + off;

                        float sampleDepth = texture(u_depthTex, sampleUV).r;
                        float depthDiff = abs(sampleDepth - depth);
                        
                        float weight = 1.0 - smoothstep(0.0, 0.08, depthDiff);
                        float sampleMask = texture(u_maskTex, sampleUV).r;
                        weight *= (1.0 - sampleMask * 0.85);

                        if (weight > 0.01) {
                            vec4 sampleCol = texture(u_mainTex, sampleUV);
                            accumColor += sampleCol * weight;
                            totalWeight += weight;
                        }
                    }
                    
                    if (totalWeight > 0.001) {
                        fragColor = accumColor / totalWeight;
                    } else {
                        fragColor = color;
                    }
                }
            """.trimIndent()

            pass1Program = createProgram(vs, fs)
        }

        private fun initPass2QuadShader() {
            val vs = """#version 300 es
                in vec2 a_position;
                out vec2 v_texCoord;
                void main() {
                    v_texCoord = vec2((a_position.x + 1.0) * 0.5, (a_position.y + 1.0) * 0.5);
                    gl_Position = vec4(a_position, 0.0, 1.0);
                }
            """.trimIndent()

            val fs = """#version 300 es
                precision mediump float;
                in vec2 v_texCoord;
                uniform sampler2D u_fboTex;
                out vec4 fragColor;
                void main() {
                    fragColor = texture(u_fboTex, v_texCoord);
                }
            """.trimIndent()

            pass2QuadProgram = createProgram(vs, fs)
        }

        private fun initPass2PointsShader() {
    val vs = """#version 300 es
        in vec2 a_position;
        
        uniform sampler2D u_mainTex;
        uniform sampler2D u_depthTex;
        uniform sampler2D u_maskTex;
        
        uniform float u_focusDepth;
        uniform float u_maxBokehSize;
        uniform float u_lightThreshold;
        uniform float u_bokehColorBoost;
        uniform vec2 u_scale;
        
        out vec4 v_color;
        out vec2 v_texCoord;
        out float v_pointSize;
        out float v_reflectionFactor;
        out float v_stretchDirection;
        
        float getBrightness(vec3 color) {
            float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
            float maxColor = max(color.r, max(color.g, color.b));
            return max(luminance, maxColor);
        }
        
        void main() {
            vec2 texCoord = vec2((a_position.x + 1.0) * 0.5, (1.0 - a_position.y) * 0.5);
            v_texCoord = texCoord;
            
            vec4 color = texture(u_mainTex, texCoord);
            float depth = texture(u_depthTex, texCoord).r;
            float mask = texture(u_maskTex, texCoord).r;
            
            float brightness = getBrightness(color.rgb);
            float coc = abs(depth - u_focusDepth);
            
            if (mask > 0.15 || coc < 0.02 || brightness < u_lightThreshold) {
                gl_PointSize = 0.0;
                v_color = vec4(0.0);
                v_pointSize = 0.0;
                v_reflectionFactor = 0.0;
                v_stretchDirection = 0.0;
                gl_Position = vec4(a_position * u_scale, 0.0, 1.0);
                return;
            }
            
            vec2 stepSize = vec2(2.0 / 250.0, 2.0 / 250.0);
            
            float bRight = getBrightness(texture(u_mainTex, texCoord + vec2(stepSize.x, 0.0)).rgb);
            float bLeft  = getBrightness(texture(u_mainTex, texCoord - vec2(stepSize.x, 0.0)).rgb);
            float bTop   = getBrightness(texture(u_mainTex, texCoord + vec2(0.0, stepSize.y)).rgb);
            float bDown  = getBrightness(texture(u_mainTex, texCoord - vec2(0.0, stepSize.y)).rgb);
            
            if (bRight > brightness || bTop > brightness || bLeft >= brightness || bDown >= brightness) {
                gl_PointSize = 0.0;
                v_color = vec4(0.0);
                v_pointSize = 0.0;
                v_reflectionFactor = 0.0;
                v_stretchDirection = 0.0;
                gl_Position = vec4(a_position * u_scale, 0.0, 1.0);
                return;
            }
            
            // ===== حساب معامل الانعكاس =====
            float deltaX = abs(brightness - (bLeft + bRight) * 0.5);
            float deltaY = abs(brightness - (bTop + bDown) * 0.5);
            float maxDelta = max(deltaX, deltaY);

            float lowContrast = 1.0 - smoothstep(0.04, 0.22, maxDelta);
            float positionAssist = smoothstep(0.4, -0.6, a_position.y) * 0.35;
            float reflection = clamp(lowContrast * 0.85 + positionAssist, 0.0, 1.0);
            v_reflectionFactor = reflection;

            // اتجاه السحب: 0 = رأسي ، 1 = أفقي
            float stretchDirection = smoothstep(0.0, 0.15, deltaX - deltaY);
            v_stretchDirection = stretchDirection;
            
            float size = smoothstep(0.02, 0.40, coc) * u_maxBokehSize;
            
            // تكبير النقطة شوية للانعكاسات
            gl_PointSize = size * mix(1.0, 1.40, v_reflectionFactor);
            v_pointSize = gl_PointSize;
            
            // حيوية اللون
            float lum = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
            float satFactor = 1.0 + u_bokehColorBoost * 2.0; 
            vec3 saturatedColor = mix(vec3(lum), color.rgb, satFactor);
            
            float alpha = smoothstep(u_lightThreshold * 0.99, 1.0, brightness) * 0.75 + 0.25;
            v_color = vec4(clamp(saturatedColor, 0.0, 1.0), alpha);
            
            gl_Position = vec4(a_position * u_scale, 0.0, 1.0);
        }
    """.trimIndent()

    val fs = """#version 300 es
        precision mediump float;
        
        in vec4 v_color;
        in vec2 v_texCoord;
        in float v_pointSize;
        in float v_reflectionFactor;
        in float v_stretchDirection;
        
        uniform sampler2D u_maskTex;
        uniform vec2 u_resolution;
        
        out vec4 fragColor;
        
        void main() {
            if (v_color.a <= 0.0 || v_pointSize <= 0.0) discard;
            
            vec2 pointOffset = (gl_PointCoord - vec2(0.5)) * (v_pointSize / u_resolution);
            vec2 fragUV = v_texCoord + pointOffset;
            
            float fragMask = texture(u_maskTex, fragUV).r;
            if (fragMask > 0.15) discard;
            
            vec2 uv = gl_PointCoord - vec2(0.5);
            
            // السحب حسب الاتجاه
            float scaleX, scaleY;
            
            if (v_stretchDirection < 0.5) {
                // سحب رأسي (أرض)
                scaleX = mix(1.0, 0.35, v_reflectionFactor);
                scaleY = mix(1.0, 1.95, v_reflectionFactor);
            } else {
                // سحب أفقي (حيطة)
                scaleX = mix(1.0, 1.95, v_reflectionFactor);
                scaleY = mix(1.0, 0.35, v_reflectionFactor);
            }
            
            vec2 stretchedUV = vec2(uv.x / scaleX, uv.y / scaleY);
            float dist = length(stretchedUV);
            
            if (dist > 0.5) discard;
            
            float edgeSmooth = mix(0.12, 0.45, v_reflectionFactor);
            float edge = smoothstep(0.50, 0.50 - edgeSmooth, dist);
            
            float alpha = v_color.a * mix(1.0, 0.78, v_reflectionFactor);
            
            fragColor = vec4(v_color.rgb, alpha * edge);
        }
    """.trimIndent()

    pass2PointsProgram = createProgram(vs, fs)
}

        private fun createProgram(vertexSrc: String, fragmentSrc: String): Int {
            val vs = loadShader(GLES30.GL_VERTEX_SHADER, vertexSrc)
            val fs = loadShader(GLES30.GL_FRAGMENT_SHADER, fragmentSrc)
            val prog = GLES30.glCreateProgram()
            GLES30.glAttachShader(prog, vs)
            GLES30.glAttachShader(prog, fs)
            GLES30.glLinkProgram(prog)
            return prog
        }

        private fun loadShader(type: Int, shaderCode: String): Int {
            val shader = GLES30.glCreateShader(type)
            GLES30.glShaderSource(shader, shaderCode)
            GLES30.glCompileShader(shader)
            return shader
        }
    }
}
