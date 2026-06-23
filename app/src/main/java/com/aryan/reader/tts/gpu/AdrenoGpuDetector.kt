/*
 * Episteme Reader - GPU Detection for TTS acceleration
 * Copyright (C) 2026 Episteme
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */
package com.aryan.reader.tts.gpu

import android.app.ActivityManager
import android.content.Context
import android.opengl.GLES20
import android.opengl.GLES30
import android.os.Build
import timber.log.Timber
import java.io.File
import javax.microedition.khronos.egl.EGL10
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.egl.EGLContext
import javax.microedition.khronos.egl.EGLDisplay

/**
 * Detects Adreno GPU availability and capabilities for TTS acceleration.
 * Based on a8nova's adreno-llms requirements.
 */
object AdrenoGpuDetector {
    
    private const val TAG = "ADRENO_GPU"
    
    // Minimum Adreno versions that support cl_qcom_dot_product8
    // Required for int8 dot product acceleration
    private val SUPPORTED_ADRENO_VERSIONS = listOf(
        "Adreno (TM) 610",
        "Adreno (TM) 612", 
        "Adreno (TM) 618",
        "Adreno (TM) 619",
        "Adreno (TM) 620",
        "Adreno (TM) 630",
        "Adreno (TM) 640",
        "Adreno (TM) 650",
        "Adreno (TM) 660",
        "Adreno (TM) 680",
        "Adreno (TM) 690",
        "Adreno (TM) 695",
        // Adreno 7xx should work but not tuned
        "Adreno (TM) 710",
        "Adreno (TM) 720",
        "Adreno (TM) 730",
        "Adreno (TM) 740",
        "Adreno (TM) 750"
    )
    
    // GPU memory requirements (in MB)
    private const val MIN_GPU_MEMORY_MB = 512
    private const val RECOMMENDED_GPU_MEMORY_MB = 1024
    
    data class GpuInfo(
        val renderer: String,
        val vendor: String,
        val version: String,
        val isAdreno: Boolean,
        val adrenoVersion: Int?,  // e.g., 620 for Adreno 620
        val isSupported: Boolean,
        val supportsDotProduct8: Boolean,
        val estimatedMemoryMb: Int
    )
    
    /**
     * Check if Adreno GPU acceleration is available and supported.
     */
    fun isAdrenoGpuAvailable(context: Context): Boolean {
        val gpuInfo = detectGpuInfo(context)
        return gpuInfo.isSupported && gpuInfo.supportsDotProduct8
    }
    
    /**
     * Get detailed GPU information.
     */
    fun detectGpuInfo(context: Context): GpuInfo {
        var renderer = "Unknown"
        var vendor = "Unknown"
        var version = "Unknown"
        
        try {
            // Create temporary EGL context to query GPU info
            val egl = javax.microedition.khronos.egl.EGLContext.getEGL() as EGL10
            val display = egl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY)
            
            if (display !== EGL10.EGL_NO_DISPLAY) {
                val versionArray = IntArray(2)
                egl.eglInitialize(display, versionArray)
                
                val configAttribs = intArrayOf(
                    EGL10.EGL_RENDERABLE_TYPE, 4, // EGL_OPENGL_ES2_BIT
                    EGL10.EGL_NONE
                )
                
                val configs = arrayOfNulls<EGLConfig>(1)
                val numConfigs = IntArray(1)
                egl.eglChooseConfig(display, configAttribs, configs, 1, numConfigs)
                
                if (numConfigs[0] > 0) {
                    // Use EGL14 constants for context client version
                    val EGL_CONTEXT_CLIENT_VERSION = 0x3098
                    val attribList = intArrayOf(
                        EGL_CONTEXT_CLIENT_VERSION, 2,
                        EGL10.EGL_NONE
                    )
                    
                    val context = egl.eglCreateContext(display, configs[0], EGL10.EGL_NO_CONTEXT, attribList)
                    
                    if (context !== EGL10.EGL_NO_CONTEXT) {
                        egl.eglMakeCurrent(display, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_SURFACE, context)
                        
                        renderer = GLES20.glGetString(GLES20.GL_RENDERER) ?: "Unknown"
                        vendor = GLES20.glGetString(GLES20.GL_VENDOR) ?: "Unknown"
                        version = GLES20.glGetString(GLES20.GL_VERSION) ?: "Unknown"
                        
                        egl.eglMakeCurrent(display, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_CONTEXT)
                        egl.eglDestroyContext(display, context)
                    }
                }
                
                egl.eglTerminate(display)
            }
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to detect GPU via EGL")
            // Fallback: try to get from system properties
            renderer = getSystemProperty("ro.hardware.egl") ?: "Unknown"
        }
        
        val isAdreno = renderer.contains("Adreno", ignoreCase = true)
        val adrenoVersion = extractAdrenoVersion(renderer)
        val isSupported = isAdreno && (adrenoVersion != null && adrenoVersion >= 610)
        
        // cl_qcom_dot_product8 requires Adreno 6xx+
        val supportsDotProduct8 = adrenoVersion != null && adrenoVersion >= 610
        
        val estimatedMemory = estimateGpuMemory(context)
        
        Timber.tag(TAG).d("GPU Detection: renderer=$renderer, vendor=$vendor, version=$version, isAdreno=$isAdreno, adrenoVersion=$adrenoVersion, supported=$isSupported")
        
        return GpuInfo(
            renderer = renderer,
            vendor = vendor,
            version = version,
            isAdreno = isAdreno,
            adrenoVersion = adrenoVersion,
            isSupported = isSupported && estimatedMemory >= MIN_GPU_MEMORY_MB,
            supportsDotProduct8 = supportsDotProduct8,
            estimatedMemoryMb = estimatedMemory
        )
    }
    
    /**
     * Extract Adreno version number from renderer string.
     * e.g., "Adreno (TM) 620" -> 620
     */
    private fun extractAdrenoVersion(renderer: String): Int? {
        val regex = Regex("Adreno.*?(\\d{3,4})", RegexOption.IGNORE_CASE)
        val match = regex.find(renderer)
        return match?.groupValues?.get(1)?.toIntOrNull()
    }
    
    /**
     * Estimate available GPU memory.
     */
    private fun estimateGpuMemory(context: Context): Int {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        
        // Rough estimate: use total memory as proxy
        // Adreno 620 typically has 3.9 GB shared memory
        val totalMemory = memoryInfo.totalMem / (1024 * 1024) // Convert to MB
        
        // More accurate: check for specific device properties
        val glesVersion = try {
            val activityInfo = context.packageManager.getActivityInfo(
                android.content.ComponentName(context, context.javaClass),
                0
            )
            activityInfo?.applicationInfo?.metaData?.getInt("android.app.opengles_version", 0) ?: 0
        } catch (e: Exception) {
            0
        }
        
        return when {
            totalMemory > 6000 -> 4096 // High-end devices
            totalMemory > 4000 -> 2048 // Mid-range
            totalMemory > 2000 -> 1024 // Low-end
            else -> 512
        }
    }
    
    /**
     * Get system property (requires READ_PHONE_STATE permission or root).
     */
    private fun getSystemProperty(key: String): String? {
        return try {
            Class.forName("android.os.SystemProperties")
                .getMethod("get", String::class.java)
                .invoke(null, key) as? String
        } catch (e: Exception) {
            null
        }
    }
    
    /**
     * Check if OpenCL is available on the device.
     * This is a heuristic - actual availability requires native code check.
     */
    fun isOpenClLikelyAvailable(): Boolean {
        // Check for OpenCL library
        val openClPaths = listOf(
            "/system/vendor/lib/libOpenCL.so",
            "/system/lib/libOpenCL.so",
            "/system/vendor/lib64/libOpenCL.so",
            "/system/lib64/libOpenCL.so",
            "/system/lib/egl/libOpenCL.so",
            "/system/vendor/lib/egl/libOpenCL.so"
        )
        
        return openClPaths.any { File(it).exists() }
    }
    
    /**
     * Get a human-readable description of GPU capabilities.
     */
    fun getGpuCapabilitiesDescription(context: Context): String {
        val info = detectGpuInfo(context)
        return buildString {
            appendLine("GPU: ${info.renderer}")
            appendLine("Vendor: ${info.vendor}")
            appendLine("OpenGL Version: ${info.version}")
            appendLine("Adreno Version: ${info.adrenoVersion ?: "N/A"}")
            appendLine("Supported: ${info.isSupported}")
            appendLine("Dot Product 8: ${info.supportsDotProduct8}")
            appendLine("Estimated Memory: ${info.estimatedMemoryMb} MB")
            appendLine("OpenCL Available: ${isOpenClLikelyAvailable()}")
        }
    }
}
