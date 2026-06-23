/*
 * Episteme Reader - JNI Bridge for Adreno GPU TTS
 * Copyright (C) 2026 Episteme
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */
package com.aryan.reader.tts.gpu

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * JNI bridge to a8nova's Kokoro-82M OpenCL implementation.
 * Provides GPU-accelerated TTS synthesis on Adreno devices.
 */
class AdrenoTtsEngine(private val context: Context) {
    
    companion object {
        private const val TAG = "ADRENO_TTS_ENGINE"
        private const val SAMPLE_RATE = 24000
        private const val BYTES_PER_SAMPLE = 2 // 16-bit PCM
        
        // Native library name
        private const val NATIVE_LIB_NAME = "adreno_tts"
        
        init {
            try {
                System.loadLibrary(NATIVE_LIB_NAME)
                Timber.tag(TAG).d("Loaded native library: $NATIVE_LIB_NAME")
            } catch (e: UnsatisfiedLinkError) {
                Timber.tag(TAG).e(e, "Failed to load native library $NATIVE_LIB_NAME")
            }
        }
        
        @JvmStatic
        external fun nativeInitialize(
            modelPath: String,
            voicePackPath: String,
            espeakDataPath: String
        ): Long
        
        @JvmStatic
        external fun nativeSynthesize(
            engineHandle: Long,
            text: String,
            voiceCode: String
        ): ByteArray?
        
        @JvmStatic
        external fun nativeSynthesizeStreaming(
            engineHandle: Long,
            text: String,
            voiceCode: String,
            callback: AdrenoTtsCallback
        ): Boolean
        
        @JvmStatic
        external fun nativeGetVersion(): String
        
        @JvmStatic
        external fun nativeIsGpuAvailable(): Boolean
        
        @JvmStatic
        external fun nativeGetGpuInfo(): String
        
        @JvmStatic
        external fun nativeDestroy(engineHandle: Long)
        
        @JvmStatic
        external fun nativeSetSpeed(engineHandle: Long, speed: Float)
        
        @JvmStatic
        external fun nativeGetLastError(): String?
    }
    
    interface AdrenoTtsCallback {
        fun onAudioChunk(audioData: ByteArray)
        fun onComplete()
        fun onError(error: String)
    }
    
    data class SynthesisResult(
        val audioData: ByteArray?,
        val sampleRate: Int = SAMPLE_RATE,
        val error: String? = null
    )
    
    data class EngineInfo(
        val version: String,
        val isGpuAvailable: Boolean,
        val gpuInfo: String
    )
    
    private var engineHandle: Long = 0L
    private var isInitialized = false
    
    /**
     * Check if the native library is loaded and GPU is available.
     */
    fun isAvailable(): Boolean {
        return try {
            nativeIsGpuAvailable()
        } catch (e: UnsatisfiedLinkError) {
            Timber.tag(TAG).e(e, "Native library not available")
            false
        }
    }
    
    /**
     * Get engine information.
     */
    fun getEngineInfo(): EngineInfo {
        return EngineInfo(
            version = try { nativeGetVersion() } catch (e: Exception) { "unknown" },
            isGpuAvailable = isAvailable(),
            gpuInfo = try { nativeGetGpuInfo() } catch (e: Exception) { "unknown" }
        )
    }
    
    /**
     * Initialize the TTS engine with model files.
     */
    suspend fun initialize(
        modelManager: AdrenoTtsModelManager,
        voicePack: String = "af_heart"
    ): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            if (isInitialized) {
                Timber.tag(TAG).w("Engine already initialized")
                return@withContext Result.success(Unit)
            }
            
            val modelPath = modelManager.getModelBinPath()
                ?: return@withContext Result.failure(
                    IllegalStateException("Model not downloaded")
                )
            
            val voicePackPath = modelManager.getVoicePackPath(voicePack)
                ?: return@withContext Result.failure(
                    IllegalStateException("Voice pack not downloaded: $voicePack")
                )
            
            val espeakDataPath = modelManager.getEspeakDataPath()
            
            Timber.tag(TAG).d("Initializing with model=$modelPath, voice=$voicePackPath")
            
            engineHandle = nativeInitialize(modelPath, voicePackPath, espeakDataPath)
            
            if (engineHandle == 0L) {
                val error = nativeGetLastError() ?: "Unknown initialization error"
                return@withContext Result.failure(RuntimeException(error))
            }
            
            isInitialized = true
            Timber.tag(TAG).i("Adreno TTS engine initialized successfully")
            Result.success(Unit)
            
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to initialize Adreno TTS engine")
            Result.failure(e)
        }
    }
    
    /**
     * Synthesize text to speech.
     * Returns raw PCM audio data (24kHz, 16-bit, mono).
     */
    suspend fun synthesize(
        text: String,
        voiceCode: String = "en-us"
    ): SynthesisResult = withContext(Dispatchers.IO) {
        try {
            if (!isInitialized || engineHandle == 0L) {
                return@withContext SynthesisResult(
                    audioData = null,
                    error = "Engine not initialized"
                )
            }
            
            if (text.isBlank()) {
                return@withContext SynthesisResult(
                    audioData = null,
                    error = "Text is empty"
                )
            }
            
            Timber.tag(TAG).d("Synthesizing: ${text.take(50)}...")
            
            val audioData = nativeSynthesize(engineHandle, text, voiceCode)
            
            if (audioData == null) {
                val error = nativeGetLastError() ?: "Synthesis failed"
                return@withContext SynthesisResult(
                    audioData = null,
                    error = error
                )
            }
            
            Timber.tag(TAG).d("Synthesized ${audioData.size} bytes of audio")
            SynthesisResult(audioData = audioData)
            
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Synthesis failed")
            SynthesisResult(
                audioData = null,
                error = e.message ?: "Unknown error"
            )
        }
    }
    
    /**
     * Synthesize text to speech with streaming callback.
     * Useful for real-time playback.
     */
    suspend fun synthesizeStreaming(
        text: String,
        voiceCode: String = "en-us",
        callback: AdrenoTtsCallback
    ): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            if (!isInitialized || engineHandle == 0L) {
                return@withContext Result.failure(
                    IllegalStateException("Engine not initialized")
                )
            }
            
            val success = nativeSynthesizeStreaming(engineHandle, text, voiceCode, callback)
            
            if (success) {
                Result.success(Unit)
            } else {
                val error = nativeGetLastError() ?: "Streaming synthesis failed"
                Result.failure(RuntimeException(error))
            }
            
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Streaming synthesis failed")
            Result.failure(e)
        }
    }
    
    /**
     * Convert raw PCM audio to WAV format.
     */
    fun pcmToWav(pcmData: ByteArray, sampleRate: Int = SAMPLE_RATE): ByteArray {
        val pcmSize = pcmData.size
        val wavSize = 44 + pcmSize // WAV header + PCM data
        
        val buffer = ByteBuffer.allocate(wavSize)
        buffer.order(ByteOrder.LITTLE_ENDIAN)
        
        // RIFF header
        buffer.put("RIFF".toByteArray())
        buffer.putInt(wavSize - 8) // File size - 8
        buffer.put("WAVE".toByteArray())
        
        // fmt chunk
        buffer.put("fmt ".toByteArray())
        buffer.putInt(16) // Subchunk1Size
        buffer.putShort(1) // AudioFormat (PCM)
        buffer.putShort(1) // NumChannels (mono)
        buffer.putInt(sampleRate) // SampleRate
        buffer.putInt(sampleRate * BYTES_PER_SAMPLE) // ByteRate
        buffer.putShort(BYTES_PER_SAMPLE.toShort()) // BlockAlign
        buffer.putShort(16) // BitsPerSample
        
        // data chunk
        buffer.put("data".toByteArray())
        buffer.putInt(pcmSize)
        buffer.put(pcmData)
        
        return buffer.array()
    }
    
    /**
     * Save audio data to a WAV file.
     */
    suspend fun saveToWavFile(
        pcmData: ByteArray,
        outputFile: File,
        sampleRate: Int = SAMPLE_RATE
    ): Result<File> = withContext(Dispatchers.IO) {
        try {
            val wavData = pcmToWav(pcmData, sampleRate)
            outputFile.parentFile?.mkdirs()
            outputFile.writeBytes(wavData)
            Result.success(outputFile)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Set playback speed.
     * @param speed Speed multiplier (0.25 to 4.0)
     */
    fun setSpeed(speed: Float) {
        if (isInitialized && engineHandle != 0L) {
            val clampedSpeed = speed.coerceIn(0.25f, 4.0f)
            nativeSetSpeed(engineHandle, clampedSpeed)
        }
    }
    
    /**
     * Release the engine and free native resources.
     */
    fun destroy() {
        if (isInitialized && engineHandle != 0L) {
            try {
                nativeDestroy(engineHandle)
                Timber.tag(TAG).d("Adreno TTS engine destroyed")
            } catch (e: Exception) {
                Timber.tag(TAG).e(e, "Error destroying engine")
            }
            engineHandle = 0L
            isInitialized = false
        }
    }
    
    protected fun finalize() {
        destroy()
    }
}
