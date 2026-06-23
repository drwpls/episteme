/*
 * Episteme Reader - Model Manager for Adreno GPU TTS
 * Copyright (C) 2026 Episteme
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */
package com.aryan.reader.tts.gpu

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Manages downloading and caching of Adreno GPU TTS model files.
 * Based on a8nova's Kokoro-82M model requirements.
 */
class AdrenoTtsModelManager(private val context: Context) {
    
    companion object {
        private const val TAG = "ADRENO_TTS_MODEL"
        private const val PREFS_NAME = "adreno_tts_prefs"
        private const val KEY_MODEL_VERSION = "model_version"
        private const val KEY_VOICE_PACK = "voice_pack"
        private const val KEY_DOWNLOAD_PROGRESS = "download_progress"
        
        // Model file sizes (approximate, for progress tracking)
        const val MODEL_FILE_SIZE_BYTES = 164_000_000L  // ~164 MB for Kokoro-82M fp16
        const val VOICE_PACK_SIZE_BYTES = 520_000L      // ~520 KB per voice
        const val ESPEAK_DATA_SIZE_BYTES = 936_000L     // ~936 KB for English
        
        // HuggingFace model repository
        const val HF_REPO_BASE = "https://huggingface.co/a8nova/kokoro-82m-android/resolve/main"
        
        // Required files
        val REQUIRED_FILES = listOf(
            ModelFile("model.bin", MODEL_FILE_SIZE_BYTES, "Kokoro-82M model weights"),
            ModelFile("tokenizer.json", 50_000L, "Tokenizer configuration"),
            ModelFile("config.json", 10_000L, "Model configuration"),
            ModelFile("espeak-data/en_dict", ESPEAK_DATA_SIZE_BYTES, "English phonemizer data"),
            ModelFile("voices/af_heart.bin", VOICE_PACK_SIZE_BYTES, "Default voice pack")
        )
    }
    
    data class ModelFile(
        val path: String,
        val sizeBytes: Long,
        val description: String
    )
    
    data class DownloadProgress(
        val fileName: String,
        val bytesDownloaded: Long,
        val totalBytes: Long,
        val status: DownloadStatus
    ) {
        val progressPercent: Int
            get() = if (totalBytes > 0) {
                ((bytesDownloaded * 100) / totalBytes).toInt()
            } else 0
    }
    
    enum class DownloadStatus {
        PENDING,
        DOWNLOADING,
        COMPLETED,
        FAILED,
        CANCELLED
    }
    
    data class ModelStatus(
        val isDownloaded: Boolean,
        val downloadedSizeBytes: Long,
        val requiredSizeBytes: Long,
        val missingFiles: List<String>,
        val currentVoicePack: String?
    ) {
        val progressPercent: Int
            get() = if (requiredSizeBytes > 0) {
                ((downloadedSizeBytes * 100) / requiredSizeBytes).toInt()
            } else 0
        
        val isComplete: Boolean
            get() = missingFiles.isEmpty()
    }
    
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()
    
    private val _downloadProgress = MutableStateFlow<DownloadProgress?>(null)
    val downloadProgress: StateFlow<DownloadProgress?> = _downloadProgress
    
    /**
     * Get the base directory for model files.
     */
    fun getModelBaseDir(): File {
        return File(context.getExternalFilesDir(null), "adreno_tts_models").apply {
            mkdirs()
        }
    }
    
    /**
     * Check if all required model files are downloaded.
     */
    fun getModelStatus(): ModelStatus {
        val baseDir = getModelBaseDir()
        var downloadedSize = 0L
        val missingFiles = mutableListOf<String>()
        
        REQUIRED_FILES.forEach { file ->
            val filePath = File(baseDir, file.path)
            if (filePath.exists() && filePath.length() > 0) {
                downloadedSize += filePath.length()
            } else {
                missingFiles.add(file.path)
            }
        }
        
        val requiredSize = REQUIRED_FILES.sumOf { it.sizeBytes }
        
        return ModelStatus(
            isDownloaded = missingFiles.isEmpty(),
            downloadedSizeBytes = downloadedSize,
            requiredSizeBytes = requiredSize,
            missingFiles = missingFiles,
            currentVoicePack = prefs.getString(KEY_VOICE_PACK, "af_heart")
        )
    }
    
    /**
     * Download all required model files.
     */
    suspend fun downloadModels(
        onProgress: ((DownloadProgress) -> Unit)? = null
    ): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val baseDir = getModelBaseDir()
            
            REQUIRED_FILES.forEach { modelFile ->
                val localFile = File(baseDir, modelFile.path)
                
                // Skip if already downloaded
                if (localFile.exists() && localFile.length() > modelFile.sizeBytes * 0.9) {
                    Timber.tag(TAG).d("File already exists: ${modelFile.path}")
                    return@forEach
                }
                
                // Create parent directories
                localFile.parentFile?.mkdirs()
                
                // Download file
                _downloadProgress.value = DownloadProgress(
                    fileName = modelFile.path,
                    bytesDownloaded = 0,
                    totalBytes = modelFile.sizeBytes,
                    status = DownloadStatus.DOWNLOADING
                )
                
                val request = Request.Builder()
                    .url("$HF_REPO_BASE/${modelFile.path}")
                    .build()
                
                httpClient.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        throw IOException("Failed to download ${modelFile.path}: HTTP ${response.code}")
                    }
                    
                    val body = response.body ?: throw IOException("Empty response body")
                    
                    FileOutputStream(localFile).use { output ->
                        body.byteStream().use { input ->
                            val buffer = ByteArray(8192)
                            var bytesRead: Int
                            var totalRead = 0L
                            
                            while ((bytesRead = input.read(buffer)) != -1) {
                                output.write(buffer, 0, bytesRead)
                                totalRead += bytesRead
                                
                                val progress = DownloadProgress(
                                    fileName = modelFile.path,
                                    bytesDownloaded = totalRead,
                                    totalBytes = modelFile.sizeBytes,
                                    status = DownloadStatus.DOWNLOADING
                                )
                                _downloadProgress.value = progress
                                onProgress?.invoke(progress)
                            }
                        }
                    }
                }
                
                Timber.tag(TAG).d("Downloaded: ${modelFile.path}")
            }
            
            // Mark as complete
            prefs.edit().putString(KEY_MODEL_VERSION, "1.0.0").apply()
            _downloadProgress.value = null
            
            Result.success(Unit)
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to download models")
            _downloadProgress.value = DownloadProgress(
                fileName = "",
                bytesDownloaded = 0,
                totalBytes = 0,
                status = DownloadStatus.FAILED
            )
            Result.failure(e)
        }
    }
    
    /**
     * Download a specific voice pack.
     */
    suspend fun downloadVoicePack(
        voiceName: String,
        onProgress: ((DownloadProgress) -> Unit)? = null
    ): Result<File> = withContext(Dispatchers.IO) {
        try {
            val baseDir = getModelBaseDir()
            val voicesDir = File(baseDir, "voices").apply { mkdirs() }
            val voiceFile = File(voicesDir, "$voiceName.bin")
            
            if (voiceFile.exists() && voiceFile.length() > VOICE_PACK_SIZE_BYTES * 0.9) {
                return@withContext Result.success(voiceFile)
            }
            
            val request = Request.Builder()
                .url("$HF_REPO_BASE/voices/$voiceName.bin")
                .build()
            
            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    throw IOException("Failed to download voice pack: HTTP ${response.code}")
                }
                
                val body = response.body ?: throw IOException("Empty response")
                
                FileOutputStream(voiceFile).use { output ->
                    body.byteStream().copyTo(output)
                }
            }
            
            prefs.edit().putString(KEY_VOICE_PACK, voiceName).apply()
            Result.success(voiceFile)
        } catch (e: Exception) {
            Timber.tag(TAG).e(e, "Failed to download voice pack: $voiceName")
            Result.failure(e)
        }
    }
    
    /**
     * Get the path to the model binary file.
     */
    fun getModelBinPath(): String? {
        val file = File(getModelBaseDir(), "model.bin")
        return if (file.exists()) file.absolutePath else null
    }
    
    /**
     * Get the path to the voice pack file.
     */
    fun getVoicePackPath(voiceName: String = "af_heart"): String? {
        val file = File(getModelBaseDir(), "voices/$voiceName.bin")
        return if (file.exists()) file.absolutePath else null
    }
    
    /**
     * Get the path to espeak data directory.
     */
    fun getEspeakDataPath(): String {
        return File(getModelBaseDir(), "espeak-data").absolutePath
    }
    
    /**
     * Delete all downloaded models to free space.
     */
    suspend fun deleteAllModels(): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            getModelBaseDir().deleteRecursively()
            prefs.edit().remove(KEY_MODEL_VERSION).remove(KEY_VOICE_PACK).apply()
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Get total size of downloaded models.
     */
    fun getDownloadedSizeBytes(): Long {
        return getModelBaseDir().walkTopDown()
            .filter { it.isFile }
            .sumOf { it.length() }
    }
    
    /**
     * Check if a specific voice pack is available.
     */
    fun isVoicePackAvailable(voiceName: String): Boolean {
        return File(getModelBaseDir(), "voices/$voiceName.bin").exists()
    }
    
    /**
     * Get list of available voice packs.
     */
    fun getAvailableVoicePacks(): List<String> {
        val voicesDir = File(getModelBaseDir(), "voices")
        if (!voicesDir.exists()) return emptyList()
        
        return voicesDir.listFiles { file -> file.extension == "bin" }
            ?.map { it.nameWithoutExtension }
            ?: emptyList()
    }
}
