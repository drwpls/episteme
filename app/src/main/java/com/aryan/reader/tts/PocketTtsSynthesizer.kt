/*
 * Episteme Reader - A native Android document reader.
 * Copyright (C) 2026 Episteme
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * mail: epistemereader@gmail.com
 */
package com.aryan.reader.tts

import android.content.Context
import android.content.res.AssetManager
import android.os.ParcelFileDescriptor
import com.aryan.reader.BuildConfig
import com.aryan.reader.epubreader.loadTtsSpeechRate
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import me.zhanghai.android.libarchive.Archive
import me.zhanghai.android.libarchive.ArchiveEntry
import me.zhanghai.android.libarchive.ArchiveException
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val POCKET_TTS_TAG = "POCKET_TTS_DIAG"
private const val POCKET_TTS_NUM_THREADS = 2
private const val POCKET_TTS_NUM_STEPS = 2
private const val PREFS_NAME = "pocket_tts_prefs"
private const val KEY_SELECTED_MODEL = "selected_model"
private const val MODELS_SUBDIR = "pocket-tts-models"

data class PocketTtsModel(
    val name: String,
    val displayName: String,
    val url: String,
    val description: String
)

class PocketTtsSynthesizer(private val context: Context) {
    private val mutex = Mutex()
    private var offlineTts: Any? = null
    private var loadedModelDir: String? = null

    companion object {
        val AVAILABLE_MODELS = listOf(
            PocketTtsModel(
                name = "sherpa-onnx-pocket-tts-int8-2026-01-26",
                displayName = "PocketTTS int8 (English)",
                url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2",
                description = "~65 MB, int8 quantized, English"
            ),
            PocketTtsModel(
                name = "sherpa-onnx-pocket-tts-2026-01-26",
                displayName = "PocketTTS float32 (English, higher quality)",
                url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-2026-01-26.tar.bz2",
                description = "~250 MB, float32, English"
            )
        )

        fun getModelsDirectory(context: Context): File {
            return File(context.filesDir, MODELS_SUBDIR)
        }

        fun getDownloadedModels(context: Context): List<String> {
            val dir = getModelsDirectory(context)
            if (!dir.isDirectory) return emptyList()
            return dir.listFiles()?.filter { it.isDirectory }?.map { it.name }?.sorted() ?: emptyList()
        }

        fun getSelectedModelName(context: Context): String {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            return prefs.getString(KEY_SELECTED_MODEL, "") ?: ""
        }

        fun setSelectedModelName(context: Context, name: String) {
            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit().putString(KEY_SELECTED_MODEL, name).apply()
        }

        fun getModelFile(context: Context, modelName: String, relativePath: String): File {
            return File(getModelsDirectory(context), "$modelName/$relativePath")
        }

        fun isModelDownloaded(context: Context, modelName: String): Boolean {
            val modelDir = File(getModelsDirectory(context), modelName)
            return modelDir.isDirectory() && modelDir.list()?.isNotEmpty() == true
        }
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun getCurrentModelName(): String = prefs.getString(KEY_SELECTED_MODEL, "") ?: ""

    fun setCurrentModelName(name: String) {
        prefs.edit().putString(KEY_SELECTED_MODEL, name).apply()
        release()
    }

    suspend fun downloadModel(
        model: PocketTtsModel,
        onProgress: (Float) -> Unit
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val modelDir = File(getModelsDirectory(context), model.name)
            modelDir.mkdirs()

            val archiveFile = File(context.cacheDir, "${model.name}.tar.bz2")
            if (archiveFile.exists()) archiveFile.delete()

            val url = URL(model.url)
            val conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = 30000
            conn.readTimeout = 30000
            conn.connect()

            val contentLength = conn.contentLengthLong
            val inputStream = conn.inputStream
            val outputStream = FileOutputStream(archiveFile)
            val buffer = ByteArray(8192)
            var bytesRead: Int
            var totalRead = 0L

            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                outputStream.write(buffer, 0, bytesRead)
                totalRead += bytesRead
                if (contentLength > 0) {
                    onProgress(totalRead.toFloat() / contentLength)
                }
            }
            outputStream.close()
            inputStream.close()
            conn.disconnect()

            onProgress(1f)

            extractTarBz2(archiveFile, modelDir)

            archiveFile.delete()

            Result.success(model.name)
        } catch (e: Exception) {
            Timber.tag(POCKET_TTS_TAG).e(e, "Failed to download model ${model.name}")
            Result.failure(e)
        }
    }

    suspend fun listDownloadedModels(): List<String> = withContext(Dispatchers.IO) {
        getDownloadedModels(context)
    }

    suspend fun synthesizeToFile(text: String): Pair<File?, String?> {
        if (text.isBlank()) return Pair(null, text)

        return withContext(Dispatchers.IO) {
            mutex.withLock {
                try {
                    val tts = getOrCreateTtsLocked()
                    val generationConfigClass = Class.forName("com.k2fsa.sherpa.onnx.GenerationConfig")
                    val generationConfig = createGenerationConfig(generationConfigClass)
                    val generatedAudio = tts.javaClass
                        .getMethod("generateWithConfig", String::class.java, generationConfigClass)
                        .invoke(tts, text, generationConfig)

                    val outFile = File.createTempFile("pocket_tts_", ".wav", context.cacheDir)
                    generatedAudio.javaClass
                        .getMethod("save", String::class.java)
                        .invoke(generatedAudio, outFile.absolutePath)

                    Pair(outFile, text)
                } catch (e: ClassNotFoundException) {
                    Timber.tag(POCKET_TTS_TAG).e(e, "sherpa-onnx classes not found. Add sherpa-onnx-*.aar to app/libs/.")
                    Pair(null, "PocketTTS is not installed. Add sherpa-onnx AAR and PocketTTS model assets.")
                } catch (e: UnsatisfiedLinkError) {
                    Timber.tag(POCKET_TTS_TAG).e(e, "sherpa-onnx native libraries missing")
                    Pair(null, "PocketTTS native libraries are missing from the APK.")
                } catch (e: Exception) {
                    Timber.tag(POCKET_TTS_TAG).e(e, "PocketTTS synthesis failed")
                    Pair(null, e.message)
                }
            }
        }
    }

    fun release() {
        runCatching { offlineTts?.javaClass?.getMethod("release")?.invoke(offlineTts) }
        offlineTts = null
        loadedModelDir = null
    }

    private fun getOrCreateTtsLocked(): Any {
        val selectedModel = getCurrentModelName()
        if (offlineTts != null && loadedModelDir == selectedModel) {
            return offlineTts!!
        }
        runCatching { offlineTts?.javaClass?.getMethod("release")?.invoke(offlineTts) }
        offlineTts = null

        val pocketClass = Class.forName("com.k2fsa.sherpa.onnx.OfflineTtsPocketModelConfig")
        val modelClass = Class.forName("com.k2fsa.sherpa.onnx.OfflineTtsModelConfig")
        val configClass = Class.forName("com.k2fsa.sherpa.onnx.OfflineTtsConfig")
        val offlineTtsClass = Class.forName("com.k2fsa.sherpa.onnx.OfflineTts")

        val pocketConfig = pocketClass.getDeclaredConstructor().newInstance()
        val useFileModel = selectedModel.isNotBlank() && isModelDownloaded(context, selectedModel)

        if (useFileModel) {
            val modelDir = File(getModelsDirectory(context), selectedModel).absolutePath
            pocketConfig.apply {
                setField("lmFlow", "$modelDir/lm_flow.int8.onnx")
                setField("lmMain", "$modelDir/lm_main.int8.onnx")
                setField("encoder", "$modelDir/encoder.onnx")
                setField("decoder", "$modelDir/decoder.int8.onnx")
                setField("textConditioner", "$modelDir/text_conditioner.onnx")
                setField("vocabJson", "$modelDir/vocab.json")
                setField("tokenScoresJson", "$modelDir/token_scores.json")
            }
        } else {
            val modelDir = BuildConfig.POCKET_TTS_MODEL_DIR.trim().trim('/')
            pocketConfig.apply {
                setField("lmFlow", "$modelDir/lm_flow.int8.onnx")
                setField("lmMain", "$modelDir/lm_main.int8.onnx")
                setField("encoder", "$modelDir/encoder.onnx")
                setField("decoder", "$modelDir/decoder.int8.onnx")
                setField("textConditioner", "$modelDir/text_conditioner.onnx")
                setField("vocabJson", "$modelDir/vocab.json")
                setField("tokenScoresJson", "$modelDir/token_scores.json")
            }
        }

        val modelConfig = modelClass.getDeclaredConstructor().newInstance().apply {
            setField("pocket", pocketConfig)
            setField("numThreads", POCKET_TTS_NUM_THREADS)
            setField("debug", BuildConfig.DEBUG)
            setField("provider", "cpu")
        }
        val config = configClass.getDeclaredConstructor().newInstance().apply {
            setField("model", modelConfig)
            setField("maxNumSentences", 1)
            setField("silenceScale", 0.2f)
        }

        val tts = if (useFileModel) {
            runCatching {
                offlineTtsClass
                    .getConstructor(configClass)
                    .newInstance(config)
            }.getOrElse {
                offlineTtsClass
                    .getConstructor(AssetManager::class.java, configClass)
                    .newInstance(context.assets, config)
            }
        } else {
            offlineTtsClass
                .getConstructor(AssetManager::class.java, configClass)
                .newInstance(context.assets, config)
        }

        offlineTts = tts
        loadedModelDir = if (useFileModel) selectedModel else "@assets"
        return tts
    }

    private fun createGenerationConfig(clazz: Class<*>): Any {
        val selectedModel = getCurrentModelName()
        val referenceWave = if (selectedModel.isNotBlank() && isModelDownloaded(context, selectedModel)) {
            val refFile = getModelFile(context, selectedModel, "test_wavs/bria.wav")
            if (refFile.exists()) {
                readReferenceWaveFile(refFile)
            } else {
                readReferenceAudioFromAssets()
            }
        } else {
            readReferenceAudioFromAssets()
        }

        return clazz.getDeclaredConstructor().newInstance().apply {
            setField("referenceAudio", referenceWave.samples)
            setField("referenceSampleRate", referenceWave.sampleRate)
            setField("numSteps", POCKET_TTS_NUM_STEPS)
            setField("speed", loadTtsSpeechRate(context))
            setField("silenceScale", 0.2f)
            setField("extra", mapOf("temperature" to "0.7", "chunk_size" to "15"))
        }
    }

    private fun Any.setField(name: String, value: Any?) {
        javaClass.getDeclaredField(name).apply {
            isAccessible = true
            set(this@setField, value)
        }
    }

    private fun readReferenceAudioFromAssets(): ReferenceWave {
        val assetPath = BuildConfig.POCKET_TTS_REFERENCE_AUDIO.trim().trim('/')
        val bytes = context.assets.open(assetPath).use { it.readBytes() }
        return parsePcm16Wav(bytes)
    }

    private fun readReferenceWaveFile(file: File): ReferenceWave {
        val bytes = file.readBytes()
        return parsePcm16Wav(bytes)
    }

    private fun parsePcm16Wav(bytes: ByteArray): ReferenceWave {
        require(bytes.size > 44) { "Reference WAV is too small" }
        require(String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF") { "Reference audio is not a RIFF WAV" }
        require(String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE") { "Reference audio is not a WAVE file" }

        var offset = 12
        var sampleRate = 0
        var channels = 1
        var bitsPerSample = 16
        var dataOffset = -1
        var dataSize = 0

        chunkLoop@ while (offset + 8 <= bytes.size) {
            val chunkId = String(bytes, offset, 4, Charsets.US_ASCII)
            val chunkSize = bytes.leInt(offset + 4)
            val chunkDataOffset = offset + 8
            when (chunkId) {
                "fmt " -> {
                    val audioFormat = bytes.leShort(chunkDataOffset).toInt()
                    require(audioFormat == 1) { "Only PCM WAV reference audio is supported" }
                    channels = bytes.leShort(chunkDataOffset + 2).toInt().coerceAtLeast(1)
                    sampleRate = bytes.leInt(chunkDataOffset + 4)
                    bitsPerSample = bytes.leShort(chunkDataOffset + 14).toInt()
                    require(bitsPerSample == 16) { "Only 16-bit PCM WAV reference audio is supported" }
                }
                "data" -> {
                    dataOffset = chunkDataOffset
                    dataSize = chunkSize.coerceAtMost(bytes.size - dataOffset)
                    break@chunkLoop
                }
            }
            offset = chunkDataOffset + chunkSize + (chunkSize and 1)
        }

        require(sampleRate > 0) { "Reference WAV sample rate missing" }
        require(dataOffset >= 0 && dataSize > 0) { "Reference WAV data chunk missing" }

        val frameCount = dataSize / (2 * channels)
        val samples = FloatArray(frameCount)
        val buffer = ByteBuffer.wrap(bytes, dataOffset, dataSize).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until frameCount) {
            var mixed = 0f
            repeat(channels) { mixed += buffer.short / 32768f }
            samples[i] = mixed / channels
        }
        return ReferenceWave(samples = samples, sampleRate = sampleRate)
    }

    private fun ByteArray.leInt(offset: Int): Int = ByteBuffer.wrap(this, offset, 4).order(ByteOrder.LITTLE_ENDIAN).int
    private fun ByteArray.leShort(offset: Int): Short = ByteBuffer.wrap(this, offset, 2).order(ByteOrder.LITTLE_ENDIAN).short

    private fun extractTarBz2(archiveFile: File, targetDir: File) {
        val archivePtr = Archive.readNew()
        try {
            Archive.readSupportFilterAll(archivePtr)
            Archive.readSupportFormatAll(archivePtr)
            Archive.readOpenFileName(archivePtr, archiveFile.absolutePath.toByteArray(), 10240)

            while (true) {
                val entry = try {
                    Archive.readNextHeader(archivePtr)
                } catch (e: ArchiveException) {
                    if (e.code == Archive.ERRNO_EOF) break
                    throw e
                }
                if (entry == 0L) break

                val pathname = ArchiveEntry.pathnameUtf8(entry) ?: continue
                val entryFile = File(targetDir, pathname)
                if (pathname.endsWith("/")) {
                    entryFile.mkdirs()
                } else {
                    entryFile.parentFile?.mkdirs()
                    entryFile.createNewFile()
                    ParcelFileDescriptor.open(entryFile, ParcelFileDescriptor.MODE_WRITE_ONLY).use { pfd ->
                        Archive.readDataIntoFd(archivePtr, pfd.fd)
                    }
                }
            }
        } finally {
            Archive.readFree(archivePtr)
        }
    }

    private data class ReferenceWave(val samples: FloatArray, val sampleRate: Int)
}
