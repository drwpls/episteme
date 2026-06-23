package com.aryan.reader.tts

import android.content.Context
import android.os.ParcelFileDescriptor
import com.aryan.reader.BuildConfig
import com.aryan.reader.R
import com.aryan.reader.epubreader.loadTtsSpeechRate
import com.k2fsa.sherpa.onnx.GenerationConfig
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsPocketModelConfig
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

private const val LOG_TAG = "SHERPA_ONNX"
private const val NUM_THREADS = 2
private const val NUM_STEPS = 2
private const val PREFS_NAME = "sherpa_onnx_prefs"
private const val KEY_SELECTED_MODEL = "selected_model"
private const val MODELS_SUBDIR = "sherpa-onnx-models"

enum class ModelType { POCKET, KOKORO }

data class SherpaOnnxModel(
    val name: String,
    val displayName: String,
    val url: String,
    val description: String,
    val modelType: ModelType,
    val huggingfaceUrl: String? = null
)

class PocketTtsSynthesizer(private val context: Context) {
    private val mutex = Mutex()
    private var offlineTts: OfflineTts? = null
    private var loadedModelName: String? = null

    companion object {
        val AVAILABLE_MODELS = listOf(
            SherpaOnnxModel(
                name = "sherpa-onnx-pocket-tts-int8-2026-01-26",
                displayName = "PocketTTS int8 (English)",
                url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2",
                description = "~65 MB, int8 quantized, English",
                modelType = ModelType.POCKET,
                huggingfaceUrl = "https://huggingface.co/csukuangfj2/sherpa-onnx-pocket-tts-int8-2026-01-26"
            ),
            SherpaOnnxModel(
                name = "sherpa-onnx-pocket-tts-2026-01-26",
                displayName = "PocketTTS float32 (English, higher quality)",
                url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-2026-01-26.tar.bz2",
                description = "~250 MB, float32, English",
                modelType = ModelType.POCKET,
                huggingfaceUrl = "https://huggingface.co/csukuangfj2/sherpa-onnx-pocket-tts-2026-01-26"
            ),
            SherpaOnnxModel(
                name = "kokoro-multi-lang-v1_0",
                displayName = "Kokoro v1.0 (Chinese + English, 53 speakers)",
                url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2",
                description = "~300 MB, 53 speakers, Chinese + English",
                modelType = ModelType.KOKORO,
                huggingfaceUrl = null
            ),
            SherpaOnnxModel(
                name = "kokoro-en-v0_19",
                displayName = "Kokoro v0.19 (English, 11 speakers)",
                url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2",
                description = "~100 MB, 11 speakers, English",
                modelType = ModelType.KOKORO,
                huggingfaceUrl = "https://huggingface.co/RuiSumida/sherpa-onnx-kokoro-int8-en-v0_19"
            )
        )

        fun getModelsDirectory(context: Context): File {
            return File(context.filesDir, MODELS_SUBDIR)
        }

        fun getDownloadedModels(context: Context): List<String> {
            val dir = getModelsDirectory(context)
            if (!dir.isDirectory) return emptyList()
            return dir.listFiles()
                ?.filter { it.isDirectory && resolveModelRootDir(it) != null }
                ?.map { it.name }
                ?.sorted()
                ?: emptyList()
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
            return resolveModelRootDir(modelDir) != null
        }

        fun detectModelType(modelDir: File): ModelType {
            val modelRoot = resolveModelRootDir(modelDir) ?: modelDir
            val files = modelRoot.list() ?: return ModelType.POCKET
            return if (files.any { it == "lm_flow.int8.onnx" }) {
                ModelType.POCKET
            } else if (files.any { it == "model.onnx" } && files.any { it == "voices.bin" }) {
                ModelType.KOKORO
            } else {
                ModelType.POCKET
            }
        }

        fun detectModelTypeByName(modelName: String): ModelType {
            val lower = modelName.lowercase()
            return if (lower.startsWith("kokoro")) ModelType.KOKORO else ModelType.POCKET
        }

        fun resolveModelRootDir(modelDir: File): File? {
            if (hasPocketModelFiles(modelDir) || hasKokoroModelFiles(modelDir)) return modelDir
            return modelDir.listFiles()
                ?.filter { it.isDirectory }
                ?.firstOrNull { hasPocketModelFiles(it) || hasKokoroModelFiles(it) }
        }

        private fun hasPocketModelFiles(dir: File): Boolean {
            return listOf(
                "lm_flow.int8.onnx",
                "lm_main.int8.onnx",
                "encoder.onnx",
                "decoder.int8.onnx",
                "text_conditioner.onnx",
                "vocab.json",
                "token_scores.json"
            ).all { File(dir, it).isFile }
        }

        private fun hasKokoroModelFiles(dir: File): Boolean {
            return listOf("model.onnx", "voices.bin", "tokens.txt", "espeak-ng-data")
                .all { File(dir, it).exists() }
        }
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun getCurrentModelName(): String = prefs.getString(KEY_SELECTED_MODEL, "") ?: ""

    fun setCurrentModelName(name: String) {
        prefs.edit().putString(KEY_SELECTED_MODEL, name).apply()
        release()
    }

    suspend fun downloadModel(
        model: SherpaOnnxModel,
        onProgress: (Float) -> Unit
    ): Result<String> = downloadFromUrl(model.url, model.name, onProgress)

    suspend fun downloadFromUrl(
        url: String,
        modelName: String,
        onProgress: (Float) -> Unit
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val modelDir = File(getModelsDirectory(context), modelName)
            modelDir.mkdirs()

            val archiveFile = File(context.cacheDir, "$modelName.tar.bz2")
            if (archiveFile.exists()) archiveFile.delete()

            val conn = URL(url).openConnection() as HttpURLConnection
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
            Result.success(modelName)
        } catch (e: Exception) {
            Timber.tag(LOG_TAG).e(e, "Failed to download model from $url")
            Result.failure(e)
        }
    }

    suspend fun synthesizeToFile(text: String): Pair<File?, String?> {
        if (text.isBlank()) return Pair(null, text)
        if (!hasUsableModel()) {
            return Pair(null, context.getString(R.string.tts_pocket_no_model))
        }

        return withContext(Dispatchers.IO) {
            mutex.withLock {
                try {
                    val tts = getOrCreateOfflineTts()
                    val config = createGenerationConfig()
                    val audio = tts.generateWithConfig(text, config)

                    val outFile = File.createTempFile("sherpa_tts_", ".wav", context.cacheDir)
                    audio.save(outFile.absolutePath)

                    Pair(outFile, text)
                } catch (e: Exception) {
                    Timber.tag(LOG_TAG).e(e, "Sherpa-ONNX synthesis failed")
                    Pair(null, e.message)
                }
            }
        }
    }

    fun hasUsableModel(): Boolean {
        val selectedModel = getCurrentModelName()
        if (selectedModel.isNotBlank() && isModelDownloaded(context, selectedModel)) return true
        return hasBundledModelAssets()
    }

    fun release() {
        runCatching { offlineTts?.release() }
        offlineTts = null
        loadedModelName = null
    }

    private fun getOrCreateOfflineTts(): OfflineTts {
        val selectedModel = getCurrentModelName()
        if (offlineTts != null && loadedModelName == selectedModel) {
            return offlineTts!!
        }
        runCatching { offlineTts?.release() }
        offlineTts = null

        val useFileModel = selectedModel.isNotBlank() && isModelDownloaded(context, selectedModel)
        val modelType = if (useFileModel) {
            detectModelType(File(getModelsDirectory(context), selectedModel))
        } else {
            ModelType.POCKET
        }

        val modelConfig = OfflineTtsModelConfig()
        if (useFileModel) {
            val modelRootDir = resolveModelRootDir(File(getModelsDirectory(context), selectedModel))
                ?: error(context.getString(R.string.tts_pocket_no_model))
            val modelDir = modelRootDir.absolutePath
            when (modelType) {
                ModelType.POCKET -> {
                    val pocketConfig = OfflineTtsPocketModelConfig()
                    pocketConfig.lmFlow = "$modelDir/lm_flow.int8.onnx"
                    pocketConfig.lmMain = "$modelDir/lm_main.int8.onnx"
                    pocketConfig.encoder = "$modelDir/encoder.onnx"
                    pocketConfig.decoder = "$modelDir/decoder.int8.onnx"
                    pocketConfig.textConditioner = "$modelDir/text_conditioner.onnx"
                    pocketConfig.vocabJson = "$modelDir/vocab.json"
                    pocketConfig.tokenScoresJson = "$modelDir/token_scores.json"
                    modelConfig.pocket = pocketConfig
                }
                ModelType.KOKORO -> {
                    val kokoroConfig = OfflineTtsKokoroModelConfig()
                    kokoroConfig.model = "$modelDir/model.onnx"
                    kokoroConfig.voices = "$modelDir/voices.bin"
                    kokoroConfig.tokens = "$modelDir/tokens.txt"
                    kokoroConfig.dataDir = "$modelDir/espeak-ng-data"
                    val lexiconEn = File("$modelDir/lexicon-us-en.txt")
                    val lexiconZh = File("$modelDir/lexicon-zh.txt")
                    kokoroConfig.lexicon = buildString {
                        if (lexiconEn.exists()) append("$modelDir/lexicon-us-en.txt")
                        if (lexiconEn.exists() && lexiconZh.exists()) append(",")
                        if (lexiconZh.exists()) append("$modelDir/lexicon-zh.txt")
                    }
                    modelConfig.kokoro = kokoroConfig
                }
            }
        } else {
            val modelDir = BuildConfig.POCKET_TTS_MODEL_DIR.trim().trim('/')
            val pocketConfig = OfflineTtsPocketModelConfig()
            pocketConfig.lmFlow = "$modelDir/lm_flow.int8.onnx"
            pocketConfig.lmMain = "$modelDir/lm_main.int8.onnx"
            pocketConfig.encoder = "$modelDir/encoder.onnx"
            pocketConfig.decoder = "$modelDir/decoder.int8.onnx"
            pocketConfig.textConditioner = "$modelDir/text_conditioner.onnx"
            pocketConfig.vocabJson = "$modelDir/vocab.json"
            pocketConfig.tokenScoresJson = "$modelDir/token_scores.json"
            modelConfig.pocket = pocketConfig
        }
        modelConfig.numThreads = NUM_THREADS
        modelConfig.debug = BuildConfig.DEBUG
        modelConfig.provider = "cpu"

        val ttsConfig = OfflineTtsConfig()
        ttsConfig.model = modelConfig
        ttsConfig.maxNumSentences = 1
        ttsConfig.silenceScale = 0.2f

        val tts = OfflineTts(context.assets, ttsConfig)
        offlineTts = tts
        loadedModelName = if (useFileModel) selectedModel else "@assets"
        return tts
    }

    private fun createGenerationConfig(): GenerationConfig {
        val selectedModel = getCurrentModelName()
        val isFileModel = selectedModel.isNotBlank() && isModelDownloaded(context, selectedModel)
        val modelType = if (isFileModel) {
            detectModelType(File(getModelsDirectory(context), selectedModel))
        } else {
            ModelType.POCKET
        }

        val genConfig = GenerationConfig()
        when (modelType) {
            ModelType.POCKET -> {
                val modelRootDir = if (isFileModel) {
                    resolveModelRootDir(File(getModelsDirectory(context), selectedModel))
                } else {
                    null
                }
                val referenceWave = if (isFileModel) {
                    val refFile = modelRootDir?.let { File(it, "test_wavs/bria.wav") }
                    if (refFile?.exists() == true) readReferenceWaveFile(refFile)
                    else readReferenceAudioFromAssets()
                } else {
                    readReferenceAudioFromAssets()
                }
                genConfig.referenceAudio = referenceWave.samples
                genConfig.referenceSampleRate = referenceWave.sampleRate
                genConfig.numSteps = NUM_STEPS
                genConfig.extra = mapOf("temperature" to "0.7", "chunk_size" to "15")
            }
            ModelType.KOKORO -> {
                genConfig.sid = 0
            }
        }
        genConfig.speed = loadTtsSpeechRate(context)
        genConfig.silenceScale = 0.2f
        return genConfig
    }

    private fun readReferenceAudioFromAssets(): ReferenceWave {
        val assetPath = BuildConfig.POCKET_TTS_REFERENCE_AUDIO.trim().trim('/')
        val bytes = context.assets.open(assetPath).use { it.readBytes() }
        return parsePcm16Wav(bytes)
    }

    private fun hasBundledModelAssets(): Boolean {
        val modelDir = BuildConfig.POCKET_TTS_MODEL_DIR.trim().trim('/')
        if (modelDir.isBlank()) return false
        return runCatching {
            val required = listOf(
                "lm_flow.int8.onnx",
                "lm_main.int8.onnx",
                "encoder.onnx",
                "decoder.int8.onnx",
                "text_conditioner.onnx",
                "vocab.json",
                "token_scores.json"
            )
            val modelFiles = context.assets.list(modelDir)?.toSet().orEmpty()
            required.all { it in modelFiles } && context.assets.open(BuildConfig.POCKET_TTS_REFERENCE_AUDIO.trim().trim('/')).use { true }
        }.getOrDefault(false)
    }

    private fun readReferenceWaveFile(file: File): ReferenceWave {
        return parsePcm16Wav(file.readBytes())
    }

    private fun parsePcm16Wav(bytes: ByteArray): ReferenceWave {
        var offset = 12
        var sampleRate = 0
        var channels = 1
        var dataOffset = -1
        var dataSize = 0

        chunkLoop@ while (offset + 8 <= bytes.size) {
            val chunkId = String(bytes, offset, 4, Charsets.US_ASCII)
            val chunkSize = leInt(bytes, offset + 4)
            val chunkDataOffset = offset + 8
            when (chunkId) {
                "fmt " -> {
                    val audioFormat = leShort(bytes, chunkDataOffset).toInt()
                    require(audioFormat == 1) { "Only PCM WAV reference audio is supported" }
                    channels = leShort(bytes, chunkDataOffset + 2).toInt().coerceAtLeast(1)
                    sampleRate = leInt(bytes, chunkDataOffset + 4)
                    require(leShort(bytes, chunkDataOffset + 14).toInt() == 16) { "Only 16-bit PCM WAV reference audio is supported" }
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

    private fun leInt(bytes: ByteArray, offset: Int): Int =
        ByteBuffer.wrap(bytes, offset, 4).order(ByteOrder.LITTLE_ENDIAN).int

    private fun leShort(bytes: ByteArray, offset: Int): Short =
        ByteBuffer.wrap(bytes, offset, 2).order(ByteOrder.LITTLE_ENDIAN).short

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
