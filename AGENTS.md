# Episteme Reader - Agent Documentation

## GPU TTS Call Flow

This document describes the call flow for Adreno GPU-accelerated TTS (Text-to-Speech) in Episteme Reader.

### Overview

The GPU TTS feature uses a8nova's Kokoro-82M OpenCL implementation to provide 2x faster speech synthesis on Qualcomm Adreno GPUs.

### Call Flow

```
1. User selects "GPU (Adreno)" mode in UI
   └─ Common.kt:1812 → adds ADRENO_GPU to mode list

2. User starts TTS playback
   └─ TtsService.kt:1174 → when(mode == ADRENO_GPU)
   └─ Calls synthesizeAdrenoGpuTtsChunk()

3. TtsService.synthesizeAdrenoGpuTtsChunk()
   ├─ Line 994: Checks GPU availability via AdrenoGpuDetector.isAdrenoGpuAvailable()
   ├─ Line 1003: Creates AdrenoTtsEngine instance
   ├─ Line 1004: Creates AdrenoTtsModelManager instance
   ├─ Line 1008: Checks if models are downloaded (getModelStatus())
   ├─ Line 1017: Initializes engine via adrenoTtsEngine.initialize()
   └─ Line 1033: Synthesizes text via adrenoTtsEngine.synthesize()

4. AdrenoTtsEngine.initialize()
   ├─ Line 141: Gets model path from modelManager.getModelBinPath()
   ├─ Line 146: Gets voice pack path from modelManager.getVoicePackPath("af_heart")
   ├─ Line 151: Gets espeak data path from modelManager.getEspeakDataPath()
   └─ Line 155: Calls nativeInitialize() → JNI

5. JNI Bridge (adreno_tts_jni.cpp)
   ├─ Line 272: Java_com_aryan_reader_tts_gpu_AdrenoTtsEngine_nativeInitialize
   ├─ Line 294: Creates kokoro::InferenceEngine via kokoro::createEngine()
   └─ Returns engine handle

6. AdrenoTtsEngine.synthesize()
   ├─ Line 176: Called with text and voice code
   └─ Calls nativeSynthesize() → JNI

7. JNI Bridge (adreno_tts_jni.cpp)
   └─ Line 322: Java_com_aryan_reader_tts_gpu_AdrenoTtsEngine_nativeSynthesize
   └─ Line 337-338: Converts Java strings to C strings
   └─ Line 355: Calls handle->engine->synthesize()
   └─ Returns byte array of PCM audio data

8. Back to AdrenoTtsEngine.synthesize()
   └─ Converts PCM to WAV format
   └─ Returns audio file

9. Back to TtsService.synthesizeAdrenoGpuTtsChunk()
   └─ Returns TtsAudioData with audio file
```

### Key Files

| File | Purpose |
|------|---------|
| `Common.kt:1812` | UI mode selector, adds GPU mode |
| `TtsService.kt:984-1056` | Main synthesis function for GPU mode |
| `AdrenoTtsEngine.kt:131-170` | Engine initialization with model/voice paths |
| `AdrenoTtsModelManager.kt` | Model download and path management |
| `adreno_tts_jni.cpp:272-320` | JNI native initialization |
| `adreno_tts_jni.cpp:322-370` | JNI native synthesis |

### Model Files Required

- `model.bin` (~164 MB) - Kokoro-82M weights from GitHub releases
- `phoneme_vocab.tsv` - Phoneme vocabulary
- `test_input_ids.bin` - Test input data
- `voices/af_heart.bin` (~520 KB) - Voice pack from HuggingFace

### Error Handling

Common errors:
- "Voice pack not downloaded: af_heart" - Voice pack missing, check AdrenoTtsModelManager.kt line 146
- "Adreno GPU not available" - Device doesn't have Adreno GPU or OpenCL
- "Failed to initialize Adreno TTS" - JNI/native library issue
