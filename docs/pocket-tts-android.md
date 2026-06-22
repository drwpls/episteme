# PocketTTS Android integration

This fork adds an optional `POCKET` TTS mode backed by sherpa-onnx PocketTTS.

The repository intentionally does **not** commit the large sherpa-onnx AAR or
PocketTTS model files. The app still builds without them. At runtime, PocketTTS
works after adding the runtime and model assets below.

## Runtime

Download a sherpa-onnx Android AAR from a k2-fsa/sherpa-onnx release, for example:

```text
sherpa-onnx-<version>.aar
```

Place it here:

```text
app/libs/sherpa-onnx-<version>.aar
```

Gradle includes `app/libs/*.aar` automatically.

## Model assets

Download the PocketTTS int8 model archive from the sherpa-onnx TTS models release:

```text
sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
```

Extract it into Android assets:

```text
app/src/main/assets/sherpa-onnx-pocket-tts-int8-2026-01-26/
  lm_flow.int8.onnx
  lm_main.int8.onnx
  encoder.onnx
  decoder.int8.onnx
  text_conditioner.onnx
  vocab.json
  token_scores.json
  test_wavs/bria.wav
```

The default asset paths are controlled by these BuildConfig fields in
`app/build.gradle.kts`:

```kotlin
POCKET_TTS_MODEL_DIR = "sherpa-onnx-pocket-tts-int8-2026-01-26"
POCKET_TTS_REFERENCE_AUDIO = "sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav"
```

## Flow

```text
TtsMode.POCKET
  -> PocketTtsSynthesizer
  -> sherpa-onnx OfflineTts Pocket model
  -> WAV file in cacheDir
  -> existing TtsPlaybackManager / ExoPlayer pipeline
```

The adapter uses reflection so builds remain possible without bundling sherpa-onnx.
