/*
 * Episteme Reader - JNI Bridge for Adreno GPU TTS
 * Copyright (C) 2026 Episteme
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <android/log.h>
#include <CL/opencl.h>

#define LOG_TAG "ADRENO_TTS_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// Forward declarations for a8nova's Kokoro implementation
// These would be implemented based on a8nova's actual C++ code
namespace kokoro {
    class InferenceEngine {
    public:
        virtual ~InferenceEngine() {}
        
        virtual std::vector<int16_t> synthesize(
            const std::string& text,
            const std::string& voiceCode
        ) = 0;
        
        virtual bool synthesizeStreaming(
            const std::string& text,
            const std::string& voiceCode,
            std::function<void(const std::vector<int16_t>&)> callback
        ) = 0;
        
        virtual void setSpeed(float speed) = 0;
        virtual std::string getLastError() const = 0;
    };
    
    InferenceEngine* createEngine(
        const std::string& modelPath,
        const std::string& voicePackPath,
        const std::string& espeakDataPath
    );
}

// Engine handle wrapper
struct EngineHandle {
    kokoro::InferenceEngine* engine;
    std::string lastError;
    
    EngineHandle() : engine(nullptr) {}
    ~EngineHandle() {
        delete engine;
    }
};

// Global registry of active engines
static std::vector<EngineHandle*> g_engines;
static std::mutex g_enginesMutex;

// Check if OpenCL is available and Adreno GPU is present
static bool checkAdrenoGpu() {
    cl_platform_id platform;
    cl_uint numPlatforms;
    
    cl_int err = clGetPlatformIDs(1, &platform, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        LOGD("No OpenCL platforms found");
        return false;
    }
    
    char platformName[256];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
    LOGD("OpenCL Platform: %s", platformName);
    
    // Check for Qualcomm/Adreno
    std::string nameStr(platformName);
    if (nameStr.find("Qualcomm") == std::string::npos &&
        nameStr.find("Adreno") == std::string::npos) {
        LOGD("Not a Qualcomm/Adreno platform");
        return false;
    }
    
    cl_device_id device;
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        LOGD("No GPU devices found");
        return false;
    }
    
    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    LOGD("OpenCL Device: %s", deviceName);
    
    // Check for Adreno
    std::string deviceStr(deviceName);
    if (deviceStr.find("Adreno") == std::string::npos) {
        LOGD("Not an Adreno GPU");
        return false;
    }
    
    // Check for cl_qcom_dot_product8 extension
    char extensions[4096];
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, nullptr);
    std::string extStr(extensions);
    
    bool hasDotProduct8 = extStr.find("cl_qcom_dot_product8") != std::string::npos;
    LOGD("Has cl_qcom_dot_product8: %s", hasDotProduct8 ? "yes" : "no");
    
    return hasDotProduct8;
}

extern "C" {

// JNI Functions

JNIEXPORT jstring JNICALL
Java_com_aryan_reader_tts_gpu_AdrenoTtsEngine_nativeGetVersion(JNIEnv* env, jclass clazz) {
    return env->NewStringUTF("1.0.0-a8nova");
}

JNIEXPORT jboolean JNICALL
Java_com_aryan_reader_tts_gpu_AdrenoTtsEngine_nativeIsGpuAvailable(JNIEnv* env, jclass clazz) {
    return checkAdrenoGpu() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_com_aryan_reader_tts_gpu_AdrenoTtsEngine_nativeGetGpuInfo(JNIEnv* env, jclass clazz) {
    std::string info;
    
    cl_platform_id platform;
    cl_uint numPlatforms;
    
    if (clGetPlatformIDs(1, &platform, &numPlatforms) == CL_SUCCESS && numPlatforms > 0) {
        char platformName[256];
        char platformVendor[256];
        char platformVersion[256];
        
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
        clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(platformVendor), platformVendor, nullptr);
        clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(platformVersion), platformVersion, nullptr);
        
        info += "Platform: " + std::string(platformName) + "\n";
        info += "Vendor: " + std::string(platformVendor) + "\n";
        info += "Version: " + std::string(platformVersion) + "\n";
        
        cl_device_id device;
        cl_uint numDevices;
        
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices) == CL_SUCCESS) {
            char deviceName[256];
            char deviceVersion[256];
            cl_ulong globalMemSize;
            
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(deviceVersion), deviceVersion, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, nullptr);
            
            info += "Device: " + std::string(deviceName) + "\n";
            info += "Device Version: " + std::string(deviceVersion) + "\n";
            info += "Global Memory: " + std::to_string(globalMemSize / (1024 * 1024)) + " MB\n";
            
            char extensions[4096];
            clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, nullptr);
            info += "Extensions: " + std::string(extensions);
        }
    } else {
        info = "OpenCL not available";
    }
    
    return env->NewStringUTF(info.c_str());
}

JNIEXPORT jlong JNICALL
Java_com_aryan_reader_tts_gpu_AdrenoTtsEngine_nativeInitialize(
    JNIEnv* env,
    jclass clazz,
    jstring modelPath,
    jstring voicePackPath,
    jstring espeakDataPath
) {
    const char* modelPathC = env->GetStringUTFChars(modelPath, nullptr);
    const char* voicePackPathC = env->GetStringUTFChars(voicePackPath, nullptr);
    const char* espeakDataPathC = env->GetStringUTFChars(espeakDataPath, nullptr);
    
    jlong handle = 0;
    
    try {
        LOGI("Initializing Adreno TTS engine");
        LOGI("Model: %s", modelPathC);
        LOGI("Voice: %s", voicePackPathC);
        
        // Create engine handle
        EngineHandle* handleObj = new EngineHandle();
        
        // Create engine
        handleObj->engine = kokoro::createEngine(
            std::string(modelPathC),
            std::string(voicePackPathC),
            std::string(espeakDataPathC)
        );
        
        if (handleObj->engine) {
            std::lock_guard<std::mutex> lock(g_enginesMutex);
            g_engines.push_back(handleObj);
            handle = reinterpret_cast<jlong>(handleObj);
            
            LOGI("Engine initialized successfully, handle=%ld", handle);
        } else {
            LOGE("Failed to create inference engine");
            delete handleObj;
        }
        
    } catch (const std::exception& e) {
        LOGE("Exception during initialization: %s", e.what());
    }
    
    env->ReleaseStringUTFChars(modelPath, modelPathC);
    env->ReleaseStringUTFChars(voicePackPath, voicePackPathC);
    env->ReleaseStringUTFChars(espeakDataPath, espeakDataPathC);
    
    return handle;
}

JNIEXPORT jbyteArray JNICALL
Java_com_aryan_reader_tts_gpu_AdrenoTtsEngine_nativeSynthesize(
    JNIEnv* env,
    jclass clazz,
    jlong engineHandle,
    jstring text,
    jstring voiceCode
) {
    if (engineHandle == 0) {
        LOGE("Invalid engine handle");
        return nullptr;
    }
    
    EngineHandle* handle = reinterpret_cast<EngineHandle*>(engineHandle);
    
    const char* textC = env->GetStringUTFChars(text, nullptr);
    const char* voiceCodeC = env->GetStringUTFChars(voiceCode, nullptr);
    
    jbyteArray result = nullptr;
    
    try {
        LOGD("Synthesizing text: %s", textC);
        
        std::vector<int16_t> audioData = handle->engine->synthesize(
            std::string(textC),
            std::string(voiceCodeC)
        );
        
        if (!audioData.empty()) {
            jsize size = static_cast<jsize>(audioData.size() * sizeof(int16_t));
            result = env->NewByteArray(size);
            env->SetByteArrayRegion(
                result,
                0,
                size,
                reinterpret_cast<const jbyte*>(audioData.data())
            );
            LOGD("Synthesis complete, %zu samples", audioData.size());
        } else {
            handle->lastError = "Synthesis produced no output";
        }
        
    } catch (const std::exception& e) {
        LOGE("Exception during synthesis: %s", e.what());
        handle->lastError = e.what();
    }
    
    env->ReleaseStringUTFChars(text, textC);
    env->ReleaseStringUTFChars(voiceCode, voiceCodeC);
    
    return result;
}

JNIEXPORT jboolean JNICALL
Java_com_aryan_reader_tts_gpu_AdrenoTtsEngine_nativeSynthesizeStreaming(
    JNIEnv* env,
    jclass clazz,
    jlong engineHandle,
    jstring text,
    jstring voiceCode,
    jobject callback
) {
    // Streaming implementation would require setting up a callback mechanism
    // For now, return false to indicate not implemented
    LOGD("Streaming synthesis not yet implemented");
    return JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_aryan_reader_tts_gpu_AdrenoTtsEngine_nativeSetSpeed(
    JNIEnv* env,
    jclass clazz,
    jlong engineHandle,
    jfloat speed
) {
    if (engineHandle == 0) return;
    
    EngineHandle* handle = reinterpret_cast<EngineHandle*>(engineHandle);
    if (handle && handle->engine) {
        handle->engine->setSpeed(speed);
    }
}

JNIEXPORT jstring JNICALL
Java_com_aryan_reader_tts_gpu_AdrenoTtsEngine_nativeGetLastError(
    JNIEnv* env,
    jclass clazz,
    jlong engineHandle
) {
    if (engineHandle == 0) {
        return env->NewStringUTF("Invalid engine handle");
    }
    
    EngineHandle* handle = reinterpret_cast<EngineHandle*>(engineHandle);
    std::string error = handle->lastError;
    
    if (error.empty() && handle->engine) {
        error = handle->engine->getLastError();
    }
    
    if (error.empty()) {
        return nullptr;
    }
    
    return env->NewStringUTF(error.c_str());
}

JNIEXPORT void JNICALL
Java_com_aryan_reader_tts_gpu_AdrenoTtsEngine_nativeDestroy(
    JNIEnv* env,
    jclass clazz,
    jlong engineHandle
) {
    if (engineHandle == 0) return;
    
    LOGI("Destroying engine handle=%ld", engineHandle);
    
    std::lock_guard<std::mutex> lock(g_enginesMutex);
    
    for (auto it = g_engines.begin(); it != g_engines.end(); ++it) {
        if (reinterpret_cast<jlong>(*it) == engineHandle) {
            delete *it;
            g_engines.erase(it);
            LOGI("Engine destroyed");
            break;
        }
    }
}

} // extern "C"

// Stub implementation for kokoro::InferenceEngine
// In production, this would be replaced with a8nova's actual implementation
namespace kokoro {

class InferenceEngineStub : public InferenceEngine {
public:
    std::vector<int16_t> synthesize(
        const std::string& text,
        const std::string& voiceCode
    ) override {
        // TODO: Implement actual synthesis using a8nova's code
        // For now, return empty vector
        LOGE("Synthesis not implemented - a8nova code integration required");
        return {};
    }
    
    bool synthesizeStreaming(
        const std::string& text,
        const std::string& voiceCode,
        std::function<void(const std::vector<int16_t>&)> callback
    ) override {
        return false;
    }
    
    void setSpeed(float speed) override {
        // TODO: Implement
    }
    
    std::string getLastError() const override {
        return lastError_;
    }
    
private:
    std::string lastError_;
};

InferenceEngine* createEngine(
    const std::string& modelPath,
    const std::string& voicePackPath,
    const std::string& espeakDataPath
) {
    // Check if files exist
    FILE* modelFile = fopen(modelPath.c_str(), "rb");
    if (!modelFile) {
        LOGE("Model file not found: %s", modelPath.c_str());
        return nullptr;
    }
    fclose(modelFile);
    
    return new InferenceEngineStub();
}

} // namespace kokoro
