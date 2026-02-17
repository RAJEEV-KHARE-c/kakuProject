/*
ESP32 Dashboard Client - Arduino C++ Version with Camera & Microphone
This sketch reads sensors, captures images, records audio, and sends data to dashboard

Required Libraries:
- ArduinoJson
- WiFi
- HTTPClient
- I2Cdev
- MPU6050
- esp_camera (ESP32 Camera)

Install via Arduino IDE: Sketch > Include Library > Manage Libraries
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include "MPU6050.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "fb_gfx.h"
#include "driver/i2s_pdm.h"
#include "base64.h"

// ================= OLED & PET ANIMATIONS =================
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "all_pets.h"

// ================= WIFI =================
#define WIFI_SSID     "123"
#define WIFI_PASSWORD "KUNAL 26"

// ================= API =================
const char* serverUrl = "http://192.168.43.67:5000/api/sensor-data";  // Updated to actual server IP
const char* eventsUrl = "http://192.168.43.67:5000/api/events?device_id=ESP32_001";  // Events endpoint
const char* eventReceivedUrl = "http://192.168.43.67:5000/api/device/event/received";  // Event acknowledgment
const char* oledDisplayUrl = "http://192.168.43.67:5000/api/oled-display/get";  // OLED display animation endpoint
// NOTE: Orientation endpoint removed - server computes direction from sensor data

// ================= CAMERA PINS (XIAO ESP32 S3 Sense) =================
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 10
#define SIOD_GPIO_NUM 40
#define SIOC_GPIO_NUM 39
#define Y9_GPIO_NUM 48
#define Y8_GPIO_NUM 11
#define Y7_GPIO_NUM 12
#define Y6_GPIO_NUM 14
#define Y5_GPIO_NUM 16
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 17
#define Y2_GPIO_NUM 15
#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM 47
#define PCLK_GPIO_NUM 13

// ================= AUDIO SETTINGS =================
#define SAMPLE_RATE     16000
#define SAMPLE_BITS     16
#define RECORD_SECONDS  1     // 1 second of audio (4x smaller payload)
#define WAV_HEADER_SIZE 44
#define VOLUME_GAIN     2
#define AUDIO_DATA_SIZE (SAMPLE_RATE * 2 * RECORD_SECONDS)

// ================= I2S PDM MIC PINS (XIAO ESP32 S3 Sense) =================
#define PDM_CLK_GPIO (gpio_num_t)42  // PDM CLK
#define PDM_DIN_GPIO (gpio_num_t)41  // PDM DATA
#define I2S_NUM I2S_NUM_0

// ================= OLED DISPLAY SETUP =================
#define SCREEN_WIDTH 64
#define SCREEN_HEIGHT 32
#define OLED_ADDR 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// ================= OLED ANIMATION DISPLAY =================
enum PetAge { INFANT = 0, CHILD = 1, ADULT = 2, OLD = 3 };
PetAge petAge = CHILD;              // Default to child
unsigned long lastAnimationTime = 0;
unsigned long lastDisplayCheckTime = 0;  // Track when we last checked server for OLED display state
const unsigned long DISPLAY_CHECK_INTERVAL = 2000;  // Poll server for OLED display state every 2 seconds
const unsigned long ANIMATION_DISPLAY_INTERVAL = 2000;  // Display animation every 2 seconds
uint8_t currentFrame = 0;
bool displayReady = false;

// MPU6050 sensor
MPU6050 mpu;
bool mpuAvailable = false;  // Track if MPU6050 is working

// Pins
const int LED_PIN = 2;         // LED indicator

// Voice Activity Detection
volatile bool speechDetected = false;
volatile bool audioReady = false;
volatile int audioEnergyLevel = 0;
String detectedAudioData = "";

// Dual-core synchronization
SemaphoreHandle_t audioMutex;
TaskHandle_t audioTaskHandle = NULL;

// VAD Settings
#define VAD_THRESHOLD 1000        // Energy threshold for speech detection
#define VAD_MIN_DURATION 500      // Minimum 500ms of speech to trigger
#define SILENCE_TIMEOUT 2000      // 2s of silence before stopping recording

// Structure for single sensor reading with timestamp
struct SingleReading {
    unsigned long timestamp_ms;  // Millisecond timestamp
    float accel_x, accel_y, accel_z;
    float gyro_x, gyro_y, gyro_z;
};

// Structure for buffered sensor data (batch of readings)
struct SensorDataBatch {
    static const int MAX_READINGS = 20;  // Store up to 20 readings
    int reading_count;
    SingleReading readings[MAX_READINGS];
    float avg_mic_level;
    int sound_data;
};

// Timing
unsigned long lastSendTime = 0;
unsigned long lastImageCapture = 0;
unsigned long lastEventPoll = 0;               // Event polling timing
unsigned long lastInternalReadTime = 0;        // Fast internal sensor reading timing
const unsigned long SEND_INTERVAL = 2000;      // Send sensor data batch every 2 seconds
const unsigned long INTERNAL_READ_INTERVAL = 100;  // NEW: Read sensor every 100ms internally
const unsigned long IMAGE_INTERVAL = 30000;    // Capture image every 30 seconds (reduced for power)
const unsigned long EVENT_POLL_INTERVAL = 5000; // Poll for events every 5 seconds
unsigned long dynamicEventPollInterval = 5000;  // Dynamic backoff for event polling
// Audio now triggered by speech detection, not timer

// NEW: Sensor reading buffer
SensorDataBatch sensorBatch = {};
float totalMicLevel = 0.0;
int micReadingCount = 0;

// ⏸️ PAUSE CONTROL FOR UPLOADS
bool isUploadingImage = false;                  // Flag to pause sensor data during image upload

// Camera and audio status
bool cameraReady = false;
bool micReady = false;
uint8_t* audio_buffer = NULL;

// Audio processing buffers for VAD
int16_t* vad_buffer;
const int VAD_BUFFER_SIZE = 512;

// PDM microphone handle
i2s_chan_handle_t rx_handle = NULL;

// Structure for sensor data
struct SensorData {
    float accel_x, accel_y, accel_z;
    float gyro_x, gyro_y, gyro_z;
    float mic_level;
    int sound_data;
    String camera_image_b64;  // Base64 encoded image
    String audio_data_b64;    // Base64 encoded audio
    bool has_new_image;
    bool has_new_audio;
    
    // Orientation tracking fields
    String device_orientation;
    float orientation_confidence;
    float calibrated_ax, calibrated_ay, calibrated_az;
    
    // NEW: Batch of sensor readings for better step detection
    SensorDataBatch sensor_batch;
};

// ================= ORIENTATION DETECTION =================
// NOTE: Direction detection moved to Flask server
// ESP32 now only sends raw MPU6050 data

// NOTE: Orientation computation moved to Flask server (see app.py)
// Server will compute direction from raw accel data

// ================= FORWARD DECLARATIONS =================
void displayPetAnimation();
SensorData readAllSensors();
String captureImageBase64();
bool sendSensorDataOnly(SensorData data);
void sendImageData(String imageBase64);
void sendAudioData(String audioBase64);
void pollForEvents();
bool isServerAlive();
void scanI2CDevices();
bool initCamera();
bool initAudio();
void audioMonitorTask(void *parameter);
void processEvent(const char* event_type, const char* message);
void acknowledgeEvent(int event_id);
void generate_wav_header(uint8_t* wav_header, uint32_t wav_size, uint32_t sample_rate);
void sendAllDataToServer(SensorData data);
String recordAudioBase64();

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n\nESP32 Dashboard Client Starting...");
    
    // Initialize LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    
    // Connect to WiFi
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Connecting WiFi");

    while (WiFi.status() != WL_CONNECTED) {
        Serial.print(".");
        delay(300);
    }

    Serial.println("\nWiFi Connected");
    Serial.println(WiFi.localIP());
    
    // FIX 6: Enable WiFi modem sleep for power savings
    WiFi.setSleep(true);
    Serial.println("💤 WiFi sleep mode enabled");
    
    // FIX 4: Lower CPU frequency from 240MHz to 160MHz
    setCpuFrequencyMhz(160);
    Serial.println("⚡ CPU frequency reduced to 160MHz");
    
    // Initialize I2C and MPU6050 with timeout protection
    Serial.println("Initializing I2C...");
    Wire.begin(5, 6);  // XIAO ESP32 S3: SDA=5, SCL=6
    Wire.setClock(400000);  // Set I2C speed to 400kHz
    delay(1000);
    
    // Initialize OLED Display
    Serial.println("Initializing OLED Display...");
    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
        Serial.println("❌ OLED initialization failed!");
        displayReady = false;
    } else {
        Serial.println("✅ OLED initialized successfully!");
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 0);
        display.println("ESP32 Pet Ready!");
        display.display();
        displayReady = true;
        delay(1000);
        display.clearDisplay();
    }
    
    Serial.println("Scanning I2C devices...");
    scanI2CDevices();
    
    Serial.println("Initializing MPU6050...");
    bool mpuSuccess = false;
    
    // Try MPU6050 initialization with timeout
    unsigned long startTime = millis();
    while (!mpuSuccess && (millis() - startTime < 5000)) {  // 5 second timeout
        mpu.initialize();
        delay(100);
        
        if (mpu.testConnection()) {
            Serial.println("✅ MPU6050 initialized successfully!");
            mpuSuccess = true;
            mpuAvailable = true;  // Set flag for sensor readings
        } else {
            Serial.print(".");
            delay(500);
        }
    }
    
    if (!mpuSuccess) {
        Serial.println("\n❌ MPU6050 initialization failed after 5 seconds");
        Serial.println("⚠️  Continuing without MPU6050 (will send dummy sensor data)");
    }
    
    // Initialize Camera
    initCamera();
    
    // Initialize I2S Microphone
    initAudio();
    
    // Create mutex for audio synchronization
    audioMutex = xSemaphoreCreateMutex();
    
    // Start audio monitoring task on Core 0 (dedicated to audio/VAD)
    xTaskCreatePinnedToCore(
        audioMonitorTask,    // Task function
        "AudioMonitor",      // Task name
        8192,               // Stack size
        NULL,               // Parameters
        2,                  // Priority (high for real-time audio)
        &audioTaskHandle,   // Task handle
        0                   // Core 0 (dedicated to audio)
    );
    
    Serial.println("System Ready!");
    Serial.println("🎤 Core 0: Audio monitoring with voice detection");
    Serial.println("🌐 Core 1: Sensors, camera, WiFi transmission");
}

void scanI2CDevices() {
    Serial.println("🔍 Scanning I2C devices...");
    byte error, address;
    int deviceCount = 0;

    for (address = 1; address < 127; address++) {
        Wire.beginTransmission(address);
        error = Wire.endTransmission();

        if (error == 0) {
            Serial.printf("✅ I2C device found at address 0x%02X", address);
            if (address == 0x68 || address == 0x69) {
                Serial.print(" (MPU6050)");
            }
            Serial.println();
            deviceCount++;
        }
        else if (error == 4) {
            Serial.printf("❌ Unknown error at address 0x%02X\n", address);
        }
    }

    if (deviceCount == 0) {
        Serial.println("⚠️  No I2C devices found");
        Serial.println("   Check wiring: SDA->GPIO6, SCL->GPIO7");
    } else {
        Serial.printf("🎯 Found %d I2C device(s)\n", deviceCount);
    }
    Serial.println();
}

// ================= PET ANIMATION FUNCTION =================
void displayPetAnimation() {
    if (!displayReady) return;
    
    // Poll server for OLED display animation every 2 seconds
    if (millis() - lastDisplayCheckTime >= DISPLAY_CHECK_INTERVAL) {
        lastDisplayCheckTime = millis();
        getOLEDDisplayFromServer();
    }
    
    // Display animation every 2 seconds
    if (millis() - lastAnimationTime >= ANIMATION_DISPLAY_INTERVAL) {
        lastAnimationTime = millis();
        display.clearDisplay();
        
        // Select frame based on pet age
        const uint8_t* frameData = nullptr;
        uint8_t frameCount = 0;
        
        switch (petAge) {
            case INFANT:
                frameData = infant_frames[currentFrame % INFANT_FRAME_COUNT];
                frameCount = INFANT_FRAME_COUNT;
                display.drawBitmap(0, 0, frameData, INFANT_WIDTH, INFANT_HEIGHT, SSD1306_WHITE);
                break;
            case CHILD:
                frameData = child_frames[currentFrame % CHILD_FRAME_COUNT];
                frameCount = CHILD_FRAME_COUNT;
                display.drawBitmap(0, 0, frameData, CHILD_WIDTH, CHILD_HEIGHT, SSD1306_WHITE);
                break;
            case ADULT:
                frameData = adult_frames[currentFrame % ADULT_FRAME_COUNT];
                frameCount = ADULT_FRAME_COUNT;
                display.drawBitmap(0, 0, frameData, ADULT_WIDTH, ADULT_HEIGHT, SSD1306_WHITE);
                break;
            case OLD:
                frameData = old_frames[currentFrame % OLD_FRAME_COUNT];
                frameCount = OLD_FRAME_COUNT;
                display.drawBitmap(0, 0, frameData, OLD_WIDTH, OLD_HEIGHT, SSD1306_WHITE);
                break;
        }
        
        // Display age label
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 0);
        const char* ageNames[] = {"INFANT", "CHILD", "ADULT", "OLD"};
        display.println(ageNames[petAge]);
        
        display.display();
        
        // Increment frame
        currentFrame++;
    }
}

void loop() {
    // This loop runs on Core 1 - handles sensors, camera, WiFi
    
    // Display pet animation on OLED (every 2 seconds with age change every 30 seconds)
    displayPetAnimation();
    
    // Check WiFi connection
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("❌ WiFi disconnected, attempting reconnect...");
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
        while (WiFi.status() != WL_CONNECTED) {
            Serial.print(".");
            delay(300);
        }
        Serial.println("\n✅ WiFi Reconnected");
        Serial.println(WiFi.localIP());
    }
    
    // Debug: Print WiFi status
    static unsigned long lastWiFiCheck = 0;
    if (millis() - lastWiFiCheck > 10000) {  // Every 10 seconds
        lastWiFiCheck = millis();
        Serial.printf("🔗 Core 1 Status | WiFi: %s | IP: %s | Signal: %d dBm\n",
                      WiFi.status() == WL_CONNECTED ? "✅ Connected" : "❌ Disconnected",
                      WiFi.localIP().toString().c_str(),
                      WiFi.RSSI());
        Serial.printf("🎤 Audio Energy: %d | Speech: %s\n", 
                      audioEnergyLevel,
                      speechDetected ? "🗣️  DETECTED" : "🔇 Silent");
    }
    
    // NEW: Collect sensor readings every 100ms internally
    if (millis() - lastInternalReadTime >= INTERNAL_READ_INTERVAL) {
        lastInternalReadTime = millis();
        
        // Only buffer if we have space
        if (sensorBatch.reading_count < SensorDataBatch::MAX_READINGS) {
            SingleReading reading;
            reading.timestamp_ms = millis();
            
            if (mpuAvailable) {
                int16_t ax, ay, az;
                mpu.getAcceleration(&ax, &ay, &az);
                reading.accel_x = ax / 16384.0 * 9.81;
                reading.accel_y = ay / 16384.0 * 9.81;
                reading.accel_z = az / 16384.0 * 9.81;
                
                int16_t gx, gy, gz;
                mpu.getRotation(&gx, &gy, &gz);
                reading.gyro_x = gx / 131.0;
                reading.gyro_y = gy / 131.0;
                reading.gyro_z = gz / 131.0;
            } else {
                reading.accel_x = reading.accel_y = reading.accel_z = 0.0;
                reading.gyro_x = reading.gyro_y = reading.gyro_z = 0.0;
            }
            
            sensorBatch.readings[sensorBatch.reading_count++] = reading;
            
            // Accumulate microphone level
            if (micReady && audioEnergyLevel > 0) {
                totalMicLevel += (20.0 + (audioEnergyLevel / 100.0));
            }
            micReadingCount++;
            
            if (sensorBatch.reading_count % 5 == 0) {
                Serial.printf("📊 Buffered %d readings (ax=%.2f ay=%.2f az=%.2f)\\n", 
                             sensorBatch.reading_count, reading.accel_x, reading.accel_y, reading.accel_z);
            }
        }
    }
    
    // Read sensor data
    SensorData data = readAllSensors();
    
    // Capture image every 20 seconds
    if (cameraReady && (millis() - lastImageCapture >= IMAGE_INTERVAL)) {
        lastImageCapture = millis();
        data.camera_image_b64 = captureImageBase64();
        data.has_new_image = !data.camera_image_b64.isEmpty();
    }
    
    // Check for speech-triggered audio (from Core 0)
    if (xSemaphoreTake(audioMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        if (speechDetected && !detectedAudioData.isEmpty()) {
            data.audio_data_b64 = detectedAudioData;
            data.has_new_audio = true;
            
            Serial.println("🗣️  Speech detected! Preparing to send audio...");
            
            // Clear the detected audio after copying
            detectedAudioData = "";
            speechDetected = false;
        }
        xSemaphoreGive(audioMutex);
    }
    
    // Send all data every 2 seconds ONLY if server is alive AND not uploading image
    if (millis() - lastSendTime >= SEND_INTERVAL) {
        // Skip sensor data while image is uploading (to avoid interference)
        if (isUploadingImage) {
            Serial.println("⏸️  Image uploading... PAUSING sensor data transmission");
        } else {
            lastSendTime = millis();

            if (!isServerAlive()) {
                Serial.println("🛑 Server offline. Skipping data send cycle.");
            } else {
                // NEW: Prepare batch with collected readings
                data.sensor_batch = sensorBatch;
                
                // Calculate average microphone level
                if (micReadingCount > 0) {
                    data.sensor_batch.avg_mic_level = totalMicLevel / micReadingCount;
                    data.sensor_batch.sound_data = audioEnergyLevel;
                } else {
                    data.sensor_batch.avg_mic_level = 0.0;
                    data.sensor_batch.sound_data = 0;
                }
                
                // Print batch summary before sending
                Serial.printf("📤 Sending BATCH: %d readings | Avg mic: %.1f dB | Time span: %ld ms\\n",
                             sensorBatch.reading_count,
                             data.sensor_batch.avg_mic_level,
                             sensorBatch.reading_count > 1 ? 
                             (sensorBatch.readings[sensorBatch.reading_count-1].timestamp_ms - 
                              sensorBatch.readings[0].timestamp_ms) : 0);
                
                // Send sensor data (includes batch of readings)
                bool serverAccepted = sendSensorDataOnly(data);
                
                // RESET batch after sending
                sensorBatch.reading_count = 0;
                totalMicLevel = 0.0;
                micReadingCount = 0;
                
                // Send images only if sensor data was accepted
                if (serverAccepted && data.has_new_image && !data.camera_image_b64.isEmpty()) {
                    sendImageData(data.camera_image_b64);
                }
                
                // Send audio only if sensor data was accepted and speech detected
                if (serverAccepted && data.has_new_audio && !data.audio_data_b64.isEmpty()) {
                    sendAudioData(data.audio_data_b64);
                    Serial.println("🎵 Speech audio sent to server!");
                }
            }
        }
    }
    
    // FIX 5: Poll for events with optimized intervals
    if (millis() - lastEventPoll >= dynamicEventPollInterval) {
        lastEventPoll = millis();
        pollForEvents();
        dynamicEventPollInterval = 5000;  // Standard interval
    }
    
    delay(10);  // Small delay for Core 1
}

// ================= CAMERA INITIALIZATION =================
bool initCamera() {
    Serial.println("Initializing Camera...");
    
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;  // Fixed: was pin_sccb_sda
    config.pin_sscb_scl = SIOC_GPIO_NUM;  // Fixed: was pin_sccb_scl
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_QQVGA; // 160x120 - reduced for power
    config.jpeg_quality = 20;             // Lower quality = less heat
    config.fb_count = 1;                  // Single buffer = less memory
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return false;
    }
    
    cameraReady = true;
    Serial.println("Camera initialized successfully");
    return true;
}

// ================= AUDIO INITIALIZATION =================
bool initAudio() {
    Serial.println("Initializing PDM Microphone...");
    
    // Create I2S channel configuration
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;
    
    // Create new I2S channel
    esp_err_t err = i2s_new_channel(&chan_cfg, NULL, &rx_handle);
    if (err != ESP_OK) {
        Serial.printf("I2S new channel failed: %d\n", err);
        return false;
    }
    
    // Configure PDM RX mode
    i2s_pdm_rx_config_t pdm_rx_cfg = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .clk = PDM_CLK_GPIO,
            .din = PDM_DIN_GPIO,
            .invert_flags = { .clk_inv = false },
        },
    };
    
    err = i2s_channel_init_pdm_rx_mode(rx_handle, &pdm_rx_cfg);
    if (err != ESP_OK) {
        Serial.printf("PDM RX mode init failed: %d\n", err);
        return false;
    }
    
    err = i2s_channel_enable(rx_handle);
    if (err != ESP_OK) {
        Serial.printf("I2S channel enable failed: %d\n", err);
        return false;
    }
    
    // Allocate VAD buffer
    vad_buffer = (int16_t*)malloc(VAD_BUFFER_SIZE * sizeof(int16_t));
    if (!vad_buffer) {
        Serial.println("❌ Failed to allocate VAD buffer");
        return false;
    }
    
    micReady = true;
    Serial.println("✅ PDM Microphone initialized successfully");
    return true;
}

// ================= DUAL-CORE AUDIO MONITORING TASK =================
void audioMonitorTask(void *parameter) {
    Serial.println("🎤 Core 0: Audio monitoring task started");
    
    size_t bytes_read;
    unsigned long speechStartTime = 0;
    unsigned long lastSoundTime = 0;
    bool currentlyRecording = false;
    
    // Audio recording buffer for when speech is detected
    uint8_t* recording_buffer = NULL;
    size_t recorded_bytes = 0;
    const size_t MAX_RECORDING_SIZE = SAMPLE_RATE * 2 * 5; // Max 5 seconds
    
    while (true) {
        if (!micReady) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        
        // Continuously read audio for VAD analysis
        esp_err_t err = i2s_channel_read(rx_handle, vad_buffer, VAD_BUFFER_SIZE * sizeof(int16_t), 
                                        &bytes_read, pdMS_TO_TICKS(10));
        
        if (err != ESP_OK || bytes_read == 0) {
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        
        // Calculate audio energy for Voice Activity Detection
        int32_t energy = 0;
        int samples = bytes_read / sizeof(int16_t);
        
        for (int i = 0; i < samples; i++) {
            int16_t sample = vad_buffer[i];
            energy += abs(sample);
        }
        
        energy = energy / samples; // Average energy
        audioEnergyLevel = energy; // Update global energy level
        
        unsigned long currentTime = millis();
        
        // Voice Activity Detection
        if (energy > VAD_THRESHOLD) {
            lastSoundTime = currentTime;
            
            if (!currentlyRecording) {
                // Start recording when speech detected
                if (speechStartTime == 0) {
                    speechStartTime = currentTime;
                }
                
                // Check if we've had continuous speech for minimum duration
                if (currentTime - speechStartTime >= VAD_MIN_DURATION) {
                    Serial.println("🎤 Core 0: Speech detected! Starting recording...");
                    currentlyRecording = true;
                    
                    // Allocate recording buffer
                    recording_buffer = (uint8_t*)ps_malloc(MAX_RECORDING_SIZE + WAV_HEADER_SIZE);
                    if (recording_buffer) {
                        generate_wav_header(recording_buffer, MAX_RECORDING_SIZE, SAMPLE_RATE);
                        recorded_bytes = WAV_HEADER_SIZE;
                    }
                }
            }
        } else {
            // Reset speech start if energy drops below threshold
            if (currentTime - lastSoundTime > 200) { // 200ms of silence resets start
                speechStartTime = 0;
            }
        }
        
        // If currently recording, add audio data to buffer
        if (currentlyRecording && recording_buffer && 
            (recorded_bytes + bytes_read) < (MAX_RECORDING_SIZE + WAV_HEADER_SIZE)) {
            
            // Apply volume gain and copy to recording buffer
            for (int i = 0; i < samples; i++) {
                int16_t sample = vad_buffer[i];
                int32_t amp = sample << VOLUME_GAIN;
                if (amp > 32767) amp = 32767;
                if (amp < -32768) amp = -32768;
                
                *((int16_t*)(recording_buffer + recorded_bytes)) = amp;
                recorded_bytes += sizeof(int16_t);
            }
        }
        
        // Stop recording after silence timeout or buffer full
        if (currentlyRecording && 
            ((currentTime - lastSoundTime > SILENCE_TIMEOUT) || 
             (recorded_bytes >= (MAX_RECORDING_SIZE + WAV_HEADER_SIZE - 1024)))) {
            
            Serial.printf("🎤 Core 0: Recording complete! %d bytes\n", recorded_bytes);
            
            if (recording_buffer && recorded_bytes > WAV_HEADER_SIZE) {
                // Convert to base64 and store for Core 1
                String audioB64 = base64::encode(recording_buffer, recorded_bytes);
                
                if (xSemaphoreTake(audioMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                    detectedAudioData = audioB64;
                    speechDetected = true;
                    Serial.printf("🎤 Core 0: Audio ready for transmission (%d chars base64)\n", audioB64.length());
                    xSemaphoreGive(audioMutex);
                }
            }
            
            // Clean up
            if (recording_buffer) {
                free(recording_buffer);
                recording_buffer = NULL;
            }
            recorded_bytes = 0;
            currentlyRecording = false;
            speechStartTime = 0;
        }
        
        // FIX 3: Increase VAD delay from 1ms to 10ms (no quality loss, big power savings)
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

SensorData readAllSensors() {
    SensorData data = {0};
    
    if (mpuAvailable) {
        // Read actual sensor data from MPU6050
        int16_t ax, ay, az;
        mpu.getAcceleration(&ax, &ay, &az);
        data.accel_x = ax / 16384.0 * 9.81;
        data.accel_y = ay / 16384.0 * 9.81;
        data.accel_z = az / 16384.0 * 9.81;
        
        int16_t gx, gy, gz;
        mpu.getRotation(&gx, &gy, &gz);
        data.gyro_x = gx / 131.0;
        data.gyro_y = gy / 131.0;
        data.gyro_z = gz / 131.0;
    } else {
        // MPU6050 not available - send zero values with error indication
        data.accel_x = 0.0;
        data.accel_y = 0.0;
        data.accel_z = 0.0;
        data.gyro_x = 0.0;
        data.gyro_y = 0.0;
        data.gyro_z = 0.0;
        Serial.println("⚠️  MPU6050 not available - sending zero values");
    }
    
    // Store calibrated accelerometer values (server will compute direction)
    data.calibrated_ax = data.accel_x;
    data.calibrated_ay = data.accel_y; 
    data.calibrated_az = data.accel_z;
    data.device_orientation = "COMPUTING";  // Placeholder - server will compute
    data.orientation_confidence = 0.0;       // Placeholder - server will compute
    
    // Use real microphone energy level from audio monitoring
    if (micReady && audioEnergyLevel > 0) {
        // Convert audio energy to approximate dB level
        data.mic_level = 20.0 + (audioEnergyLevel / 100.0); // Scale energy to dB range
        data.sound_data = audioEnergyLevel;
    } else {
        // Microphone not ready or no audio data
        data.mic_level = 0.0;
        data.sound_data = 0;
    }
    
    // Initialize image/audio fields
    data.camera_image_b64 = "";
    data.audio_data_b64 = "";
    data.has_new_image = false;
    data.has_new_audio = false;
    
    return data;
}

// ================= SERVER HEALTH CHECK =================
bool isServerAlive() {
    HTTPClient http;
    http.setConnectTimeout(5000);  // Increased from 2000 to 5000ms
    http.setTimeout(8000);         // Increased from 2000 to 8000ms

    if (!http.begin("http://192.168.43.67:5000/api/health")) {
        return false;
    }

    int code = http.GET();
    http.end();

    return (code == 200);
}

// ================= OLED DISPLAY ANIMATION POLLING =================
void getOLEDDisplayFromServer() {
    HTTPClient http;
    http.setConnectTimeout(5000);
    http.setTimeout(5000);

    if (!http.begin(oledDisplayUrl)) {
        return;  // Silently fail, keep showing current animation
    }

    int httpCode = http.GET();
    
    if (httpCode == 200) {
        String response = http.getString();
        
        // Parse JSON response: {"animation_type": "pet", "animation_id": 0-3, "animation_name": "INFANT|CHILD|ADULT|OLD"}
        DynamicJsonDocument doc(512);
        DeserializationError error = deserializeJson(doc, response);
        
        if (!error && doc.containsKey("animation_id")) {
            int newAnimationId = doc["animation_id"].as<int>();
            String animationType = doc["animation_type"] | "pet";
            String animationName = doc["animation_name"] | "UNKNOWN";
            
            if (newAnimationId >= 0 && newAnimationId <= 3) {
                // Only update if changed
                if ((int)petAge != newAnimationId) {
                    petAge = (PetAge)newAnimationId;
                    Serial.printf("🎬 OLED animation from server: %s (type: %s, id: %d)\n", 
                                  animationName.c_str(), animationType.c_str(), newAnimationId);
                }
            }
        }
    }
    
    http.end();
}

// ================= CAMERA FUNCTIONS =================
String captureImageBase64() {
    Serial.println("📸 Capturing image...");
    
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("❌ Camera capture failed");
        return "";
    }
    
    // Convert to base64
    String imageB64 = base64::encode(fb->buf, fb->len);
    
    Serial.printf("✅ Image captured: %d bytes → %d chars base64\n", fb->len, imageB64.length());
    
    esp_camera_fb_return(fb);
    return imageB64;
}

// ================= AUDIO FUNCTIONS =================
// Note: Audio recording is now handled by audioMonitorTask on Core 0
// This function is kept for compatibility but not used in dual-core mode

String recordAudioBase64() {
    Serial.println("⚠️  recordAudioBase64() called, but using VAD on Core 0 instead");
    return "";  // Return empty - audio handled by voice detection
}

// ================= UNIFIED DATA TRANSMISSION =================
bool sendSensorDataOnly(SensorData data) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.printf("❌ WiFi not connected, skipping send\n");
        return false;
    }
    
    HTTPClient http;
    http.setConnectTimeout(5000);
    http.setTimeout(10000);
    
    if (!http.begin(serverUrl)) {
        Serial.println("❌ Failed to begin HTTP connection");
        return false;
    }
    
    http.addHeader("Content-Type", "application/json");
    
    // Larger JSON payload - sensor data + batch of readings
    StaticJsonDocument<4096> jsonDoc;
    
    jsonDoc["accel_x"] = data.accel_x;
    jsonDoc["accel_y"] = data.accel_y;
    jsonDoc["accel_z"] = data.accel_z;
    jsonDoc["gyro_x"] = data.gyro_x;
    jsonDoc["gyro_y"] = data.gyro_y;
    jsonDoc["gyro_z"] = data.gyro_z;
    jsonDoc["mic_level"] = data.mic_level;
    jsonDoc["sound_data"] = data.sound_data;
    
    // Add sensor batch with all buffered readings
    JsonObject batchObj = jsonDoc.createNestedObject("sensor_batch");
    batchObj["reading_count"] = data.sensor_batch.reading_count;
    batchObj["avg_mic_level"] = data.sensor_batch.avg_mic_level;
    batchObj["sound_data"] = data.sensor_batch.sound_data;
    
    // Serialize all readings in the batch
    JsonArray readingsArray = batchObj.createNestedArray("readings");
    for (int i = 0; i < data.sensor_batch.reading_count; i++) {
        JsonObject readingObj = readingsArray.createNestedObject();
        readingObj["timestamp_ms"] = data.sensor_batch.readings[i].timestamp_ms;
        readingObj["accel_x"] = data.sensor_batch.readings[i].accel_x;
        readingObj["accel_y"] = data.sensor_batch.readings[i].accel_y;
        readingObj["accel_z"] = data.sensor_batch.readings[i].accel_z;
        readingObj["gyro_x"] = data.sensor_batch.readings[i].gyro_x;
        readingObj["gyro_y"] = data.sensor_batch.readings[i].gyro_y;
        readingObj["gyro_z"] = data.sensor_batch.readings[i].gyro_z;
    }
    
    String payload;
    serializeJson(jsonDoc, payload);
    
    Serial.printf("📊 Sending sensor data: %d bytes\n", payload.length());
    
    int httpCode = http.POST(payload);
    
    if (httpCode == 200) {
        Serial.println("✅ Sensor data sent!");
        Serial.printf("    Accel: X=%.2f, Y=%.2f, Z=%.2f m/s²\n", 
                     data.accel_x, data.accel_y, data.accel_z);
        Serial.printf("    Gyro:  X=%.2f, Y=%.2f, Z=%.2f °/s\n", 
                     data.gyro_x, data.gyro_y, data.gyro_z);
        Serial.printf("    Orient: %s (%.1f%% confidence)\n", 
                     data.device_orientation.c_str(), data.orientation_confidence);
        digitalWrite(LED_PIN, HIGH);
        delay(50);
        digitalWrite(LED_PIN, LOW);
    } else {
        Serial.printf("❌ HTTP error: %s\n", http.errorToString(httpCode).c_str());
    }
    
    http.end();
    return (httpCode == 200);
}

void sendImageData(String imageBase64) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("⚠️  WiFi not connected, skipping image upload");
        return;
    }
    
    // Check free heap before capturing image
    size_t freeHeap = ESP.getFreeHeap();
    if (freeHeap < 50000) {  // Need at least 50KB free for image processing
        Serial.printf("⚠️  Low memory (%d bytes), skipping image capture\n", freeHeap);
        return;
    }
    
    // Capture fresh image
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("❌ Camera capture failed");
        isUploadingImage = false;  // Resume on error
        return;
    }
    
    // 🔴 SET FLAG: Pause sensor data transmission
    isUploadingImage = true;
    Serial.println("🔴 STARTING IMAGE UPLOAD - Sensor data PAUSED");
    Serial.printf("🖼️ Sending image: %d bytes (binary stream)\n", fb->len);
    
    WiFiClient client;
    HTTPClient http;
    
    if (!http.begin(client, "http://192.168.43.67:5000/upload")) {
        Serial.println("❌ Failed to connect to server");
        esp_camera_fb_return(fb);
        isUploadingImage = false;  // Resume on error
        return;
    }
    
    // Configure HTTP client for large image data
    http.setTimeout(20000);          // 20 second timeout for large images
    http.setConnectTimeout(5000);    // 5 second connection timeout
    http.addHeader("Content-Type", "application/octet-stream");
    http.addHeader("Connection", "close");  // Close connection after upload
    
    // 🔥 Use sendRequest() to stream binary data directly
    int httpCode = http.sendRequest("POST", fb->buf, fb->len);
    
    if (httpCode == 200) {
        Serial.println("✅ Image sent successfully!");
        Serial.printf("   Sent %d bytes to server\n", fb->len);
        digitalWrite(LED_PIN, HIGH);
        delay(50);
        digitalWrite(LED_PIN, LOW);
    } else if (httpCode > 0) {
        // HTTP error response from server
        Serial.printf("❌ HTTP Error: %d (%s)\n", httpCode, http.errorToString(httpCode).c_str());
        String payload = http.getString();
        if (payload.length() > 0) {
            Serial.printf("   Server response: %s\n", payload.c_str());
        }
    } else {
        // Connection or timeout error
        Serial.printf("❌ Connection Error: %d (%s)\n", httpCode, http.errorToString(httpCode).c_str());
        Serial.println("   Check WiFi connection and server availability");
    }
    
    http.end();
    esp_camera_fb_return(fb);
    delay(200);  // Server breathing room after large upload
    
    // 🟢 CLEAR FLAG: Resume sensor data transmission
    isUploadingImage = false;
    Serial.println("🟢 IMAGE UPLOAD COMPLETE - Sensor data RESUMED");
}

void sendAudioData(String audioBase64) {
    if (WiFi.status() != WL_CONNECTED) return;
    
    if (audioBase64.length() == 0) {
        Serial.println("⚠️  Audio data empty, skipping");
        return;
    }
    
    Serial.printf("🎵 Sending audio: %d bytes base64\n", audioBase64.length());
    
    HTTPClient http;
    http.setConnectTimeout(5000);
    http.setTimeout(15000);  // Shorter timeout for smaller audio
    
    if (!http.begin("http://192.168.43.67:5000/upload-audio")) {
        Serial.println("❌ Failed to connect to audio server");
        return;
    }
    
    http.addHeader("Content-Type", "application/json");
    
    // Build JSON manually to avoid buffer overflow
    String payload = "{\"audio_data\":\"";
    payload += audioBase64;
    payload += "\"}";
    
    Serial.printf("📨 Payload size: %d bytes\n", payload.length());
    
    int httpCode = http.POST(payload);
    
    if (httpCode == 200) {
        Serial.println("✅ Audio data sent!");
    } else {
        Serial.printf("❌ Audio send failed: %d (%s)\n", httpCode, http.errorToString(httpCode).c_str());
    }
    
    http.end();
    delay(200);  // Server breathing room after audio upload
}

void sendAllDataToServer(SensorData data) {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.printf("❌ WiFi not connected (status=%d), skipping data send\n", WiFi.status());
        return;
    }
    
    Serial.printf("🌐 Connecting to server: %s\n", serverUrl);
    
    HTTPClient http;
    http.setConnectTimeout(10000);  // 10 second connection timeout
    http.setTimeout(15000);          // 15 second read timeout (for large payloads)
    
    if (!http.begin(serverUrl)) {
        Serial.println("❌ Failed to begin HTTP connection");
        return;
    }
    
    http.addHeader("Content-Type", "application/json");
    
    // Create comprehensive JSON payload
    StaticJsonDocument<8192> jsonDoc;  // Increased size for image/audio data
    
    // Basic sensor data
    jsonDoc["accel_x"] = data.accel_x;
    jsonDoc["accel_y"] = data.accel_y;
    jsonDoc["accel_z"] = data.accel_z;
    jsonDoc["gyro_x"] = data.gyro_x;
    jsonDoc["gyro_y"] = data.gyro_y;
    jsonDoc["gyro_z"] = data.gyro_z;
    jsonDoc["mic_level"] = data.mic_level;
    jsonDoc["sound_data"] = data.sound_data;
    
    // Metadata
    jsonDoc["timestamp"] = millis();
    jsonDoc["device_id"] = "esp32_xiao_s3";
    
    // Add image data if available
    if (data.has_new_image && !data.camera_image_b64.isEmpty()) {
        jsonDoc["camera_image"] = data.camera_image_b64;
        Serial.println("📸 Including image data in payload");
    }
    
    // Add audio data if available  
    if (data.has_new_audio && !data.audio_data_b64.isEmpty()) {
        jsonDoc["audio_data"] = data.audio_data_b64;
        Serial.println("🎵 Including audio data in payload");
    }
    
    String payload;
    serializeJson(jsonDoc, payload);
    
    Serial.printf("\n📊 Sending data: %d bytes\n", payload.length());
    Serial.printf("    Sensors: ✅ | Image: %s | Audio: %s\n",
                  data.has_new_image ? "✅" : "⬜",
                  data.has_new_audio ? "✅" : "⬜");
    Serial.println("⏳ Waiting for server response...");
    
    int httpCode = http.POST(payload);
    
    if (httpCode > 0) {
        Serial.printf("📤 POST Response: %d\n", httpCode);
        if (httpCode == 200) {
            Serial.println("✅ All data sent successfully!");
            Serial.printf("    Accel: X=%.2f, Y=%.2f, Z=%.2f m/s²\n", 
                         data.accel_x, data.accel_y, data.accel_z);
            Serial.printf("    Gyro:  X=%.2f, Y=%.2f, Z=%.2f °/s\n", 
                         data.gyro_x, data.gyro_y, data.gyro_z);
            Serial.printf("    Mic:   %.1f dB\n", data.mic_level);
            
            // Success LED blink
            digitalWrite(LED_PIN, HIGH);
            delay(100);
            digitalWrite(LED_PIN, LOW);
        } else {
            Serial.printf("❌ Server error: %s\n", http.getString().c_str());
        }
    } else {
        Serial.printf("❌ HTTP error: %s\n", http.errorToString(httpCode).c_str());
    }
    
    http.end();
}

// WAV Header generation function
void generate_wav_header(uint8_t* wav_header, uint32_t wav_size, uint32_t sample_rate) {
    uint32_t file_size = wav_size + WAV_HEADER_SIZE - 8;
    uint32_t byte_rate = sample_rate * SAMPLE_BITS / 8;

    const uint8_t header[] = {
        'R','I','F','F',
        file_size, file_size >> 8, file_size >> 16, file_size >> 24,
        'W','A','V','E','f','m','t',' ',
        0x10,0x00,0x00,0x00,
        0x01,0x00,
        0x01,0x00,
        sample_rate, sample_rate >> 8, sample_rate >> 16, sample_rate >> 24,
        byte_rate, byte_rate >> 8, byte_rate >> 16, byte_rate >> 24,
        0x02,0x00,
        0x10,0x00,
        'd','a','t','a',
        wav_size, wav_size >> 8, wav_size >> 16, wav_size >> 24,
    };

    memcpy(wav_header, header, sizeof(header));
}

// ================= EVENT POLLING FUNCTIONS =================
void pollForEvents() {
    HTTPClient http;
    
    Serial.println("🔍 Polling for important events...");
    
    if (!http.begin(eventsUrl)) {
        Serial.println("❌ Failed to initialize HTTP client for events");
        return;
    }
    
    // Set timeout
    http.setTimeout(5000);
    
    // Add headers
    http.addHeader("Content-Type", "application/json");
    http.addHeader("User-Agent", "ESP32-Dashboard/2.0");
    
    // Make GET request
    int httpCode = http.GET();
    
    if (httpCode > 0) {
        Serial.printf("📡 Server response: %d\n", httpCode);
        
        if (httpCode == HTTP_CODE_OK) {
            String response = http.getString();
            
            if (response.length() > 0) {
                Serial.println("📋 Server Response:");
                Serial.println("-------------------");
                
                // Parse JSON response
                DynamicJsonDocument doc(2048);
                DeserializationError error = deserializeJson(doc, response);
                
                if (!error) {
                    if (doc.containsKey("events")) {
                        JsonArray events = doc["events"];
                        
                        if (events.size() > 0) {
                            Serial.printf("🚨 FOUND %d IMPORTANT EVENT(S):\n", events.size());
                            Serial.println("========================================");
                            
                            for (int i = 0; i < events.size(); i++) {
                                JsonObject event = events[i];
                                
                                int event_id = event["id"].as<int>();
                                const char* event_type = event["event_type"];
                                const char* message = event["message"];
                                const char* created_at = event["created_at"];
                                
                                Serial.printf("   🚨 EVENT #%d:\n", i + 1);
                                Serial.printf("     ID: %d\n", event_id);
                                Serial.printf("     Type: %s\n", event_type);
                                Serial.printf("     Message: %s\n", message);
                                Serial.printf("     Time: %s\n", created_at);
                                Serial.println();
                                
                                // Process the event
                                processEvent(event_type, message);
                                
                                // Acknowledge event received
                                acknowledgeEvent(event_id);
                                
                                delay(100);  // Small delay between events
                            }
                        } else {
                            Serial.println("✅ No new important events (all quiet)");
                        }
                    }
                    
                    if (doc.containsKey("message")) {
                        Serial.printf("💬 Status: %s\n", doc["message"].as<const char*>());
                    }
                } else {
                    Serial.println("❌ JSON parsing error");
                    Serial.println(response);
                }
                
                Serial.println("-------------------");
            } else {
                Serial.println("✅ Empty response (no events)");
            }
        } else {
            Serial.printf("⚠️ Unexpected response code: %d\n", httpCode);
        }
    } else {
        Serial.printf("❌ HTTP request failed: %s\n", http.errorToString(httpCode).c_str());
        Serial.println("   Server might be down or unreachable");
    }
    
    http.end();
    Serial.println("");
}

void processEvent(const char* event_type, const char* message) {
    Serial.println("🔧 Processing event...");
    
    // Simple event processing - you can extend this based on your needs
    if (strcmp(event_type, "high_sound") == 0) {
        Serial.println("🔊 High sound detected - might want to take action!");
        digitalWrite(LED_PIN, HIGH);  // Turn on LED for high sound
        delay(200);
        digitalWrite(LED_PIN, LOW);   // Blink LED
    }
    else if (strcmp(event_type, "sudden_motion") == 0) {
        Serial.println("🏃 Sudden motion detected - something's happening!");
        // Blink LED multiple times for motion
        for (int i = 0; i < 3; i++) {
            digitalWrite(LED_PIN, HIGH);
            delay(100);
            digitalWrite(LED_PIN, LOW);
            delay(100);
        }
    }
    else if (strcmp(event_type, "alert") == 0) {
        Serial.println("⚠️ Generic alert received!");
        digitalWrite(LED_PIN, HIGH);  // Solid LED for alert
        delay(500);
        digitalWrite(LED_PIN, LOW);
    }
    else {
        Serial.printf("❓ Unknown event type: %s\n", event_type);
    }
    
    Serial.printf("   📝 Event message: %s\n", message);
}

void acknowledgeEvent(int event_id) {
    HTTPClient http;
    
    Serial.printf("📤 Acknowledging event #%d...\n", event_id);
    
    if (!http.begin(eventReceivedUrl)) {
        Serial.println("❌ Failed to initialize HTTP client for acknowledgment");
        return;
    }
    
    // Set timeout and headers
    http.setTimeout(5000);
    http.addHeader("Content-Type", "application/json");
    http.addHeader("User-Agent", "ESP32-Dashboard/2.0");
    
    // Create JSON payload
    DynamicJsonDocument doc(256);
    doc["device_id"] = "ESP32_001";
    doc["event_id"] = event_id;
    doc["status"] = "received";
    doc["timestamp"] = millis();  // Simple timestamp
    
    String jsonString;
    serializeJson(doc, jsonString);
    
    // Send POST request
    int httpCode = http.POST(jsonString);
    
    if (httpCode > 0) {
        if (httpCode == HTTP_CODE_OK || httpCode == 200) {
            String response = http.getString();
            Serial.printf("✅ Event acknowledged successfully: %s\n", response.c_str());
        } else {
            Serial.printf("⚠️ Acknowledgment response: %d\n", httpCode);
        }
    } else {
        Serial.printf("❌ Acknowledgment failed: %s\n", http.errorToString(httpCode).c_str());
    }
    
    http.end();
}

/*
WIRING DIAGRAM for XIAO ESP32 S3 Sense: 
===========================================

🔌 Built-in Components (No Wiring Needed):
- Camera Module (OV2640) - Built into XIAO ESP32 S3 Sense
- PDM Microphone - Built into XIAO ESP32 S3 Sense  
- PSRAM - Built into XIAO ESP32 S3 Sense

🔧 External Components to Connect:

MPU6050 (Accelerometer/Gyroscope):
- VCC -> 3.3V
- GND -> GND  
- SDA -> GPIO 21 (I2C Data)
- SCL -> GPIO 22 (I2C Clock)
- INT -> Not connected (optional)

LED Indicator:
- + -> GPIO 2 (with 220Ω resistor)
- - -> GND

📋 PIN ASSIGNMENTS (XIAO ESP32 S3 Sense):
==========================================

Camera Pins (Built-in OV2640):
- XCLK  -> GPIO 10
- SIOD  -> GPIO 40 (I2C SDA)
- SIOC  -> GPIO 39 (I2C SCL) 
- Y9    -> GPIO 48
- Y8    -> GPIO 11
- Y7    -> GPIO 12
- Y6    -> GPIO 14
- Y5    -> GPIO 16
- Y4    -> GPIO 18
- Y3    -> GPIO 17
- Y2    -> GPIO 15
- VSYNC -> GPIO 38
- HREF  -> GPIO 47
- PCLK  -> GPIO 13

I2S PDM Microphone (Built-in):
- CLK -> GPIO 42 (I2S WS)
- DATA-> GPIO 41 (I2S SD)

MPU6050 (External):
- SDA -> GPIO 5  (XIAO ESP32 S3)
- SCL -> GPIO 6  (XIAO ESP32 S3)
  
Alternative I2C pins if GPIO 6/7 don't work:
- SDA -> GPIO 21, SCL -> GPIO 22 (Generic ESP32)
- Check your specific board's pinout diagram

Other:
- LED -> GPIO 2

⚠️  IMPORTANT NOTES:
===================
1. XIAO ESP32 S3 Sense has BUILT-IN camera and microphone
2. No external camera/mic wiring needed
3. Only connect MPU6050 externally via I2C
4. Camera uses JPEG compression for efficient transmission
5. 🎤 SMART AUDIO: Voice Activity Detection with dual-core processing
6. All data sent to dashboard via WiFi

🚀 FEATURES:
============
- ✅ MPU6050 sensor data (every 1 second) - Core 1
- ✅ Camera images (every 7 seconds) - Core 1
- ✅ 🎤 SMART AUDIO: Voice Activity Detection - Core 0
- ✅ Dual-core processing for optimal performance
- ✅ Real-time dashboard updates
- ✅ WiFi connectivity with auto-reconnect
- ✅ LED status indicators
- ✅ PSRAM for audio/image buffering

🧠 DUAL-CORE ARCHITECTURE:
==========================
**Core 0 (Audio Core):**
- Continuous microphone monitoring
- Voice Activity Detection (VAD)
- Energy-based speech detection
- Automatic audio recording when speech detected
- Real-time audio processing (high priority)

**Core 1 (Main Core):**
- WiFi management and HTTP transmission
- Sensor data collection (MPU6050)
- Camera image capture
- Dashboard communications
- LED status indicators

🎤 INTELLIGENT AUDIO SYSTEM:
============================
**Voice Activity Detection:**
- Continuously monitors audio energy levels
- Detects speech when energy > 1000 threshold
- Requires minimum 500ms of continuous speech
- Records up to 5 seconds of audio
- Stops recording after 2 seconds of silence
- Only transmits audio when speech is detected

**Energy-Based Detection:**
```
Audio Energy > 1000    → Speech Detected 🗣️
Audio Energy < 1000    → Silent 🔇
Continuous Speech 500ms+ → Start Recording 🎙️
Silence 2000ms+ → Stop Recording & Send 📤
```

**Benefits:**
- ⚡ No bandwidth wasted on silent audio
- 🔋 Power efficient - only records when needed
- 📡 Real-time speech detection and transmission
- 🎯 High accuracy voice activity detection
- 🚀 Multi-core performance optimization

📊 DATA TRANSMISSION SCHEDULE:
==============================
**Real-time (Core 1):**
- Sensor readings: Every 1 second (always)
- Camera images: Every 7 seconds

**Event-driven (Core 0 → Core 1):**
- Audio: Only when speech detected
- Voice detection: Continuous monitoring
- Inter-core communication via mutex/semaphores

🔧 SMART THRESHOLDS:
===================
- VAD_THRESHOLD: 1000 (adjust based on environment)
- VAD_MIN_DURATION: 500ms (minimum speech length)
- SILENCE_TIMEOUT: 2000ms (stop recording delay)
- MAX_RECORDING: 5 seconds (prevent buffer overflow)

🎛️ TUNING VOICE DETECTION:
===========================
**Quiet Environment:** Lower VAD_THRESHOLD to 500-800
**Noisy Environment:** Raise VAD_THRESHOLD to 1500-2000
**Sensitive Detection:** Decrease VAD_MIN_DURATION to 300ms
**Less False Triggers:** Increase VAD_MIN_DURATION to 800ms

💾 MEMORY USAGE:
===============
- Images: ~1-3KB (QQVGA 160x120, quality 20)
- Audio: ~32KB per 1-second recording (only when speech)
- VAD Buffer: 1KB for real-time energy analysis
- PSRAM: Dynamic allocation for recordings
- Automatic cleanup after transmission

🌐 NETWORK ENDPOINTS:
====================
- Sensors: POST /api/sensor-data (small, frequent)
- Images: POST /upload (binary, every 7s)
- Audio: POST /upload-audio (JSON, speech-triggered)

⚡ PERFORMANCE BENEFITS:
========================
1. **Dual-Core**: Audio processing doesn't block main operations
2. **Event-Driven**: Audio only sent when speech detected
3. **Energy-Efficient**: No continuous audio transmission
4. **Real-Time**: Voice detection with minimal latency
5. **Intelligent**: Automatic silence detection and recording stop

🔧 CONFIGURATION:
=================
Update these before uploading:
1. WIFI_SSID and WIFI_PASSWORD
2. Server IP address: "192.168.43.67:5000"
3. Adjust VAD_THRESHOLD for your environment
4. Three separate optimized endpoints for different data types

**Data Sending Examples:**
- Sensor readings: Every 1 second (always) - 146 bytes JSON
- Camera images: Every 7 seconds - 1-3KB binary
- Audio recordings: Only when you speak - 32KB+ base64
- Silence periods: 0 bytes audio transmission ✨
*/
