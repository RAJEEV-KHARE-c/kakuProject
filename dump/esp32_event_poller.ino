/*
ESP32 Event Poller - Simple Testing Client
This sketch polls the server for important events and displays them on serial monitor

Required Libraries:
- WiFi
- HTTPClient
- ArduinoJson

Install via Arduino IDE: Sketch > Include Library > Manage Libraries
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ================= WIFI CONFIGURATION =================
#define WIFI_SSID     "123"
#define WIFI_PASSWORD "KUNAL 26"

// ================= SERVER CONFIGURATION =================
const char* serverUrl = "http://192.168.61.252:5000/api/events?device_id=ESP32_001";  // Events endpoint
const char* healthUrl = "http://192.168.61.252:5000/api/health";  // Health check
const char* eventReceivedUrl = "http://192.168.61.252:5000/api/device/event/received";  // Event acknowledgment

// ================= TIMING =================
unsigned long lastPollTime = 0;
const unsigned long pollInterval = 5000;  // Poll every 5 seconds
unsigned long lastHealthCheck = 0;
const unsigned long healthInterval = 30000;  // Health check every 30 seconds

// ================= FUNCTIONS =================

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("=================================");
    Serial.println("🚀 ESP32 Event Poller Starting");
    Serial.println("=================================");
    
    // Connect to WiFi
    connectToWiFi();
    
    Serial.println("📡 Ready to poll for events...");
    Serial.println("");
}

void loop() {
    // Check WiFi connection
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("❌ WiFi disconnected! Reconnecting...");
        connectToWiFi();
        return;
    }
    
    unsigned long currentTime = millis();
    
    // Poll for events
    if (currentTime - lastPollTime >= pollInterval) {
        pollForEvents();
        lastPollTime = currentTime;
    }
    
    // Health check
    if (currentTime - lastHealthCheck >= healthInterval) {
        checkServerHealth();
        lastHealthCheck = currentTime;
    }
    
    delay(100);  // Small delay to prevent watchdog issues
}

void connectToWiFi() {
    Serial.println("🌐 Connecting to WiFi...");
    Serial.printf("   SSID: %s\n", WIFI_SSID);
    
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(1000);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println(" ✅ Connected!");
        Serial.printf("   IP Address: %s\n", WiFi.localIP().toString().c_str());
        Serial.printf("   Signal: %d dBm\n", WiFi.RSSI());
        Serial.println("");
    } else {
        Serial.println(" ❌ Failed to connect!");
        Serial.println("   Retrying in 5 seconds...");
        delay(5000);
    }
}

void pollForEvents() {
    HTTPClient http;
    
    Serial.println("🔍 Polling for important events...");
    
    if (!http.begin(serverUrl)) {
        Serial.println("❌ Failed to initialize HTTP client for events");
        return;
    }
    
    // Set timeout
    http.setTimeout(5000);
    
    // Add headers
    http.addHeader("Content-Type", "application/json");
    http.addHeader("User-Agent", "ESP32-EventPoller/2.0");
    
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

void checkServerHealth() {
    HTTPClient http;
    
    Serial.println("💓 Checking server health...");
    
    if (!http.begin(healthUrl)) {
        Serial.println("❌ Failed to initialize HTTP client for health check");
        return;
    }
    
    http.setTimeout(3000);
    int httpCode = http.GET();
    
    if (httpCode == HTTP_CODE_OK) {
        String response = http.getString();
        Serial.printf("✅ Server is healthy: %s\n", response.c_str());
    } else if (httpCode > 0) {
        Serial.printf("⚠️ Server health issue: %d\n", httpCode);
    } else {
        Serial.printf("❌ Server unreachable: %s\n", http.errorToString(httpCode).c_str());
    }
    
    http.end();
    Serial.println("");
}

void processEvent(const char* event_type, const char* message) {
    Serial.println("🔧 Processing event...");
    
    // Simple event processing - you can extend this based on your needs
    if (strcmp(event_type, "high_sound") == 0) {
        Serial.println("🔊 High sound detected - might want to take action!");
        // Add your logic here (LED blink, buzzer, etc.)
    }
    else if (strcmp(event_type, "sudden_motion") == 0) {
        Serial.println("🏃 Sudden motion detected - something's happening!");
        // Add your logic here (alert, trigger camera, etc.)
    }
    else if (strcmp(event_type, "alert") == 0) {
        Serial.println("⚠️ Generic alert received!");
        // Add your logic here
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
    http.addHeader("User-Agent", "ESP32-EventPoller/2.0");
    
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
=== USAGE INSTRUCTIONS ===

1. Upload this sketch to your ESP32
2. Open Serial Monitor (115200 baud)
3. Watch for event polling every 5 seconds
4. Server health checks every 30 seconds

=== SERVER ENDPOINT ===

The ESP32 will poll: GET /api/events

Expected JSON response format:
{
  "events": [
    {
      "type": "alert",
      "message": "Motion detected in living room",
      "timestamp": "2026-02-11T10:30:00Z",
      "priority": "high"
    },
    {
      "type": "sensor",
      "message": "Temperature above threshold: 35°C",
      "timestamp": "2026-02-11T10:25:00Z", 
      "priority": "medium"
    }
  ]
}

Or for no events:
{
  "events": [],
  "message": "No new events"
}

=== TESTING ===

You can test the endpoint manually:
curl http://192.168.61.252:5000/api/events

*/