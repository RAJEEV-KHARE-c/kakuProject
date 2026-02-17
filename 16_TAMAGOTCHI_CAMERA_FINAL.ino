/*
 * KAKU with Camera Detection & Egg Hatching Animation
 * - Egg hatching animation at startup
 * - Person NEAR = Happy face
 * - Person FAR/None = Idle face with blinking
 * - All life stages: INFANT → CHILD → TEEN → ADULT → OLD
 */

#include "esp_camera.h"
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <MPU6050.h>
#include <Preferences.h>
#include <time.h>
#include <sys/time.h>

// ============ OLED SETUP ============
#define SCREEN_WIDTH 64
#define SCREEN_HEIGHT 32
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C
#define SDA_PIN 5
#define SCL_PIN 6

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
MPU6050 mpu;

// ============ SCREEN STATE ============
enum ScreenState {
  MAIN_SCREEN,
  MENU_SCREEN
};

ScreenState currentScreen = MAIN_SCREEN;

// ============ CAMERA PINS ============
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// ============ CHARACTER STATES ============
enum CharacterState {
  IDLE,
  BLINKING,
  HAPPY,
  SAD,
  SLEEPING
};

enum LifeStage {
  INFANT,
  CHILD,
  TEEN,
  ADULT,
  OLD
};

enum MenuPage {
  MENU_STATUS,
  MENU_FOOD,
  MENU_CLEAN,
  MENU_HEALTH,
  MENU_DISCIPLINE,
  MENU_TOTAL
};



bool personDetected = false;
// ============ VARIABLES ============
bool cameraReady = false;
CharacterState currentState = IDLE;
LifeStage lifeStage = INFANT;

// Camera detection
int detectionLevel = 0;
unsigned long lastCheckTime = 0;
const int CHECK_INTERVAL = 300;

// Thresholds for detection
const int NEAR_THRESHOLD = 1500;  // Lowered for better detection - was 2500
const int FAR_THRESHOLD = 500;

// Camera black detection for menu
#define BLACK_BRIGHTNESS_TH 25
#define BLACK_CONTRAST_TH 25
#define HOLD_TIME_MS 3000         // 3 sec = menu toggle
#define QUICK_COVER_TIME_MS 500   // 0.5 sec = menu page change
#define COOLDOWN_MS 1500
bool blackActive = false;
unsigned long blackStartTime = 0;
unsigned long lastMenuChangeTime = 0;
unsigned long lastMenuPageChangeTime = 0;

// Auto-sleep system
#define SLEEP_DELAY_MS 5400000     // 1.5 hours after feeding
#define SLEEP_DURATION_MS 600000   // 10 minutes sleep time
bool autoSleepMode = false;
bool sleepScheduled = false;
unsigned long sleepScheduleTime = 0;
unsigned long sleepStartTime = 0;

// Animation
unsigned long lastAnimationTime = 0;
int currentFrame = 0;
const int ANIMATION_DELAY = 500;

// ============ FEEDING SYSTEM ============
#define SHAKE_THRESHOLD 1.2
#define REQUIRED_SHAKES 2
#define SHAKE_TIMEOUT 2000
#define INITIAL_MAX_FEED_LEVEL 1
#define ABSOLUTE_MAX_FEED_LEVEL 6
#define HUNGER_INCREASE_INTERVAL 3600000  // 1 hour in milliseconds
#define DRINK_TIME_MS 5000

int maxFeedLevel = INITIAL_MAX_FEED_LEVEL;
int feedLevel = INITIAL_MAX_FEED_LEVEL;  // Start full
int shakeCount = 0;
float lastAx = 0;
float lastAy = 0;
unsigned long lastShakeTime = 0;
unsigned long lastFeedTime = 0;  // Track last time fed

// ============ AUTOMATIC HUNGER SYSTEM ============
// Age-based hunger timing (in milliseconds)
#define INFANT_HUNGER_MIN 1800000    // 30 minutes
#define INFANT_HUNGER_MAX 3600000    // 60 minutes
#define CHILD_HUNGER_MIN 5400000     // 1.5 hours
#define CHILD_HUNGER_MAX 7200000     // 2 hours
#define TEEN_HUNGER_MIN 10800000     // 3 hours
#define TEEN_HUNGER_MAX 12600000     // 3.5 hours
#define ADULT_HUNGER_MIN 10800000    // 3 hours
#define ADULT_HUNGER_MAX 12600000    // 3.5 hours
#define OLD_HUNGER_MIN 18000000      // 5 hours *UPDATED*
#define OLD_HUNGER_MAX 18000000      // 5 hours *UPDATED*

unsigned long nextHungerTime = 0;
bool autoHungerInitialized = false;

// ============ STEP COUNTER ============
#define STEP_THRESHOLD 1.30
#define STEP_DELAY 350
#define STEPS_TO_HUNGRY 5
#define STEPS_PER_AGE 7000

int stepCount = 0;        // Global step counter for walking
int feedStepCount = 0;    // Feed-only counter (counts in MENU when hungry)
int characterAge = 1;     // Age starts at 1, increments on each wake-up
unsigned long lastStepTime = 0;
float filteredAcc = 1.0;
bool wasBelow = true;
bool isHungry = false;
bool isFatty = false;      // Character becomes fatty if no walk for 3 days
unsigned long lastWalkTime = 0;  // Track last time character walked
const unsigned long NO_WALK_THRESHOLD = 259200000;  // 3 days in milliseconds

// ============ RTC TIME TRACKING (for day/hour based events) ============ *UPDATED*
time_t currentDay = 0;           // Current day (tracked via RTC)
time_t lastRecordedDay = 0;      // Last day we checked
unsigned long dirtyStartTime = 0; // Track when baby became dirty
unsigned long lastLazyCheckDay = 0; // Track last day we checked for 1-day lazy penalty
bool deviceStarted = false;      // Flag for 10-day health animation feature
unsigned long deviceStartTime = 0;  // When device first started

// ============ HEALTH ANIMATION LEVELS ============ *UPDATED*
enum HealthLevel {
  HEALTH_WEAK = 0,       // health < 2000
  HEALTH_TIRED = 1,      // steps < 8000/day OR health 2000-4000
  HEALTH_AVERAGE = 2,    // health 4000-6000 (default at start)
  HEALTH_STRONG = 3,     // avg 6000 steps/day
  HEALTH_STRONG_PLUS = 4, // avg 8000 steps/day
  HEALTH_MAX = 5         // avg 10000+ steps/day
};
HealthLevel currentHealthLevel = HEALTH_AVERAGE;
int stepsPerDay = 0;     // Track daily step average
int daysTracked = 0;     // Number of days device has been running

// ============ MENU SYSTEM ============
MenuPage currentMenuPage = MENU_STATUS;
unsigned long lastMenuCycleTime = 0;
const unsigned long MENU_CYCLE_INTERVAL = 2000; // 2 seconds

// ============ CARE SYSTEM ============
bool isDirty = false;       // Poop present
bool isSick = false;        // Health issue
uint8_t discipline = 50;    // 0-100 (starts at 50)
int healthScore = 10000;    // Hidden health score (max 10000, decreases when neglected)

// Care system timing
unsigned long lastPoopCheckTime = 0;
// unsigned long dirtyStartTime = 0;
unsigned long hungerStartTime = 0;
unsigned long lastCleanStepCount = 0;
int neglectCount = 0;

// Care system thresholds
#define POOP_CHECK_INTERVAL 300000     // Check for poop every 5 minutes
#define POOP_CHANCE_OVERFED 70         // 70% chance when overfed
#define POOP_CHANCE_RANDOM 20          // 20% random chance
#define DIRTY_SICK_TIME 600000         // Sick after 10 min dirty
#define HUNGRY_SICK_TIME 1200000       // Sick after 20 min hungry
#define CLEAN_STEPS_NEEDED 200         // 200 steps to auto-clean
#define DISCIPLINE_GOOD_FEED 2         // +2 when feeding properly
#define DISCIPLINE_BAD_SHAKE 5         // -5 for excessive shaking
#define DISCIPLINE_SLEEP_DISTURB 3     // -3 for noise during sleep

Preferences preferences;

// ============ BITMAP CHARACTER ARRAYS ============
// ========== INFANT STAGE - Simple cute baby face ==========
const unsigned char infantHappy[] PROGMEM = {
  0b00000001, 0b10000000,  0b00000111, 0b11100000,  0b00011111, 0b11111000,
  0b00111111, 0b11111100,  0b01111111, 0b11111110,  0b01110011, 0b11001110,
  0b11110011, 0b11001111,  0b11111111, 0b11111111,  0b11111111, 0b11111111,
  0b11110110, 0b01101111,  0b11111111, 0b11111111,  0b01110000, 0b00001110,
  0b00111111, 0b11111100,  0b00011111, 0b11111000,  0b00001111, 0b11110000,
  0b00000110, 0b01100000
};

const unsigned char infantBlink[] PROGMEM = {
  0b00000001, 0b10000000,  0b00000111, 0b11100000,  0b00011111, 0b11111000,
  0b00111111, 0b11111100,  0b01111111, 0b11111110,  0b01111111, 0b11111110,
  0b11111111, 0b11111111,  0b11111111, 0b11111111,  0b11111111, 0b11111111,
  0b11110110, 0b01101111,  0b11111111, 0b11111111,  0b01110000, 0b00001110,
  0b00111111, 0b11111100,  0b00011111, 0b11111000,  0b00001111, 0b11110000,
  0b00000110, 0b01100000
};

// ========== CHILD STAGE - Two hair bumps, playful look ==========
const unsigned char childHappy[] PROGMEM = {
  0b00010000, 0b00001000,  0b00011000, 0b00011000,  0b00011111, 0b11111000,
  0b00111111, 0b11111100,  0b01111111, 0b11111110,  0b01110110, 0b01101110,
  0b11110110, 0b01101111,  0b11111111, 0b11111111,  0b11111110, 0b01111111,
  0b11111111, 0b11111111,  0b01111000, 0b00011110,  0b00111111, 0b11111100,
  0b00011111, 0b11111000,  0b00001111, 0b11110000,  0b00000110, 0b01100000,
  0b11000000, 0b00000011
};

const unsigned char childBlink[] PROGMEM = {
  0b00010000, 0b00001000,  0b00011000, 0b00011000,  0b00011111, 0b11111000,
  0b00111111, 0b11111100,  0b01111111, 0b11111110,  0b01111111, 0b11111110,
  0b11111111, 0b11111111,  0b11111111, 0b11111111,  0b11111110, 0b01111111,
  0b11111111, 0b11111111,  0b01111000, 0b00011110,  0b00111111, 0b11111100,
  0b00011111, 0b11111000,  0b00001111, 0b11110000,  0b00000110, 0b01100000,
  0b11000000, 0b00000011
};

// ========== TEEN STAGE - Spiky hair, BIG headphones ==========
const unsigned char teenHappy[] PROGMEM = {
  0b00010100, 0b00101000,  0b00011100, 0b00111000,  0b11011111, 0b11111011,
  0b11101111, 0b11110111,  0b11111111, 0b11111111,  0b11100110, 0b01100111,
  0b11110110, 0b01101111,  0b11111111, 0b11111111,  0b11111100, 0b00111111,
  0b01111111, 0b11111110,  0b01111000, 0b00011110,  0b00111111, 0b11111100,
  0b00011111, 0b11111000,  0b00001111, 0b11110000,  0b00000110, 0b01100000,
  0b01110000, 0b00001110
};

const unsigned char teenBlink[] PROGMEM = {
  0b00010100, 0b00101000,  0b00011100, 0b00111000,  0b11011111, 0b11111011,
  0b11101111, 0b11110111,  0b11111111, 0b11111111,  0b11111111, 0b11111111,
  0b11111111, 0b11111111,  0b11111111, 0b11111111,  0b11111100, 0b00111111,
  0b01111111, 0b11111110,  0b01111000, 0b00011110,  0b00111111, 0b11111100,
  0b00011111, 0b11111000,  0b00001111, 0b11110000,  0b00000110, 0b01100000,
  0b01110000, 0b00001110
};

// ========== ADULT STAGE - Clear glasses and tie ==========
const unsigned char adultHappy[] PROGMEM = {
  0b00011111, 0b11111000,  0b00111111, 0b11111100,  0b01111111, 0b11111110,
  0b01100000, 0b00000110,  0b11100110, 0b01100111,  0b11100110, 0b01100111,
  0b11100000, 0b00000111,  0b11111100, 0b00111111,  0b11111111, 0b11111111,
  0b01111000, 0b00011110,  0b01100110, 0b01100110,  0b00111111, 0b11111100,
  0b00011111, 0b11111000,  0b00001110, 0b01110000,  0b00000110, 0b01100000,
  0b00000011, 0b11000000
};

const unsigned char adultBlink[] PROGMEM = {
  0b00011111, 0b11111000,  0b00111111, 0b11111100,  0b01111111, 0b11111110,
  0b01100000, 0b00000110,  0b11111111, 0b11111111,  0b11111111, 0b11111111,
  0b11100000, 0b00000111,  0b11111100, 0b00111111,  0b11111111, 0b11111111,
  0b01111000, 0b00011110,  0b01100110, 0b01100110,  0b00111111, 0b11111100,
  0b00011111, 0b11111000,  0b00001110, 0b01110000,  0b00000110, 0b01100000,
  0b00000011, 0b11000000
};

// ========== OLD STAGE - BALD top, glasses, THICK cane ==========
const unsigned char oldHappy[] PROGMEM = {
  0b00000000, 0b00000000,  0b00111111, 0b11111100,  0b01111111, 0b11111110,
  0b01110110, 0b01101110,  0b01100000, 0b00000110,  0b11100110, 0b01100111,
  0b11100110, 0b01100111,  0b11100000, 0b00000111,  0b11111111, 0b11111111,
  0b11111110, 0b01111111,  0b01111100, 0b00111110,  0b00111111, 0b11111100,
  0b00011111, 0b11111000,  0b00001111, 0b11110000,  0b00111100, 0b01100000,
  0b00111000, 0b00000000
};

const unsigned char oldBlink[] PROGMEM = {
  0b00000000, 0b00000000,  0b00111111, 0b11111100,  0b01111111, 0b11111110,
  0b01110110, 0b01101110,  0b01100000, 0b00000110,  0b11111111, 0b11111111,
  0b11111111, 0b11111111,  0b11100000, 0b00000111,  0b11111111, 0b11111111,
  0b11111110, 0b01111111,  0b01111100, 0b00111110,  0b00111111, 0b11111100,
  0b00011111, 0b11111000,  0b00001111, 0b11110000,  0b00111100, 0b01100000,
  0b00111000, 0b00000000
};

// ============ FORWARD DECLARATIONS ============
void drawBlinkingAnimation();
void drawHappyAnimation();
void drawSadAnimation();
void drawSleepingAnimation();
void drawStatus();
void drawMenuCharacterAnimation(int centerX, int centerY);
void drawMenuBodyForAge(int centerX, int centerY);
void drawMenuScreen();
void drawMenuStatus();
void drawMenuFood();
void drawMenuClean();
void drawMenuHealth();
void drawMenuDiscipline();
void drawHealthHeart(int cx, int cy, bool filled);
void drawHealthHeartCharacter(int cx, int cy, HealthLevel level);
int analyzeFrame(camera_fb_t * fb);
bool isFrameMostlyBlack(camera_fb_t * fb);
void updateEmotionState(bool personDetected);
void saveState();
void loadState();
void checkPoopGeneration();
void checkAutoClean();
void checkHealthSystem();
void updateDiscipline(int change);
void checkAutoHunger();
void scheduleNextHunger();
void updateDailyStats();

// ============ CENTRALIZED EMOTION SYSTEM ============
// Priority: SLEEPING > SICK > DIRTY > HUNGRY > HAPPY > IDLE
// This prevents state conflicts!
void updateEmotionState(bool personDetected) {
  // Priority 1: Auto-Sleep (HIGHEST - nothing overrides this)
  if (autoSleepMode) {
    currentState = SLEEPING;
    return;
  }
  
  // Priority 2: Sick (blocks happy state)
  if (isSick) {
    currentState = SAD;  // Sad/sick face
    return;
  }
  
  // Priority 3: Dirty (blocks happy state)
  if (isDirty) {
    currentState = SAD;  // Unhappy when dirty
    return;
  }
  
  // Priority 4: Hungry/Crying (URGENT)
  if (isHungry) {
    currentState = SAD;  // Crying face
    return;
  }
  
  // Priority 5: Happy when person detected (only if healthy)
  if (personDetected && !isSick && !isDirty) {
    currentState = HAPPY;
    return;
  }
  
  // Priority 6: Default idle/blinking
  currentState = IDLE;
}

// ============ I2C DIAGNOSTIC FUNCTION ============
void scanI2C() {
  Serial.println("\nScanning I2C bus for devices...");
  int count = 0;
  
  for (uint8_t i = 1; i < 127; i++) {
    Wire.beginTransmission(i);
    uint8_t error = Wire.endTransmission();
    
    if (error == 0) {
      Serial.printf("Found device at address: 0x%02X\n", i);
      count++;
    }
  }
  
  if (count == 0) {
    Serial.println("No I2C devices found! Check your connections.");
  } else {
    Serial.printf("Total devices found: %d\n", count);
  }
}

// ============ EGG ANIMATION HELPER FUNCTIONS ============
static inline int clampi(int v, int lo, int hi) { 
  return (v < lo) ? lo : ((v > hi) ? hi : v); 
}

void drawEllipse(int xc, int yc, int rx, int ry, uint16_t color) {
  long rx2 = (long)rx * rx;
  long ry2 = (long)ry * ry;
  long twoRx2 = 2 * rx2;
  long twoRy2 = 2 * ry2;

  long x = 0;
  long y = ry;
  long px = 0;
  long py = twoRx2 * y;

  long p = (long)(ry2 - rx2 * ry + (rx2 / 4));
  while (px < py) {
    display.drawPixel(xc + x, yc + y, color);
    display.drawPixel(xc - x, yc + y, color);
    display.drawPixel(xc + x, yc - y, color);
    display.drawPixel(xc - x, yc - y, color);

    x++;
    px += twoRy2;
    if (p < 0) {
      p += ry2 + px;
    } else {
      y--;
      py -= twoRx2;
      p += ry2 + px - py;
    }
  }

  p = (long)(ry2 * (x + 0.5f) * (x + 0.5f) + rx2 * (y - 1) * (y - 1) - rx2 * ry2);
  while (y >= 0) {
    display.drawPixel(xc + x, yc + y, color);
    display.drawPixel(xc - x, yc + y, color);
    display.drawPixel(xc + x, yc - y, color);
    display.drawPixel(xc - x, yc - y, color);

    y--;
    py -= twoRx2;
    if (p > 0) {
      p += rx2 - py;
    } else {
      x++;
      px += twoRy2;
      p += rx2 - py + px;
    }
  }
}

void fillEllipse(int xc, int yc, int rx, int ry, uint16_t color) {
  long rx2 = (long)rx * rx;
  long ry2 = (long)ry * ry;
  long twoRx2 = 2 * rx2;
  long twoRy2 = 2 * ry2;

  long x = 0;
  long y = ry;
  long px = 0;
  long py = twoRx2 * y;

  long p = (long)(ry2 - rx2 * ry + (rx2 / 4));
  while (px < py) {
    display.drawFastHLine(xc - x, yc + y, 2 * x + 1, color);
    display.drawFastHLine(xc - x, yc - y, 2 * x + 1, color);

    x++;
    px += twoRy2;
    if (p < 0) {
      p += ry2 + px;
    } else {
      y--;
      py -= twoRx2;
      p += ry2 + px - py;
    }
  }

  p = (long)(ry2 * (x + 0.5f) * (x + 0.5f) + rx2 * (y - 1) * (y - 1) - rx2 * ry2);
  while (y >= 0) {
    display.drawFastHLine(xc - x, yc + y, 2 * x + 1, color);
    display.drawFastHLine(xc - x, yc - y, 2 * x + 1, color);

    y--;
    py -= twoRx2;
    if (p > 0) {
      p += rx2 - py;
    } else {
      x++;
      px += twoRy2;
      p += rx2 - py + px;
    }
  }
}

void drawEmojiWalkFrame(int centerX, int centerY, int frame) {
  int rad = 6;
  display.fillCircle(centerX, centerY - 2, rad, SSD1306_WHITE);

  int eyeY = centerY - 4;
  int eyeDist = 3;
  display.fillCircle(centerX - eyeDist, eyeY, 1, SSD1306_BLACK);
  display.fillCircle(centerX + eyeDist, eyeY, 1, SSD1306_BLACK);

  int mouthX1 = centerX - 2;
  int mouthX2 = centerX + 2;
  int mouthY = centerY + 1;
  display.drawLine(mouthX1, mouthY, mouthX2, mouthY, SSD1306_BLACK);

  int legLen = 3;
  int leftLegX = centerX - 2;
  int rightLegX = centerX + 2;
  int legStartY = centerY + rad - 2;

  if (frame % 2 == 0) {
    display.drawLine(leftLegX, legStartY, leftLegX - 1, legStartY + legLen, SSD1306_WHITE);
    display.drawLine(rightLegX, legStartY, rightLegX + 1, legStartY + legLen, SSD1306_WHITE);
  } else {
    display.drawLine(leftLegX, legStartY, leftLegX + 1, legStartY + legLen, SSD1306_WHITE);
    display.drawLine(rightLegX, legStartY, rightLegX - 1, legStartY + legLen, SSD1306_WHITE);
  }

  int armLen = 2;
  int armY = centerY;
  int leftArmX = centerX - rad + 1;
  int rightArmX = centerX + rad - 1;

  if (frame % 2 == 0) {
    display.drawLine(leftArmX, armY, leftArmX - 2, armY + armLen, SSD1306_WHITE);
    display.drawLine(rightArmX, armY, rightArmX + 2, armY - armLen, SSD1306_WHITE);
  } else {
    display.drawLine(leftArmX, armY, leftArmX - 2, armY - armLen, SSD1306_WHITE);
    display.drawLine(rightArmX, armY, rightArmX + 2, armY + armLen, SSD1306_WHITE);
  }
}

void playEggHatchingAnimation() {
  const int eggCX = 32;
  const int eggCY = 16;
  const int eggRx = 10;
  const int eggRy = 14;

  // Phase 1: Wobble egg (0-1500ms)
  for (int t = 0; t < 1500; t += 100) {
    display.clearDisplay();
    int wobble = sin(t * 0.01) * 2;
    fillEllipse(eggCX + wobble, eggCY, eggRx, eggRy, SSD1306_WHITE);
    drawEllipse(eggCX + wobble, eggCY, eggRx, eggRy, SSD1306_BLACK);
    display.display();
    delay(100);
  }

  // Phase 2: Crack appears (1500-2000ms)
  for (int t = 0; t < 500; t += 100) {
    display.clearDisplay();
    fillEllipse(eggCX, eggCY, eggRx, eggRy, SSD1306_WHITE);
    drawEllipse(eggCX, eggCY, eggRx, eggRy, SSD1306_BLACK);
    
    // Zigzag crack
    display.drawLine(eggCX - 2, eggCY - 8, eggCX, eggCY - 4, SSD1306_BLACK);
    display.drawLine(eggCX, eggCY - 4, eggCX + 2, eggCY, SSD1306_BLACK);
    display.drawLine(eggCX + 2, eggCY, eggCX, eggCY + 4, SSD1306_BLACK);
    display.display();
    delay(100);
  }

  // Phase 3: Egg splits and emoji rises (2000-4000ms)
  for (int rise = 0; rise <= 20; rise++) {
    display.clearDisplay();
    
    int splitDist = clampi(rise / 2, 0, 8);
    int leftX = eggCX - splitDist;
    int rightX = eggCX + splitDist;

    // Left half
    for (int dy = 0; dy <= eggRy; dy++) {
      int x = (int)(eggRx * sqrt(1.0 - (dy * dy) / (float)(eggRy * eggRy)));
      display.drawLine(leftX - x, eggCY - dy, leftX, eggCY - dy, SSD1306_WHITE);
      display.drawLine(leftX - x, eggCY + dy, leftX, eggCY + dy, SSD1306_WHITE);
    }

    // Right half
    for (int dy = 0; dy <= eggRy; dy++) {
      int x = (int)(eggRx * sqrt(1.0 - (dy * dy) / (float)(eggRy * eggRy)));
      display.drawLine(rightX, eggCY - dy, rightX + x, eggCY - dy, SSD1306_WHITE);
      display.drawLine(rightX, eggCY + dy, rightX + x, eggCY + dy, SSD1306_WHITE);
    }

    // Emoji rising with easing
    float ease = rise / 20.0;
    ease = ease * ease; // ease-out
    int emojiY = eggCY - (int)(ease * 16);
    if (emojiY >= 8) {
      int emojiRad = 6;
      display.fillCircle(eggCX, emojiY, emojiRad, SSD1306_WHITE);
      
      int eyeY = emojiY - 2;
      display.fillCircle(eggCX - 3, eyeY, 1, SSD1306_BLACK);
      display.fillCircle(eggCX + 3, eyeY, 1, SSD1306_BLACK);
      
      int mouthY = emojiY + 2;
      display.drawLine(eggCX - 2, mouthY, eggCX + 2, mouthY, SSD1306_BLACK);
    }

    display.display();
    delay(100);
  }

  delay(500);

  // Phase 4: Arrow appears gradually (4500-5500ms)
  for (int arrowGrow = 0; arrowGrow <= 10; arrowGrow++) {
    display.clearDisplay();
    
    // Emoji at final position
    int finalY = 8;
    int emojiRad = 6;
    display.fillCircle(eggCX, finalY, emojiRad, SSD1306_WHITE);
    
    int eyeY = finalY - 2;
    display.fillCircle(eggCX - 3, eyeY, 1, SSD1306_BLACK);
    display.fillCircle(eggCX + 3, eyeY, 1, SSD1306_BLACK);
    
    int mouthY = finalY + 2;
    display.drawLine(eggCX - 2, mouthY, eggCX + 2, mouthY, SSD1306_BLACK);

    // Arrow shaft
    int arrowX = 50;
    int arrowY = 16;
    int arrowLen = clampi(arrowGrow, 0, 8);
    display.drawLine(arrowX, arrowY, arrowX + arrowLen, arrowY, SSD1306_WHITE);
    
    if (arrowGrow > 4) {
      display.drawLine(arrowX + arrowLen - 2, arrowY - 2, arrowX + arrowLen, arrowY, SSD1306_WHITE);
      display.drawLine(arrowX + arrowLen - 2, arrowY + 2, arrowX + arrowLen, arrowY, SSD1306_WHITE);
    }

    display.display();
    delay(100);
  }

  delay(500);

  // Phase 5: Screen slides right (6000-7000ms)
  for (int slide = 0; slide <= 64; slide += 4) {
    display.clearDisplay();
    
    int emojiX = eggCX - slide;
    int finalY = 8;
    
    if (emojiX > -10) {
      int emojiRad = 6;
      display.fillCircle(emojiX, finalY, emojiRad, SSD1306_WHITE);
      
      int eyeY = finalY - 2;
      display.fillCircle(emojiX - 3, eyeY, 1, SSD1306_BLACK);
      display.fillCircle(emojiX + 3, eyeY, 1, SSD1306_BLACK);
      
      int mouthY = finalY + 2;
      display.drawLine(emojiX - 2, mouthY, emojiX + 2, mouthY, SSD1306_BLACK);
    }

    int arrowX = 50 - slide;
    int arrowY = 16;
    if (arrowX > -10 && arrowX < 64) {
      display.drawLine(arrowX, arrowY, arrowX + 8, arrowY, SSD1306_WHITE);
      display.drawLine(arrowX + 6, arrowY - 2, arrowX + 8, arrowY, SSD1306_WHITE);
      display.drawLine(arrowX + 6, arrowY + 2, arrowX + 8, arrowY, SSD1306_WHITE);
    }

    display.display();
    delay(60);
  }

  // Phase 6: Emoji walks across screen (7000-10000ms)
  for (int walkX = -10; walkX <= 70; walkX += 2) {
    display.clearDisplay();
    
    int frame = walkX / 4;
    drawEmojiWalkFrame(walkX, 20, frame);
    
    display.display();
    delay(80);
  }

  delay(500);
}

void setup() {
  Serial.begin(115200);
  delay(100);
  
  // ============ INITIALIZE I2C ============
  Wire.begin(SDA_PIN, SCL_PIN);
  delay(100);
  
  Serial.println("\n\n=== KAKU STARTUP ===");
  Serial.printf("SDA_PIN: %d, SCL_PIN: %d\n", SDA_PIN, SCL_PIN);
  Serial.printf("OLED Address: 0x%02X\n", SCREEN_ADDRESS);
  
  // ============ INITIALIZE MPU6050 ============
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 not detected");
  }
  
  // ============ INITIALIZE OLED ============
  Serial.println("Initializing OLED...");
  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println("ERROR: OLED not found at 0x3C!");
    Serial.println("Trying to scan I2C bus for devices...");
    scanI2C();
    while(1);  // Hang if OLED not found
  }
  
  Serial.println("OLED initialized successfully!");
  
  // ============ PLAY EGG HATCHING ANIMATION ============
  playEggHatchingAnimation();
  
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 8);
  display.println("KAKU");
  display.setCursor(0, 18);
  display.println("Starting...");
  display.display();
  delay(1000);
  
  // ============ INITIALIZE CAMERA ============
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
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 2;
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    display.clearDisplay();
    display.setCursor(0, 8);
    display.println("Camera Failed");
    display.display();
    while(1);
  }
  
  cameraReady = true;
  
  // ============ LOAD SAVED STATE ============
  // loadState();  // COMMENTED FOR TESTING
  
  display.clearDisplay();
  display.setCursor(0, 8);
  display.println("Ready!");
  display.display();
  delay(1000);
  
  lastCheckTime = millis();
  lastAnimationTime = millis();
  lastFeedTime = millis();  // Initialize feed timer
  lastPoopCheckTime = millis(); // Initialize poop check
}

void loop() {
  unsigned long currentTime = millis();
  
  // ============ READ MPU6050 ONCE ============
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);
  
  float ax_g = ax / 16384.0;
  float ay_g = ay / 16384.0;
  float az_g = az / 16384.0;
  
  // ============ STEP COUNTER - DUAL SYSTEM ============
  // Global step counter (for age/growth) - counts everywhere
  // Feed step counter - counts only in MENU when hungry
  
  float accMag = sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g);
  filteredAcc = 0.8 * filteredAcc + 0.2 * accMag;
  
  // Detect steps
  if (filteredAcc > STEP_THRESHOLD &&
      wasBelow &&
      (currentTime - lastStepTime) > STEP_DELAY) {
    
    // Increment global step counter (for walking tracking)
    stepCount++;
    lastWalkTime = currentTime;  // Update last walk time
    
    // Reset fatty status when walking resumes
    if (isFatty) {
      isFatty = false;
      healthScore += 5;  // +5 health for resuming walking
      if (healthScore > 10000) healthScore = 10000;
      Serial.println(">>> WALKING RESUMED! Health +5!");
    }
    
    // Health increases by +5 for every 1000 steps
    static int lastHealthStepCount = 0;
    if (stepCount - lastHealthStepCount >= 1000) {
      healthScore += 5;
      if (healthScore > 10000) healthScore = 10000;
      lastHealthStepCount = stepCount;
      Serial.printf(">>> STEPS: 1000! Health +5! Total: %d\n", healthScore);
    }
    
    // Walking feeds the baby on MAIN screen when hungry (via camera frames) *UPDATED*
    if (currentScreen == MAIN_SCREEN && isHungry && feedLevel < maxFeedLevel) {
      // Walking triggers feeding detection via camera frames
      // No more step counter for feeding - camera AI detects feeding
      Serial.printf(">>> STEP! Global: %d (feeding detection via camera)\n", stepCount);
      
      // Feeding is now camera-based, not step-based
      // Camera will capture 10 frames and AI decides if feeding occurred
    } else {
      Serial.printf(">>> STEP! Global: %d\n", stepCount);
    }
    
    lastStepTime = currentTime;
    wasBelow = false;
  }
  
  if (filteredAcc < 1.05) {
    wasBelow = true;
  }
  
  // ============ CAMERA FEEDING - 10 FRAMES CAPTURE (REMOVED SHAKE MECHANIC) ============ *UPDATED*
  // Shake detection removed - camera now continuously captures frames for feeding
  // System will capture 10 new frames per feeding cycle for AI analysis
  // No more shake-based triggering
  
  // Update last acceleration values for next loop
  lastAx = ax_g;
  lastAy = ay_g;
  
  // ============ HUNGER DEGRADATION (INCREASE MAX OVER TIME) ============
  if (!isHungry && feedLevel >= maxFeedLevel && lastFeedTime > 0) {
    // Check if 1 hour passed since last feed
    if (currentTime - lastFeedTime >= HUNGER_INCREASE_INTERVAL) {
      if (maxFeedLevel < ABSOLUTE_MAX_FEED_LEVEL) {
        maxFeedLevel++;  // Increase difficulty
        feedLevel = maxFeedLevel - 1;  // Bar drops by 1 level
        isHungry = true;  // Baby gets hungry
        lastFeedTime = currentTime;  // Reset timer
      }
    }
  }
  
  // ============ CHECK CAMERA DETECTION ============
  if (cameraReady && currentTime - lastCheckTime >= CHECK_INTERVAL) {
    lastCheckTime = currentTime;
    
    camera_fb_t * fb = esp_camera_fb_get();
    if (fb) {
      int activityLevel = analyzeFrame(fb);
      
      // Check for black frame (lens covered) for menu actions
      bool isBlack = isFrameMostlyBlack(fb);
      
      if (isBlack) {
        if (!blackActive) {
          blackActive = true;
          blackStartTime = currentTime;
        }
        // Just keep holding, don't trigger anything yet
      } else {
        // Camera uncovered - check how long it was held
        if (blackActive) {
          unsigned long holdDuration = currentTime - blackStartTime;
          
          if (holdDuration >= HOLD_TIME_MS && 
              (currentTime - lastMenuChangeTime) >= COOLDOWN_MS) {
            // Held for 3+ seconds = Menu toggle (open/close menu)
            currentScreen = (currentScreen == MAIN_SCREEN) ? MENU_SCREEN : MAIN_SCREEN;
            lastMenuChangeTime = currentTime;
            Serial.printf(">>> MENU TOGGLE (%.1fs hold)\n", holdDuration / 1000.0);
          }
          else if (holdDuration >= QUICK_COVER_TIME_MS && holdDuration < HOLD_TIME_MS &&
                   currentScreen == MENU_SCREEN &&
                   (currentTime - lastMenuPageChangeTime) >= 800) {
            // Quick cover (0.5-3 sec) in menu = change page
            currentMenuPage = (MenuPage)((currentMenuPage + 1) % MENU_TOTAL);
            lastMenuPageChangeTime = currentTime;
            Serial.printf(">>> MENU PAGE CHANGE (%.1fs cover)\n", holdDuration / 1000.0);
          }
          
          blackActive = false;
        }
      }
      
      esp_camera_fb_return(fb);
      
      // Detect person presence
      personDetected = (activityLevel > NEAR_THRESHOLD);
      detectionLevel = personDetected ? 2 : 0;
      
      // Update emotion using centralized logic
      updateEmotionState(personDetected);
    }
  }
  
  // ============ AUTO-SLEEP SYSTEM ============
  // Check if sleep should start (5 minutes after feeding)
  if (sleepScheduled && !autoSleepMode) {
    if (currentTime - sleepScheduleTime >= SLEEP_DELAY_MS) {
      autoSleepMode = true;
      sleepStartTime = currentTime;
      sleepScheduled = false;
      Serial.println(">>> AUTO-SLEEP: Baby sleeping for 10 minutes...");
      // Note: updateEmotionState() will set SLEEPING based on autoSleepMode
    }
  }
  
  // Check if sleep should end (after 10 minutes)
  if (autoSleepMode) {
    if (currentTime - sleepStartTime >= SLEEP_DURATION_MS) {
      autoSleepMode = false;
      
      // ON WAKE-UP: Age increases by 1
      characterAge++;
      
      // ON WAKE-UP: Health stays SAME (only -5 if disturbed) *UPDATED*
      // Health was decreased during sleep disturbance, not on natural wake-up
      
      // Check if character is fatty (no walking for 3 days) *UPDATED*
      if (lastWalkTime > 0) {
        unsigned long timeSinceWalk = currentTime - lastWalkTime;
        if (timeSinceWalk > NO_WALK_THRESHOLD) {
          isFatty = true;
          Serial.println(">>> TOO LAZY! Character is getting FATTY!");
        }
      }
      
      // Update daily stats for health animation *UPDATED*
      updateDailyStats();
      
      Serial.printf(">>> WAKE UP! Age: %d | Health: %d\n", characterAge, healthScore);
    }
  }
  
  // ============ CARE SYSTEM CHECKS ============
  checkPoopGeneration();
  checkAutoClean();
  checkHealthSystem();
  checkAutoHunger();
  
  // ============ UPDATE ANIMATION FRAME ============
  if (currentTime - lastAnimationTime >= ANIMATION_DELAY) {
    lastAnimationTime = currentTime;
    currentFrame++;
  }
  
  // ============ MENU PAGE NAVIGATION ============
  // Menu pages change via quick camera cover (0.5-3 sec) in menu screen
  // Long hold (3+ sec) toggles menu open/close
  
  // ============ DRAW CHARACTER ============
  if (currentScreen == MENU_SCREEN) {
    drawMenuScreen();
  } else {
    display.clearDisplay();
    switch (currentState) {
      case IDLE:
      case BLINKING:
        drawBlinkingAnimation();
        break;
      case HAPPY:
        drawHappyAnimation();
        break;
      case SAD:
        drawSadAnimation();
        break;
      case SLEEPING:
        drawSleepingAnimation();
        break;
    }
    
    drawStatus();
  }
  
  display.display();
}

// ============ CAMERA FUNCTIONS ============
int analyzeFrame(camera_fb_t * fb) {
  if (fb == NULL || fb->len == 0) return 0;
  
  long totalBrightness = 0;
  int sampleCount = 0;
  int maxBright = 0;
  int minBright = 255;
  
  for (int i = 0; i < fb->len && i < 12000; i += 8) {
    uint8_t pixel = fb->buf[i];
    totalBrightness += pixel;
    sampleCount++;
    
    if (pixel > maxBright) maxBright = pixel;
    if (pixel < minBright) minBright = pixel;
  }
  
  if (sampleCount == 0) return 0;
  
  int avgBrightness = totalBrightness / sampleCount;
  int contrast = maxBright - minBright;
  int activityScore = (contrast * avgBrightness) / 8;
  
  return activityScore;
}

bool isFrameMostlyBlack(camera_fb_t * fb) {
  if (fb == NULL || fb->len == 0) return false;
  
  long total = 0;
  int maxB = 0;
  int minB = 255;
  int count = 0;
  
  for (int i = 0; i < fb->len && i < 12000; i += 8) {
    uint8_t p = fb->buf[i];
    total += p;
    count++;
    if (p > maxB) maxB = p;
    if (p < minB) minB = p;
  }
  
  if (count == 0) return false;
  
  int avgBrightness = total / count;
  int contrast = maxB - minB;
  
  return (avgBrightness <= BLACK_BRIGHTNESS_TH && contrast <= BLACK_CONTRAST_TH);
}

// ============ BITMAP CHARACTER DRAWING HELPER ============
void drawCharacter(int x, int y, const unsigned char* bitmap) {
  display.drawBitmap(x, y, bitmap, 16, 16, SSD1306_WHITE);
}

// ============ GET BITMAP FOR CURRENT STAGE ============
const unsigned char* getBitmapForStage(bool blinking) {
  switch (lifeStage) {
    case INFANT:
      return blinking ? infantBlink : infantHappy;
    case CHILD:
      return blinking ? childBlink : childHappy;
    case TEEN:
      return blinking ? teenBlink : teenHappy;
    case ADULT:
      return blinking ? adultBlink : adultHappy;
    case OLD:
      return blinking ? oldBlink : oldHappy;
    default:
      return infantHappy;
  }
}

// ============ BLINKING ANIMATION ============
void drawBlinkingAnimation() {
  int centerX = 30;
  int centerY = 8;
  
  int blinkFrame = currentFrame % 4;
  bool isBlinking = (blinkFrame >= 2);
  
  const unsigned char* bitmap = getBitmapForStage(isBlinking);
  drawCharacter(centerX, centerY, bitmap);
  
  // Show fatty indicator
  if (isFatty) {
    display.drawCircle(centerX - 8, centerY + 5, 3, SSD1306_WHITE);
    display.drawCircle(centerX + 8, centerY + 5, 3, SSD1306_WHITE);
  }
  
  // Show crying indicator when hungry
  if (isHungry) {
    // Tears
    display.drawPixel(centerX + 4, centerY + 12, SSD1306_WHITE);
    display.drawPixel(centerX + 12, centerY + 12, SSD1306_WHITE);
    display.drawPixel(centerX + 4, centerY + 13, SSD1306_WHITE);
    display.drawPixel(centerX + 12, centerY + 13, SSD1306_WHITE);
  }
}

// ============ HAPPY ANIMATION ============
void drawHappyAnimation() {
  int centerX = 30;
  int centerY = 8;
  
  const unsigned char* bitmap = getBitmapForStage(false);
  drawCharacter(centerX, centerY, bitmap);
  
  // Add smile visual indicator when happy
  display.drawLine(centerX - 3, centerY + 5, centerX + 3, centerY + 5, SSD1306_WHITE);
  display.drawPixel(centerX - 2, centerY + 6, SSD1306_WHITE);
  display.drawPixel(centerX + 2, centerY + 6, SSD1306_WHITE);
  
  // Show fatty indicator
  if (isFatty) {
    display.drawCircle(centerX - 8, centerY + 5, 3, SSD1306_WHITE);
    display.drawCircle(centerX + 8, centerY + 5, 3, SSD1306_WHITE);
  }
}

// ============ SAD ANIMATION ============
void drawSadAnimation() {
  int centerX = 30;
  int centerY = 8;
  
  const unsigned char* bitmap = getBitmapForStage(false);
  drawCharacter(centerX, centerY, bitmap);
  
  // Show fatty indicator
  if (isFatty) {
    display.drawCircle(centerX - 8, centerY + 5, 3, SSD1306_WHITE);
    display.drawCircle(centerX + 8, centerY + 5, 3, SSD1306_WHITE);
  }
  
  // Draw sad tears
  display.drawPixel(centerX + 4, centerY + 12, SSD1306_WHITE);
  display.drawPixel(centerX + 12, centerY + 12, SSD1306_WHITE);
  display.drawPixel(centerX + 4, centerY + 13, SSD1306_WHITE);
  display.drawPixel(centerX + 12, centerY + 13, SSD1306_WHITE);
}

// ============ SLEEPING ANIMATION ============
void drawSleepingAnimation() {
  int centerX = 30;
  int centerY = 8;
  
  const unsigned char* bitmap = getBitmapForStage(true);  // Use blink version (closed eyes)
  drawCharacter(centerX, centerY, bitmap);
  
  // Draw Z's for sleep
  int sleepFrame = currentFrame % 3;
  int offsetX = 20;
  int offsetY = 2;
  
  display.drawLine(offsetX, offsetY, offsetX + 3, offsetY, SSD1306_WHITE);
  display.drawLine(offsetX + 3, offsetY, offsetX, offsetY + 3, SSD1306_WHITE);
  display.drawLine(offsetX, offsetY + 3, offsetX + 3, offsetY + 3, SSD1306_WHITE);
  
  if (sleepFrame == 0) {
    display.drawLine(offsetX + 6, offsetY - 4, offsetX + 9, offsetY - 4, SSD1306_WHITE);
    display.drawLine(offsetX + 9, offsetY - 4, offsetX + 6, offsetY - 1, SSD1306_WHITE);
    display.drawLine(offsetX + 6, offsetY - 1, offsetX + 9, offsetY - 1, SSD1306_WHITE);
  } else if (sleepFrame == 1) {
    display.drawLine(offsetX + 4, offsetY - 4, offsetX + 7, offsetY - 4, SSD1306_WHITE);
    display.drawLine(offsetX + 7, offsetY - 4, offsetX + 4, offsetY - 1, SSD1306_WHITE);
    display.drawLine(offsetX + 4, offsetY - 1, offsetX + 7, offsetY - 1, SSD1306_WHITE);
  }
}

// ============ STATUS DISPLAY ============
void drawStatus() {
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  
  int barX = 1;
  int barY = 2;
  int barWidth = 3;
  int barHeight = 28;
  
  display.drawRect(barX, barY, barWidth, barHeight, SSD1306_WHITE);
  display.fillRect(barX + 1, barY + barHeight - 1 - 7, barWidth - 2, 7, SSD1306_WHITE);
  
  display.setCursor(6, 0);
  switch (lifeStage) {
    case INFANT: display.print("I"); break;
    case CHILD: display.print("C"); break;
    case TEEN: display.print("T"); break;
    case ADULT: display.print("A"); break;
    case OLD: display.print("O"); break;
  }
  
  display.setCursor(6, 25);
  if (isDirty) {
    // Blinking CLEAN! text when dirty
    if ((millis() / 400) % 2 == 0) {
      display.print("CLEAN");
    }
  } else if (isSick) {
    // Blinking SICK! text when sick
    if ((millis() / 400) % 2 == 0) {
      display.print("SICK!");
    }
  } else if (isHungry) {
    // Blinking FEED text when hungry
    if ((millis() / 400) % 2 == 0) {
      display.print("FEED!");
    }
  } else if (currentState == HAPPY) {
    display.print("Happy!");
  } else {
    display.print("Idle");
  }
  
  // ============ HUNGER TIMER HEART (TOP RIGHT) ============
  // Draw heart indicator showing time until next hunger
  if (!isHungry && autoHungerInitialized) {
    unsigned long currentTime = millis();
    unsigned long timeRemaining = (nextHungerTime > currentTime) ? (nextHungerTime - currentTime) : 0;
    
    // Calculate hunger interval for current life stage
    unsigned long hungerInterval;
    switch (lifeStage) {
      case INFANT: hungerInterval = (INFANT_HUNGER_MIN + INFANT_HUNGER_MAX) / 2; break;
      case CHILD: hungerInterval = (CHILD_HUNGER_MIN + CHILD_HUNGER_MAX) / 2; break;
      case TEEN: hungerInterval = (TEEN_HUNGER_MIN + TEEN_HUNGER_MAX) / 2; break;
      case ADULT: hungerInterval = (ADULT_HUNGER_MIN + ADULT_HUNGER_MAX) / 2; break;
      case OLD: hungerInterval = (OLD_HUNGER_MIN + OLD_HUNGER_MAX) / 2; break;
      default: hungerInterval = INFANT_HUNGER_MIN; break;
    }
    
    // Calculate fill percentage (0-100)
    int fillPercent = (timeRemaining * 100) / hungerInterval;
    fillPercent = constrain(fillPercent, 0, 100);
    
    // Heart position (top right corner)
    int heartX = 55;
    int heartY = 2;
    int heartSize = 6;
    
    // Draw heart outline
    display.drawPixel(heartX + 1, heartY, SSD1306_WHITE);
    display.drawPixel(heartX + 2, heartY, SSD1306_WHITE);
    display.drawPixel(heartX + 4, heartY, SSD1306_WHITE);
    display.drawPixel(heartX + 5, heartY, SSD1306_WHITE);
    
    display.drawPixel(heartX, heartY + 1, SSD1306_WHITE);
    display.drawPixel(heartX + 6, heartY + 1, SSD1306_WHITE);
    
    display.drawPixel(heartX, heartY + 2, SSD1306_WHITE);
    display.drawPixel(heartX + 6, heartY + 2, SSD1306_WHITE);
    
    display.drawPixel(heartX + 1, heartY + 3, SSD1306_WHITE);
    display.drawPixel(heartX + 5, heartY + 3, SSD1306_WHITE);
    
    display.drawPixel(heartX + 2, heartY + 4, SSD1306_WHITE);
    display.drawPixel(heartX + 4, heartY + 4, SSD1306_WHITE);
    
    display.drawPixel(heartX + 3, heartY + 5, SSD1306_WHITE);
    
    // Fill heart based on time remaining
    int fillLines = (fillPercent * 4) / 100;  // 0-4 fill levels
    
    if (fillLines >= 4) {
      display.drawPixel(heartX + 1, heartY + 1, SSD1306_WHITE);
      display.drawPixel(heartX + 2, heartY + 1, SSD1306_WHITE);
      display.drawPixel(heartX + 3, heartY + 1, SSD1306_WHITE);
      display.drawPixel(heartX + 4, heartY + 1, SSD1306_WHITE);
      display.drawPixel(heartX + 5, heartY + 1, SSD1306_WHITE);
    }
    if (fillLines >= 3) {
      display.drawPixel(heartX + 1, heartY + 2, SSD1306_WHITE);
      display.drawPixel(heartX + 2, heartY + 2, SSD1306_WHITE);
      display.drawPixel(heartX + 3, heartY + 2, SSD1306_WHITE);
      display.drawPixel(heartX + 4, heartY + 2, SSD1306_WHITE);
      display.drawPixel(heartX + 5, heartY + 2, SSD1306_WHITE);
    }
    if (fillLines >= 2) {
      display.drawPixel(heartX + 2, heartY + 3, SSD1306_WHITE);
      display.drawPixel(heartX + 3, heartY + 3, SSD1306_WHITE);
      display.drawPixel(heartX + 4, heartY + 3, SSD1306_WHITE);
    }
    if (fillLines >= 1) {
      display.drawPixel(heartX + 3, heartY + 4, SSD1306_WHITE);
    }
  }
}



// ============ FEEDING MENU HELPERS ============
void drawFoodBar() {
  int barHeight = 26;
  int barX = 59;  // Right edge with margin
  int barY = 3;   // Top margin
  
  display.drawRect(barX, barY, 4, barHeight, SSD1306_WHITE);
  
  int filledHeight = (barHeight * feedLevel) / maxFeedLevel;
  if (filledHeight > 0) {
    display.fillRect(barX + 1,
                     barY + barHeight - filledHeight,
                     2,
                     filledHeight,
                     SSD1306_WHITE);
  }
}

void drawInfantFeed(bool crying) {
  int cx = 32;  // Centered horizontally (64/2)
  int cy = 16;  // Centered vertically (32/2)
  
  // Face
  display.fillCircle(cx, cy, 7, SSD1306_WHITE);
  display.drawCircle(cx, cy, 7, SSD1306_BLACK);
  
  if (crying) {
    // Sad slanted eyes
    display.drawLine(cx - 5, cy - 3, cx - 2, cy - 2, SSD1306_BLACK);
    display.drawLine(cx + 2, cy - 2, cx + 5, cy - 3, SSD1306_BLACK);
    
    // Tears
    display.drawLine(cx - 3, cy, cx - 3, cy + 2, SSD1306_BLACK);
    display.drawLine(cx + 3, cy, cx + 3, cy + 2, SSD1306_BLACK);
    
    // Crying mouth (down curve)
    display.drawLine(cx - 3, cy + 5, cx + 3, cy + 5, SSD1306_BLACK);
    display.drawPixel(cx - 3, cy + 4, SSD1306_BLACK);
    display.drawPixel(cx + 3, cy + 4, SSD1306_BLACK);
  } else {
    // Normal eyes
    display.fillCircle(cx - 3, cy - 2, 1, SSD1306_BLACK);
    display.fillCircle(cx + 3, cy - 2, 1, SSD1306_BLACK);
    
    // Smile
    display.drawLine(cx - 2, cy + 3, cx + 2, cy + 3, SSD1306_BLACK);
  }
  
  // Baby hair
  display.drawLine(cx, cy - 9, cx, cy - 7, SSD1306_WHITE);
}

void drawBottle(int level) {
  int x = 46;
  int y = 12;
  int h = 8;
  
  // Bottle outline
  display.drawRect(x, y, 4, h, SSD1306_WHITE);
  display.drawRect(x + 1, y - 2, 2, 2, SSD1306_WHITE); // nipple
  
  // Milk level (animated)
  int fillHeight = map(level, 5, 0, h - 1, 0);
  
  if (fillHeight > 0) {
    display.fillRect(
      x + 1,
      y + h - fillHeight,
      2,
      fillHeight,
      SSD1306_WHITE
    );
  }
}



// ============ MENU ANIMATED CHARACTER ============
// Draws character with emotions-based animation in menu screens
void drawMenuCharacterAnimation(int centerX, int centerY) {
  // Use currentFrame for smooth animation
  int animFrame = currentFrame % 4;
  int blinkFrame = (millis() / 300) % 4;  // Blink timing
  
  // Draw character body (simple pixel art representation)
  // Head circle
  display.drawCircle(centerX, centerY - 2, 6, SSD1306_WHITE);
  
  // Draw based on EMOTION STATE with animation
  
  if (autoSleepMode) {
    // ========= SLEEPING ANIMATION =========
    // Closed eyes
    display.drawLine(centerX - 2, centerY - 4, centerX - 1, centerY - 4, SSD1306_WHITE);
    display.drawLine(centerX + 1, centerY - 4, centerX + 2, centerY - 4, SSD1306_WHITE);
    
    // Mouth (peaceful)
    display.drawLine(centerX - 2, centerY + 1, centerX + 2, centerY + 1, SSD1306_WHITE);
    
    // Z animation (animated floating Z)
    int zOffset = (millis() / 500) % 3;
    int zX = centerX + 8;
    int zY = centerY - 5 + zOffset;
    display.drawPixel(zX, zY, SSD1306_WHITE);
    display.drawPixel(zX + 1, zY, SSD1306_WHITE);
    display.drawPixel(zX, zY + 1, SSD1306_WHITE);
    display.drawPixel(zX + 1, zY + 2, SSD1306_WHITE);
    
  } else if (isSick || isDirty) {
    // ========= SAD/CRY ANIMATION =========
    // Eyes (no blink, sad)
    if (isSick) {
      // X eyes when sick
      display.drawLine(centerX - 3, centerY - 4, centerX - 2, centerY - 3, SSD1306_WHITE);
      display.drawLine(centerX - 2, centerY - 4, centerX - 3, centerY - 3, SSD1306_WHITE);
      display.drawLine(centerX + 2, centerY - 3, centerX + 3, centerY - 4, SSD1306_WHITE);
      display.drawLine(centerX + 3, centerY - 3, centerX + 2, centerY - 4, SSD1306_WHITE);
    } else {
      // Sad eyes
      display.drawPixel(centerX - 2, centerY - 4, SSD1306_WHITE);
      display.drawPixel(centerX + 2, centerY - 4, SSD1306_WHITE);
    }
    
    // Sad mouth (upside down)
    display.drawLine(centerX - 1, centerY + 2, centerX + 1, centerY + 2, SSD1306_WHITE);
    
    // Tears (animated falling)
    int tearOffset = (millis() / 200) % 3;
    if (tearOffset < 2) {
      display.drawPixel(centerX - 3, centerY + 1 + tearOffset, SSD1306_WHITE);
      display.drawPixel(centerX + 3, centerY + 1 + tearOffset, SSD1306_WHITE);
    }
    
  } else if (isHungry) {
    // ========= HUNGRY/SAD ANIMATION =========
    // Open eyes (normal)
    display.drawPixel(centerX - 2, centerY - 4, SSD1306_WHITE);
    display.drawPixel(centerX + 2, centerY - 4, SSD1306_WHITE);
    
    // Sad mouth
    display.drawLine(centerX - 2, centerY + 2, centerX + 2, centerY + 2, SSD1306_WHITE);
    
    // Animated tears
    int tearBlink = (millis() / 300) % 2;
    if (tearBlink == 0) {
      display.drawPixel(centerX - 3, centerY, SSD1306_WHITE);
      display.drawPixel(centerX + 3, centerY, SSD1306_WHITE);
    }
    
  } else if (personDetected && !isSick && !isDirty) {
    // ========= HAPPY ANIMATION =========
    // Blinking animation
    if (blinkFrame >= 2) {
      // Blinking (closed eyes)
      display.drawLine(centerX - 2, centerY - 4, centerX - 1, centerY - 4, SSD1306_WHITE);
      display.drawLine(centerX + 1, centerY - 4, centerX + 2, centerY - 4, SSD1306_WHITE);
    } else {
      // Open eyes with happy shine
      display.drawPixel(centerX - 2, centerY - 4, SSD1306_WHITE);
      display.drawPixel(centerX + 2, centerY - 4, SSD1306_WHITE);
    }
    
    // Happy mouth (smile with curve)
    display.drawLine(centerX - 2, centerY + 1, centerX + 2, centerY + 1, SSD1306_WHITE);
    display.drawPixel(centerX - 1, centerY + 2, SSD1306_WHITE);
    display.drawPixel(centerX + 1, centerY + 2, SSD1306_WHITE);
    
    // Shine/spark (animated)
    if ((millis() / 400) % 2 == 0) {
      display.drawPixel(centerX - 5, centerY - 6, SSD1306_WHITE);
      display.drawPixel(centerX + 5, centerY - 6, SSD1306_WHITE);
    }
    
  } else {
    // ========= IDLE/BLINKING ANIMATION (Default) =========
    // Blinking eyes
    if (blinkFrame >= 2) {
      // Blink (closed eyes)
      display.drawLine(centerX - 2, centerY - 4, centerX - 1, centerY - 4, SSD1306_WHITE);
      display.drawLine(centerX + 1, centerY - 4, centerX + 2, centerY - 4, SSD1306_WHITE);
    } else {
      // Open eyes
      display.drawPixel(centerX - 2, centerY - 4, SSD1306_WHITE);
      display.drawPixel(centerX + 2, centerY - 4, SSD1306_WHITE);
    }
    
    // Neutral mouth
    display.drawLine(centerX - 1, centerY + 1, centerX + 1, centerY + 1, SSD1306_WHITE);
  }
  
  // Draw body/arms based on AGE
  drawMenuBodyForAge(centerX, centerY);
}

// ============ AGE-SPECIFIC BODY ============
// Draw different body/appearance based on character age
void drawMenuBodyForAge(int centerX, int centerY) {
  // Body
  display.drawRect(centerX - 3, centerY + 4, 6, 5, SSD1306_WHITE);
  
  // Age-specific features
  if (characterAge == 0) {
    // ===== INFANT =====
    // Small body, simple
    display.drawRect(centerX - 2, centerY + 5, 4, 3, SSD1306_WHITE);
    
  } else if (characterAge == 1) {
    // ===== CHILD =====
    // Medium body
    display.drawRect(centerX - 3, centerY + 4, 6, 5, SSD1306_WHITE);
    
    // Hair (simple lines on top)
    display.drawPixel(centerX - 1, centerY - 8, SSD1306_WHITE);
    display.drawPixel(centerX, centerY - 8, SSD1306_WHITE);
    display.drawPixel(centerX + 1, centerY - 8, SSD1306_WHITE);
    
  } else if (characterAge == 2) {
    // ===== TEEN =====
    // Taller body
    display.drawRect(centerX - 3, centerY + 3, 6, 6, SSD1306_WHITE);
    
    // Spiky hair
    for (int i = -1; i <= 1; i++) {
      display.drawPixel(centerX + i, centerY - 8, SSD1306_WHITE);
      display.drawPixel(centerX + i, centerY - 9, SSD1306_WHITE);
    }
    
  } else if (characterAge == 3) {
    // ===== ADULT =====
    // Normal body
    display.drawRect(centerX - 3, centerY + 4, 6, 5, SSD1306_WHITE);
    
    // Glasses (animated blink with glasses)
    if ((millis() / 500) % 2 == 0) {
      display.drawCircle(centerX - 2, centerY - 4, 1, SSD1306_WHITE);
      display.drawCircle(centerX + 2, centerY - 4, 1, SSD1306_WHITE);
      display.drawLine(centerX - 1, centerY - 4, centerX + 1, centerY - 4, SSD1306_WHITE);
    }
    
  } else if (characterAge >= 4) {
    // ===== OLD =====
    // Larger body
    display.drawRect(centerX - 4, centerY + 3, 8, 6, SSD1306_WHITE);
    
    // Mustache
    display.drawLine(centerX - 2, centerY, centerX + 2, centerY, SSD1306_WHITE);
    
    // Wrinkles (animated)
    if ((millis() / 600) % 2 == 0) {
      display.drawPixel(centerX - 4, centerY - 2, SSD1306_WHITE);
      display.drawPixel(centerX + 4, centerY - 2, SSD1306_WHITE);
    }
  }
}

// ============ MENU SCREEN ============
void drawMenuScreen() {
  display.clearDisplay();
  
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  
  switch (currentMenuPage) {
    case MENU_STATUS:
      drawMenuStatus();
      break;
    
    case MENU_FOOD:
      drawMenuFood();
      break;
    
    case MENU_CLEAN:
      drawMenuClean();
      break;
    
    case MENU_HEALTH:
      drawMenuHealth();
      break;
    
    case MENU_DISCIPLINE:
      drawMenuDiscipline();
      break;
  }
  
  // Page dots (navigation indicator) - bottom
  for (int i = 0; i < MENU_TOTAL; i++) {
    if (i == currentMenuPage) {
      display.fillCircle(26 + i*3, 30, 1, SSD1306_WHITE);
    } else {
      display.drawPixel(26 + i*3, 30, SSD1306_WHITE);
    }
  }
}

// ============ MENU PAGE: STATUS ============
void drawMenuStatus() {
  display.setCursor(18, 2);
  display.print("STATUS");
  
  // Draw character with animation based on current emotion state & age
  drawMenuCharacterAnimation(24, 14);
  
  // Bottom: Age info
  display.setCursor(22, 26);
  display.print("A:");
  display.print(characterAge);
}

// ============ MENU PAGE: FOOD ============
void drawMenuFood() {
  display.setCursor(20, 2);
  display.print("FOOD");
  
  // Draw character with animation based on current emotion state
  drawMenuCharacterAnimation(12, 14);
  
  // Right: Vertical food bar
  int barX = 52;
  int barY = 6;
  int barWidth = 8;
  int barHeight = 20;
  
  display.drawRect(barX, barY, barWidth, barHeight, SSD1306_WHITE);
  
  // Fill bar based on feed level
  int fillHeight = (feedLevel * (barHeight - 2)) / maxFeedLevel;
  if (fillHeight > 0) {
    display.fillRect(barX + 1, barY + barHeight - 1 - fillHeight, barWidth - 2, fillHeight, SSD1306_WHITE);
  }
  
  // Bottom: Action hint
  display.setCursor(14, 26);
  if (isHungry) {
    display.print("Walk");
  } else {
    display.print("Full");
  }
}

// ============ MENU PAGE: CLEAN ============
void drawMenuClean() {
  display.setCursor(18, 2);
  display.print("CLEAN");
  
  // Draw character with animation - shows emotion based on cleanliness
  drawMenuCharacterAnimation(12, 14);
  
  // Right side: Show cleaning status
  if (isDirty) {
    // Poop icon
    display.fillCircle(48, 14, 3, SSD1306_WHITE);
    display.fillCircle(46, 17, 2, SSD1306_WHITE);
    display.fillCircle(50, 16, 1, SSD1306_WHITE);
    
    // Show cleaning progress (steps toward 200)
    int stepsSinceClean = stepCount - lastCleanStepCount;
    int cleanPercent = (stepsSinceClean * 100) / CLEAN_STEPS_NEEDED;
    cleanPercent = constrain(cleanPercent, 0, 100);
    
    display.setCursor(26, 26);
    display.print(cleanPercent);
    display.print("%");
  } else {
    // Sparkle (clean)
    display.drawPixel(48, 14, SSD1306_WHITE);
    display.drawLine(43, 14, 53, 14, SSD1306_WHITE);
    display.drawLine(48, 9, 48, 19, SSD1306_WHITE);
    
    // Clean status text
    display.setCursor(32, 26);
    display.print("Clean");
  }
}

// ============ MENU PAGE: HEALTH ============
// ============ HEALTH HEART HELPER ============
void drawHealthHeart(int cx, int cy, bool filled) {
  if (filled) {
    display.fillCircle(cx - 3, cy - 2, 3, SSD1306_WHITE);
    display.fillCircle(cx + 3, cy - 2, 3, SSD1306_WHITE);
    display.fillTriangle(cx - 6, cy - 1, cx + 6, cy - 1, cx, cy + 5, SSD1306_WHITE);
  } else {
    display.drawCircle(cx - 3, cy - 2, 3, SSD1306_WHITE);
    display.drawCircle(cx + 3, cy - 2, 3, SSD1306_WHITE);
    display.drawTriangle(cx - 6, cy - 1, cx + 6, cy - 1, cx, cy + 5, SSD1306_WHITE);
  }
}

// ============ HEALTH HEART CHARACTER ============
void drawHealthHeartCharacter(int cx, int cy, HealthLevel level) {
  int pulse = (millis() / 400) % 2;
  bool filled = level >= HEALTH_AVERAGE;

  // Vertical energy
  if (level >= HEALTH_STRONG) cy -= pulse;
  if (level == HEALTH_WEAK)   cy += pulse;

  // HEART BODY
  drawHealthHeart(cx, cy, filled);

  // EYES
  if (level <= HEALTH_TIRED) {
    display.drawLine(cx - 3, cy - 2, cx - 2, cy - 1, SSD1306_BLACK);
    display.drawLine(cx + 2, cy - 1, cx + 3, cy - 2, SSD1306_BLACK);
  } else {
    display.drawPixel(cx - 2, cy - 2, SSD1306_BLACK);
    display.drawPixel(cx + 2, cy - 2, SSD1306_BLACK);
  }

  // MOUTH
  if (level >= HEALTH_STRONG) {
    display.drawLine(cx - 2, cy + 1, cx + 2, cy + 1, SSD1306_BLACK);
  } else if (level == HEALTH_WEAK) {
    display.drawPixel(cx, cy + 2, SSD1306_BLACK);
  }

  // ARMS
  int armY = cy + 1;
  if (level < HEALTH_AVERAGE) {
    display.drawLine(cx - 7, armY, cx - 4, armY + 3, SSD1306_WHITE);
    display.drawLine(cx + 7, armY, cx + 4, armY + 3, SSD1306_WHITE);
  } else if (level < HEALTH_STRONG_PLUS) {
    display.drawLine(cx - 7, armY, cx - 4, armY, SSD1306_WHITE);
    display.drawLine(cx + 7, armY, cx + 4, armY, SSD1306_WHITE);
  } else {
    display.drawLine(cx - 7, armY + 1, cx - 4, armY - 2, SSD1306_WHITE);
    display.drawLine(cx + 7, armY + 1, cx + 4, armY - 2, SSD1306_WHITE);
  }

  // LEGS
  display.drawLine(cx - 2, cy + 6, cx - 2, cy + 9, SSD1306_WHITE);
  display.drawLine(cx + 2, cy + 6, cx + 2, cy + 9, SSD1306_WHITE);

  // MAX POWER AURA
  if (level == HEALTH_MAX) {
    if ((millis() / 200) % 2 == 0) {
      display.drawPixel(cx - 8, cy - 6, SSD1306_WHITE);
      display.drawPixel(cx + 8, cy - 6, SSD1306_WHITE);
      display.drawPixel(cx, cy - 9, SSD1306_WHITE);
    }
  }
}

// ============ MENU PAGE: HEALTH ============
void drawMenuHealth() {
  display.setCursor(16, 2);
  display.print("HEALTH");
  
  // Left: Animated character with emotion based on health
  drawMenuCharacterAnimation(12, 14);
  
  // Right: Health status indicator
  int cx = 48;
  int cy = 14;
  
  // Map health score to strength level
  HealthLevel level;
  if (healthScore >= 9000) level = HEALTH_MAX;
  else if (healthScore >= 7000) level = HEALTH_STRONG_PLUS;
  else if (healthScore >= 5000) level = HEALTH_STRONG;
  else if (healthScore >= 3000) level = HEALTH_AVERAGE;
  else if (healthScore >= 1000) level = HEALTH_TIRED;
  else level = HEALTH_WEAK;
  
  // Draw health hearts (show how healthy)
  for (int i = 0; i < 3; i++) {
    bool filled = (i < (level / 2 + 1));
    drawHealthHeart(cx - 4 + (i * 8), cy, filled);
  }
  
  // Bottom text
  const char* statusText;
  if (healthScore >= 8000) statusText = "MAX";
  else if (healthScore >= 6000) statusText = "GREAT";
  else if (healthScore >= 4000) statusText = "GOOD";
  else if (healthScore >= 2000) statusText = "OK";
  else statusText = "LOW";
  
  display.setCursor(18, 26);
  display.print(statusText);
}

// ============ MENU PAGE: DISCIPLINE ============
void drawMenuDiscipline() {
  display.setCursor(8, 2);
  display.print("CONTROL");
  
  // Left: Animated character - shows emotion based on discipline level
  drawMenuCharacterAnimation(12, 14);
  
  // Right: Discipline indicator
  int barWidth = 10;
  int barHeight = 18;
  int barX = 48;
  int barY = 6;
  
  // Bar outline
  display.drawRect(barX, barY, barWidth, barHeight, SSD1306_WHITE);
  
  // Fill based on discipline level (0-100)
  int fillHeight = (discipline * (barHeight - 2)) / 100;
  if (fillHeight > 0) {
    display.fillRect(barX + 1, barY + barHeight - 1 - fillHeight, barWidth - 2, fillHeight, SSD1306_WHITE);
  }
  
  // Percentage below
  display.setCursor(38, 26);
  display.print(discipline);
  display.print("%");
}

// ============ MENU PAGE: SLEEP ============
void drawMenuSleep() {
  display.setCursor(16, 2);
  display.print("SLEEP");
  
  // Animated sleeping character (forces SLEEPING animation via emotion state)
  // Temporarily use autoSleepMode to force sleeping animation
  bool savedSleep = autoSleepMode;
  autoSleepMode = true;
  drawMenuCharacterAnimation(24, 14);
  autoSleepMode = savedSleep;
  
  // Bottom: Sleep status
  if (autoSleepMode) {
    display.setCursor(10, 26);
    display.print("Sleeping..");
  } else {
    display.setCursor(14, 26);
    display.print("Awake");
  }
}

// ============ NVS PERSISTENCE SYSTEM ============
void saveState() {
  preferences.begin("kaku", false);
  
  // Life stage & age
  preferences.putUChar("lifeStage", (uint8_t)lifeStage);
  preferences.putInt("stepCount", stepCount);
  preferences.putInt("characterAge", characterAge);
  
  // Feeding system
  preferences.putInt("feedLevel", feedLevel);
  preferences.putInt("maxFeedLevel", maxFeedLevel);
  preferences.putBool("isHungry", isHungry);
  
  // Care system
  preferences.putBool("isDirty", isDirty);
  preferences.putBool("isSick", isSick);
  preferences.putUChar("discipline", discipline);
  preferences.putInt("neglectCount", neglectCount);
  preferences.putInt("healthScore", healthScore);
  
  // Timing
  preferences.putULong("lastFeedTime", lastFeedTime);
  preferences.putULong("dirtyStartTime", dirtyStartTime);
  preferences.putULong("hungerStartTime", hungerStartTime);
  
  preferences.end();
  Serial.println(">>> STATE SAVED TO NVS");
}

void loadState() {
  preferences.begin("kaku", true); // Read-only
  
  // Check if data exists
  if (preferences.isKey("lifeStage")) {
    // Life stage & age
    lifeStage = (LifeStage)preferences.getUChar("lifeStage", (uint8_t)INFANT);
    stepCount = preferences.getInt("stepCount", 0);
    characterAge = preferences.getInt("characterAge", 1);
    
    // Feeding system
    feedLevel = preferences.getInt("feedLevel", INITIAL_MAX_FEED_LEVEL);
    maxFeedLevel = preferences.getInt("maxFeedLevel", INITIAL_MAX_FEED_LEVEL);
    isHungry = preferences.getBool("isHungry", false);
    
    // Care system
    isDirty = preferences.getBool("isDirty", false);
    isSick = preferences.getBool("isSick", false);
    discipline = preferences.getUChar("discipline", 50);
    neglectCount = preferences.getInt("neglectCount", 0);
    healthScore = preferences.getInt("healthScore", 10000);
    
    // Timing
    lastFeedTime = preferences.getULong("lastFeedTime", 0);
    dirtyStartTime = preferences.getULong("dirtyStartTime", 0);
    hungerStartTime = preferences.getULong("hungerStartTime", 0);
    
    Serial.println(">>> STATE LOADED FROM NVS");
    Serial.printf("Stage: %d, Steps: %d, Feed: %d/%d, Dirty: %d, Sick: %d, Disc: %d\\n",
                  lifeStage, stepCount, feedLevel, maxFeedLevel, isDirty, isSick, discipline);
  } else {
    Serial.println(">>> NO SAVED STATE - STARTING FRESH");
  }
  
  preferences.end();
}

// ============ POOP/DIRTY GENERATION LOGIC ============
void checkPoopGeneration() {
  unsigned long currentTime = millis();
  
  // Check periodically (every 5 minutes)
  if (currentTime - lastPoopCheckTime < POOP_CHECK_INTERVAL) {
    return;
  }
  lastPoopCheckTime = currentTime;
  
  // Don't generate poop if already dirty or sleeping
  if (isDirty || autoSleepMode) {
    return;
  }
  
  int poopChance = 0;
  
  // Age-based poop frequency
  // CHILD and OLD: 2x chance, Others: 1x chance
  int ageMultiplier = (lifeStage == CHILD || lifeStage == OLD) ? 2 : 1;
  
  // High chance when overfed
  if (feedLevel >= maxFeedLevel) {
    poopChance = POOP_CHANCE_OVERFED * ageMultiplier;
  } else {
    // Random chance otherwise
    poopChance = POOP_CHANCE_RANDOM * ageMultiplier;
  }
  
  // Roll the dice
  if (random(100) < poopChance) {
    isDirty = true;
    dirtyStartTime = currentTime;
    Serial.println(">>> POOP! Baby is dirty now!");
    // saveState();  // COMMENTED FOR TESTING
  }
}

// ============ AUTO-CLEANING SYSTEM ============
void checkAutoClean() {
  if (!isDirty) {
    return;
  }
  
  // Clean after walking enough steps
  int stepsSinceClean = stepCount - lastCleanStepCount;
  if (stepsSinceClean >= CLEAN_STEPS_NEEDED) {
    isDirty = false;
    lastCleanStepCount = stepCount;
    Serial.println(">>> AUTO-CLEAN: Walked away from poop!");
    // saveState();  // COMMENTED FOR TESTING
    return;
  }
  
  // Also clean if in MENU_CLEAN for a while (passive cleaning)
  static unsigned long menuCleanStartTime = 0;
  if (currentScreen == MENU_SCREEN && currentMenuPage == MENU_CLEAN) {
    if (menuCleanStartTime == 0) {
      menuCleanStartTime = millis();
    } else if (millis() - menuCleanStartTime > 5000) { // 5 seconds
      isDirty = false;
      menuCleanStartTime = 0;
      lastCleanStepCount = stepCount;
      Serial.println(">>> AUTO-CLEAN: Cleaned in menu!");
      // saveState();  // COMMENTED FOR TESTING
    }
  } else {
    menuCleanStartTime = 0;
  }
}

// ============ HEALTH/SICKNESS SYSTEM ============
void checkHealthSystem() {
  unsigned long currentTime = millis();
  
  // ============ HEALTH PENALTIES *UPDATED* ============
  // Dirty >1 hour penalty (-5 health) *UPDATED*
  if (isDirty) {
    unsigned long dirtySinceTime = currentTime - dirtyStartTime;
    const unsigned long ONE_HOUR = 3600000;
    
    if (dirtySinceTime > ONE_HOUR) {
      // Check if we haven't already applied this penalty today
      static unsigned long lastDirtyPenaltyTime = 0;
      if (currentTime - lastDirtyPenaltyTime > ONE_HOUR) {
        healthScore -= 5;
        if (healthScore < 0) healthScore = 0;
        lastDirtyPenaltyTime = currentTime;
        Serial.println(">>> HEALTH PENALTY: Dirty too long! Health -5");
      }
    }
  }
  
  // Lazy 1 day penalty (-5 health) *UPDATED*
  // Track if user hasn't walked for a full day
 // Lazy 1 day penalty (-5 health)
// Use millis() only (do NOT mix RTC time and millis time)
const unsigned long ONE_DAY_MS = 86400000UL;

if (!isFatty && lastWalkTime > 0) {
  unsigned long timeSinceWalk = millis() - lastWalkTime;

  if (timeSinceWalk > ONE_DAY_MS) {
    static unsigned long lastLazyPenaltyTime = 0;

    // Apply penalty only once per day
    if (millis() - lastLazyPenaltyTime > ONE_DAY_MS) {
      healthScore -= 5;
      if (healthScore < 0) healthScore = 0;

      lastLazyPenaltyTime = millis();
      Serial.println(">>> HEALTH PENALTY: Lazy for 1 day! Health -5");
    }
  }
}

  
  // Become sick if dirty for too long
  if (isDirty && !isSick) {
    if (currentTime - dirtyStartTime > DIRTY_SICK_TIME) {
      isSick = true;
      Serial.println(">>> SICK! Baby got sick from being dirty!");
      // saveState();  // COMMENTED FOR TESTING
    }
  }
  
  // Become sick if hungry for too long
  if (isHungry && !isSick) {
    if (hungerStartTime == 0) {
      hungerStartTime = currentTime;
    } else if (currentTime - hungerStartTime > HUNGRY_SICK_TIME) {
      isSick = true;
      Serial.println(">>> SICK! Baby got sick from hunger!");
      // saveState();  // COMMENTED FOR TESTING
    }
  } else {
    hungerStartTime = 0;
  }
  
  // Become sick if discipline very low
  if (discipline < 20 && !isSick) {
    isSick = true;
    Serial.println(">>> SICK! Baby got sick from low discipline!");
    // saveState();  // COMMENTED FOR TESTING
  }
  
  // Recover from sickness if conditions improve
  if (isSick) {
    bool canRecover = !isDirty && !isHungry && (discipline >= 30);
    if (canRecover) {
      // Need rest (sleep) to recover
      static unsigned long recoveryStartTime = 0;
      if (autoSleepMode) {
        if (recoveryStartTime == 0) {
          recoveryStartTime = currentTime;
        } else if (currentTime - recoveryStartTime > 300000) { // 5 min of sleep
          isSick = false;
          recoveryStartTime = 0;
          Serial.println(">>> RECOVERED! Baby is healthy again!");
          // saveState();  // COMMENTED FOR TESTING
        }
      } else {
        recoveryStartTime = 0;
      }
    }
  }
}

// ============ DISCIPLINE TRACKING SYSTEM ============
void updateDiscipline(int change) {
  int oldDiscipline = discipline;
  discipline = constrain(discipline + change, 0, 100);
  
  if (discipline != oldDiscipline) {
    Serial.printf(">>> DISCIPLINE: %d -> %d (%+d)\n", oldDiscipline, discipline, change);
    // saveState();  // COMMENTED FOR TESTING
  }
}

// ============ AUTOMATIC HUNGER SYSTEM ============
void scheduleNextHunger() {
  unsigned long hungerInterval;
  
  // Determine hunger interval based on life stage
  switch (lifeStage) {
    case INFANT:
      hungerInterval = random(INFANT_HUNGER_MIN, INFANT_HUNGER_MAX);
      Serial.printf(">>> INFANT hunger scheduled in %.1f minutes\n", hungerInterval / 60000.0);
      break;
    case CHILD:
      hungerInterval = random(CHILD_HUNGER_MIN, CHILD_HUNGER_MAX);
      Serial.printf(">>> CHILD hunger scheduled in %.1f hours\n", hungerInterval / 3600000.0);
      break;
    case TEEN:
      hungerInterval = random(TEEN_HUNGER_MIN, TEEN_HUNGER_MAX);
      Serial.printf(">>> TEEN hunger scheduled in %.1f hours\n", hungerInterval / 3600000.0);
      break;
    case ADULT:
      hungerInterval = random(ADULT_HUNGER_MIN, ADULT_HUNGER_MAX);
      Serial.printf(">>> ADULT hunger scheduled in %.1f hours\n", hungerInterval / 3600000.0);
      break;
    case OLD:
      hungerInterval = random(OLD_HUNGER_MIN, OLD_HUNGER_MAX);
      Serial.printf(">>> OLD hunger scheduled in %.1f hours\n", hungerInterval / 3600000.0);
      break;
    default:
      hungerInterval = INFANT_HUNGER_MIN;
      break;
  }
  
  nextHungerTime = millis() + hungerInterval;
  autoHungerInitialized = true;
}

void checkAutoHunger() {
  // Initialize hunger timer on first run after feeding
  if (!autoHungerInitialized && !isHungry) {
    scheduleNextHunger();
    return;
  }
  
  // Skip if already hungry
  if (isHungry) {
    return;
  }
  
  // Check if it's time to get hungry
  unsigned long currentTime = millis();
  if (currentTime >= nextHungerTime) {
    isHungry = true;
    feedLevel = 0;
    feedStepCount = 0;
    hungerStartTime = currentTime;
    autoHungerInitialized = false;  // Reset so next feeding schedules new hunger
    
    Serial.println(">>> AUTO-HUNGER! Baby is now hungry!");
    // saveState();  // COMMENTED FOR TESTING
  }
}

// ============ DAILY STATS & HEALTH ANIMATION SYSTEM ============ *UPDATED*
void updateDailyStats() {
  // Called every wake-up to update health animation level
  // Based on steps/day average and health score
  
  time_t now = time(nullptr);
  daysTracked++;
  
  // Calculate current health animation level
  if (healthScore < 2000) {
    currentHealthLevel = HEALTH_WEAK;
  }
  else if ((stepsPerDay < 8000) || (healthScore >= 2000 && healthScore < 4000)) {
    currentHealthLevel = HEALTH_TIRED;
  }
  else if (healthScore >= 4000 && healthScore <= 6000) {
    // Check if device has been running 10+ days
    if (daysTracked < 10) {
      currentHealthLevel = HEALTH_AVERAGE;  // Default for first 10 days
    } else {
      // After 10 days, continuous animation updates
      currentHealthLevel = HEALTH_AVERAGE;
    }
  }
  else if (stepsPerDay >= 6000 && stepsPerDay < 8000) {
    currentHealthLevel = HEALTH_STRONG;
  }
  else if (stepsPerDay >= 8000 && stepsPerDay < 10000) {
    currentHealthLevel = HEALTH_STRONG_PLUS;
  }
  else if (stepsPerDay >= 10000) {
    currentHealthLevel = HEALTH_MAX;
  }
  
  Serial.printf(">>> HEALTH ANIMATION UPDATE: Level %d | Steps: %d | Health: %d | Days: %d\n", 
                currentHealthLevel, stepsPerDay, healthScore, daysTracked);
}

// ============ GET HEALTH ANIMATION ICON ============ *UPDATED*
void drawHealthAnimation() {
  // Draw based on current health level
  int cx = 50;
  int cy = 15;
  
  switch(currentHealthLevel) {
    case HEALTH_WEAK:
      // Weak: X eyes
      display.drawLine(cx-2, cy-2, cx+2, cy+2, SSD1306_WHITE);
      display.drawLine(cx-2, cy+2, cx+2, cy-2, SSD1306_WHITE);
      break;
    
    case HEALTH_TIRED:
      // Tired: -_- face
      display.drawLine(cx-3, cy, cx-1, cy, SSD1306_WHITE);
      display.drawLine(cx+1, cy, cx+3, cy, SSD1306_WHITE);
      display.drawLine(cx-2, cy+3, cx+2, cy+3, SSD1306_WHITE);
      break;
    
    case HEALTH_AVERAGE:
      // Average: o_o face
      display.drawCircle(cx-2, cy, 1, SSD1306_WHITE);
      display.drawCircle(cx+2, cy, 1, SSD1306_WHITE);
      display.drawLine(cx-2, cy+3, cx+2, cy+3, SSD1306_WHITE);
      break;
    
    case HEALTH_STRONG:
      // Strong: ^_^ happy
      display.drawLine(cx-3, cy, cx-1, cy, SSD1306_WHITE);
      display.drawLine(cx+1, cy, cx+3, cy, SSD1306_WHITE);
      display.drawLine(cx-3, cy+3, cx+3, cy+3, SSD1306_WHITE);
      break;
    
    case HEALTH_STRONG_PLUS:
      // Strong+: ^_^ with stars
      display.drawLine(cx-3, cy-2, cx-1, cy-2, SSD1306_WHITE);
      display.drawLine(cx+1, cy-2, cx+3, cy-2, SSD1306_WHITE);
      display.drawPixel(cx-4, cy, SSD1306_WHITE);
      display.drawPixel(cx+4, cy, SSD1306_WHITE);
      break;
    
    case HEALTH_MAX:
      // Max: ^^^ super happy with heart
      display.drawPixel(cx, cy-3, SSD1306_WHITE);
      display.drawLine(cx-2, cy-1, cx+2, cy-1, SSD1306_WHITE);
      display.drawLine(cx-3, cy+2, cx+3, cy+2, SSD1306_WHITE);
      break;
  }
}
