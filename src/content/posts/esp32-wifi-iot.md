---
title: ESP32 WiFi IoT Application Development
published: 2025-01-05
description: 使用ESP32开发物联网应用，涵盖WiFi连接、HTTP客户端、MQTT通信和Web服务器。
tags: [ESP32, WiFi, IoT, Arduino]
category: 技术
draft: false
---

## ESP32简介

ESP32是乐鑫科技推出的一款功能强大的WiFi和蓝牙双模芯片。它不仅具有丰富的外设接口，还内置了WiFi和蓝牙功能，非常适合物联网（IoT）应用开发。

## ESP32的主要特性

- 双核32位处理器（最高240MHz）
- 内置WiFi 802.11 b/g/n
- 内置蓝牙4.2和BLE
- 丰富的外设：GPIO、ADC、DAC、SPI、I2C、UART等
- 低功耗设计
- 价格低廉，易于获取

## 开发环境搭建

### 1. 安装Arduino IDE

虽然ESP32可以使用多种开发环境，但Arduino IDE是最简单易用的选择。

### 2. 添加ESP32开发板支持

1. 打开Arduino IDE，进入"文件" → "首选项"
2. 在"附加开发板管理器网址"中添加：
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. 打开"工具" → "开发板" → "开发板管理器"
4. 搜索"esp32"并安装"esp32 by Espressif Systems"

## WiFi连接示例

以下是一个基本的WiFi连接示例：

```cpp
#include <WiFi.h>

const char* ssid = "你的WiFi名称";
const char* password = "你的WiFi密码";

void setup() {
    Serial.begin(115200);

    // 连接WiFi
    WiFi.begin(ssid, password);

    Serial.print("正在连接WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.print("WiFi已连接！IP地址: ");
    Serial.println(WiFi.localIP());
}

void loop() {
    // 检查WiFi连接状态
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("WiFi连接正常");
    } else {
        Serial.println("WiFi连接断开，尝试重连...");
        WiFi.begin(ssid, password);
    }
    delay(5000);
}
```

## HTTP客户端示例

连接WiFi后，我们可以发送HTTP请求获取数据：

```cpp
#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid = "你的WiFi名称";
const char* password = "你的WiFi密码";

void setup() {
    Serial.begin(115200);
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println("\nWiFi已连接");
}

void loop() {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;

        // 发送GET请求
        http.begin("http://httpbin.org/get");
        int httpCode = http.GET();

        if (httpCode > 0) {
            String payload = http.getString();
            Serial.println(httpCode);
            Serial.println(payload);
        } else {
            Serial.println("请求失败");
        }

        http.end();
    }

    delay(10000);
}
```

## MQTT通信

MQTT是物联网中常用的轻量级消息传输协议。ESP32可以轻松实现MQTT通信：

```cpp
#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "你的WiFi名称";
const char* password = "你的WiFi密码";
const char* mqtt_server = "broker.hivemq.com";  // 公共MQTT服务器

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
    Serial.begin(115200);
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    client.setServer(mqtt_server, 1883);
    client.setCallback(callback);
}

void callback(char* topic, byte* payload, unsigned int length) {
    Serial.print("收到消息 [");
    Serial.print(topic);
    Serial.print("]: ");
    for (int i = 0; i < length; i++) {
        Serial.print((char)payload[i]);
    }
    Serial.println();
}

void reconnect() {
    while (!client.connected()) {
        Serial.print("尝试连接MQTT服务器...");
        if (client.connect("ESP32Client")) {
            Serial.println("已连接");
            client.subscribe("esp32/topic");
        } else {
            Serial.print("失败，重试中...");
            delay(5000);
        }
    }
}

void loop() {
    if (!client.connected()) {
        reconnect();
    }
    client.loop();

    // 发布消息
    client.publish("esp32/topic", "Hello from ESP32!");
    delay(5000);
}
```

## Web服务器示例

ESP32还可以作为Web服务器，提供Web界面：

```cpp
#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "你的WiFi名称";
const char* password = "你的WiFi密码";

WebServer server(80);

void handleRoot() {
    String html = "<html><body>";
    html += "<h1>ESP32 Web服务器</h1>";
    html += "<p>这是一个ESP32 Web服务器示例</p>";
    html += "</body></html>";
    server.send(200, "text/html", html);
}

void setup() {
    Serial.begin(115200);
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println("\nWiFi已连接");
    Serial.print("IP地址: ");
    Serial.println(WiFi.localIP());

    server.on("/", handleRoot);
    server.begin();
    Serial.println("Web服务器已启动");
}

void loop() {
    server.handleClient();
}
```

## 实际应用场景

- **智能家居**：控制灯光、温度、安防系统
- **环境监测**：实时监测温度、湿度、空气质量
- **远程控制**：通过手机APP控制设备
- **数据采集**：收集传感器数据并上传到云端

## 总结

ESP32凭借其强大的功能和低廉的价格，是物联网开发的理想选择。通过WiFi连接，我们可以轻松实现设备与互联网的通信，构建各种智能应用。

> **提示：** 在实际项目中，要注意WiFi连接的稳定性处理，添加重连机制和错误处理，确保设备能够可靠运行。
