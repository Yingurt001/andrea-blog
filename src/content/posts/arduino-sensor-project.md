---
title: "Arduino Sensor Project: Temperature and Humidity Monitoring System"
published: 2025-01-10
description: 使用Arduino Uno和DHT22传感器构建实时温湿度监测系统。
tags: [Arduino, Sensors, IoT]
category: 技术
draft: false
---

## 项目介绍

本项目使用Arduino Uno和DHT22温湿度传感器构建一个实时温湿度监测系统。系统可以读取环境温度和湿度，并通过串口输出数据，也可以连接LCD显示屏进行本地显示。

## 所需材料

- Arduino Uno开发板 x 1
- DHT22温湿度传感器 x 1
- 10kΩ上拉电阻 x 1
- 面包板和杜邦线若干
- （可选）16x2 LCD显示屏

## 硬件连接

### DHT22连接方式

- **VCC** → Arduino 5V
- **GND** → Arduino GND
- **DATA** → Arduino Digital Pin 2（通过10kΩ上拉电阻连接到VCC）

## 软件实现

### 1. 安装DHT库

在Arduino IDE中，打开"工具" → "管理库"，搜索"DHT sensor library"并安装。

### 2. 完整代码

```cpp
#include <DHT.h>

#define DHTPIN 2        // DHT22数据引脚
#define DHTTYPE DHT22   // 传感器类型

DHT dht(DHTPIN, DHTTYPE);

void setup() {
    Serial.begin(9600);
    Serial.println("DHT22温湿度监测系统");
    dht.begin();
}

void loop() {
    // 读取需要约250ms，延迟2秒以确保数据稳定
    delay(2000);

    // 读取湿度
    float humidity = dht.readHumidity();
    // 读取温度（摄氏度）
    float temperature = dht.readTemperature();

    // 检查读取是否成功
    if (isnan(humidity) || isnan(temperature)) {
        Serial.println("读取传感器数据失败！");
        return;
    }

    // 计算热指数（体感温度）
    float heatIndex = dht.computeHeatIndex(temperature, humidity, false);

    // 输出数据
    Serial.print("湿度: ");
    Serial.print(humidity);
    Serial.print(" %\t");
    Serial.print("温度: ");
    Serial.print(temperature);
    Serial.print(" °C\t");
    Serial.print("体感温度: ");
    Serial.print(heatIndex);
    Serial.println(" °C");
}
```

## 功能扩展

### 添加LCD显示

如果你有LCD显示屏，可以添加以下代码来显示数据：

```cpp
#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 6);

void setup() {
    lcd.begin(16, 2);
    // ... 其他初始化代码
}

void loop() {
    // ... 读取传感器数据

    // 显示在LCD上
    lcd.setCursor(0, 0);
    lcd.print("Temp: ");
    lcd.print(temperature);
    lcd.print("C");

    lcd.setCursor(0, 1);
    lcd.print("Humidity: ");
    lcd.print(humidity);
    lcd.print("%");
}
```

### 添加数据记录功能

可以使用SD卡模块记录数据，或者通过WiFi模块（如ESP8266）将数据上传到云端。

## 常见问题

- **读取失败**：检查接线是否正确，确保上拉电阻已连接
- **数据不准确**：DHT22需要2秒的读取间隔，不要过于频繁读取
- **串口无输出**：检查波特率设置是否正确（9600）

## 项目总结

这个项目展示了如何使用Arduino读取传感器数据。你可以在此基础上扩展更多功能，比如：

- 添加多个传感器
- 实现数据可视化
- 添加报警功能（温度/湿度超出范围时报警）
- 连接物联网平台实现远程监控

> **提示：** DHT22的精度为±0.5°C（温度）和±1%RH（湿度），适合大多数应用场景。如果需要更高精度，可以考虑使用SHT30等传感器。
