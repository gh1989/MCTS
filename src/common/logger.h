#ifndef LOGGER_H_
#define LOGGER_H_

#include <iostream>
#include <string>

enum class LogLevel {
    DEBUG,
    TEST,
    INFO,
    WARNING,
    ERROR
};

class Logger {
 public:
  static void Log(LogLevel level, const std::string& message) {
    if (level >= current_level_) {
      std::cout << "[" << ToString(level) << "] " << message << std::endl;
    }
  }

  static void SetLogLevel(LogLevel level) {
    current_level_ = level;
  }

 private:
  static std::string ToString(LogLevel level) {
    switch (level) {
      case LogLevel::TEST: return "TEST";
      case LogLevel::DEBUG: return "DEBUG";
      case LogLevel::INFO: return "INFO";
      case LogLevel::WARNING: return "WARNING";
      case LogLevel::ERROR: return "ERROR";
      default: return "UNKNOWN";
    }
  }

  static LogLevel current_level_;
};

#endif  // LOGGER_H_