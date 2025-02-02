#pragma once
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;

// Enum to represent log levels
enum LogLevel { DEBUG, INFO, WARNING, ERROR, CRITICAL };

class Logger {
public:
    // Constructor: Opens the log file in append mode
    Logger(const string& filename);


    // Destructor: Closes the log file
    ~Logger() ;

    // Logs a message with a given log level
    void log(LogLevel level, const string& message);

private:
	// File stream for the log file
    ofstream logFile; 

    // Converts log level to a string for output
    string levelToString(LogLevel level);
};


