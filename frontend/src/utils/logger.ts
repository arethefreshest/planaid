/**
 * Logger Utility
 * 
 * Provides consistent logging functionality across the application,
 * supporting both browser and Node.js environments.
 * 
 * Features:
 * - Multiple log levels (info, error, warn, debug)
 * - Timestamp inclusion
 * - Environment-aware output
 * - Development mode console output
 * - Docker-compatible logging
 */

/** Available log levels */
type LogLevel = 'info' | 'error' | 'warn' | 'debug';

/** Structure of a log message */
interface LogMessage {
  timestamp: string;
  level: LogLevel;
  message: string;
  data?: any;
}

const API_URL = process.env.REACT_APP_API_URL ?? 'http://localhost:5251';

/**
 * Core logging function that handles message formatting and output
 */
const log = (level: LogLevel, message: string, data?: any) => {
  const logMessage: LogMessage = {
    timestamp: new Date().toISOString(),
    level,
    message,
    data
  };
  
  // Always log to console in development
  if (process.env.NODE_ENV === 'development') {
    const timestamp = new Date().toLocaleTimeString();
    switch (level) {
      case 'info':
        console.info(`[${timestamp}] [INFO] ${message}`, data);
        break;
      case 'error':
        console.error(`[${timestamp}] [ERROR] ${message}`, data);
        break;
      case 'warn':
        console.warn(`[${timestamp}] [WARN] ${message}`, data);
        break;
      case 'debug':
        console.debug(`[${timestamp}] [DEBUG] ${message}`, data);
        break;
    }
  }
  
  // Send to backend
  fetch(`${API_URL}/api/log`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(logMessage)
  }).catch(error => {
    console.error('Failed to send log to backend:', error);
  });
};

/**
 * Exported logger interface with typed methods for each log level
 */
export const logger = {
  info: (message: string, data?: any) => log('info', message, data),
  error: (message: string, data?: any) => log('error', message, data),
  warn: (message: string, data?: any) => log('warn', message, data),
  debug: (message: string, data?: any) => log('debug', message, data)
};

// Type exports
export type { LogMessage, LogLevel }; 