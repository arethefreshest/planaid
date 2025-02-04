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
  
  // Send to Docker logs via stdout
  if (typeof window === 'undefined') {
    // Node.js environment (server-side)
    process.stdout.write(JSON.stringify(logMessage) + '\n');
  } else {
    // Browser environment (client-side)
    console.log(JSON.stringify(logMessage));
  }
  
  // Also log to browser console in development
  if (process.env.NODE_ENV === 'development' && typeof window !== 'undefined') {
    switch (level) {
      case 'info':
        console.info(message, data);
        break;
      case 'error':
        console.error(message, data);
        break;
      case 'warn':
        console.warn(message, data);
        break;
      case 'debug':
        console.debug(message, data);
        break;
    }
  }
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