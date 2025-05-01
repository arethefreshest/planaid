using Microsoft.Extensions.Logging;
using backend.Models;
using Serilog;

namespace backend.Extensions
{
    public static class LoggingExtensions
    {
        public static ILoggingBuilder AddFile(this ILoggingBuilder builder, FileLoggerOptions options)
        {
            var logger = new LoggerConfiguration()
                .WriteTo.File(
                    options.Path,
                    fileSizeLimitBytes: options.MaxFileSizeInMB * 1024 * 1024,
                    retainedFileCountLimit: options.MaxFileCount,
                    rollingInterval: RollingInterval.Day)
                .CreateLogger();

            builder.AddSerilog(logger);
            return builder;
        }
    }
} 