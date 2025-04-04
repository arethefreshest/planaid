namespace backend.Models
{
    public class FileLoggerOptions
    {
        public string Path { get; set; } = string.Empty;
        public int MaxFileSizeInMB { get; set; } = 10;
        public int MaxFileCount { get; set; } = 5;
    }
} 