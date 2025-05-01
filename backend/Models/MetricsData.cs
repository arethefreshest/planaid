using System.Text.Json.Serialization;

namespace backend.Models
{
    public class MetricsData
    {
        [JsonPropertyName("run_id")]
        public string RunId { get; set; } = Guid.NewGuid().ToString();

        [JsonPropertyName("operation")]
        public string Operation { get; set; } = string.Empty;

        [JsonPropertyName("timestamp")]
        public string Timestamp { get; set; } = DateTime.UtcNow.ToString("o");

        [JsonPropertyName("timings")]
        public Dictionary<string, double> Timings { get; set; } = new();

        [JsonPropertyName("field_counts")]
        public Dictionary<string, int> FieldCounts { get; set; } = new();

        [JsonPropertyName("resource_usage")]
        public Dictionary<string, double> ResourceUsage { get; set; } = new();

        [JsonPropertyName("document_info")]
        public DocumentInfo DocumentInfo { get; set; } = new();

        [JsonPropertyName("error")]
        public string? Error { get; set; }
    }

    public class DocumentInfo
    {
        [JsonPropertyName("documents")]
        public Dictionary<string, DocumentDetails> Documents { get; set; } = new();
    }

    public class DocumentDetails
    {
        [JsonPropertyName("filename")]
        public string Filename { get; set; } = string.Empty;

        [JsonPropertyName("size")]
        public long? Size { get; set; }
    }
} 