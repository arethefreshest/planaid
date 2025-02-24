namespace backend.Models
{
    using System.Text.Json.Serialization;

    public class PythonResponse<T>
    {
        [JsonPropertyName("status")]
        public string Status { get; set; } = string.Empty;

        [JsonPropertyName("result")]
        public T Result { get; set; } = default!;
    }
} 