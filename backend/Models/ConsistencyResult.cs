namespace backend.Models
{
    using System.Text.Json.Serialization;

    public class ConsistencyResult
    {
        [JsonPropertyName("matching_fields")]
        public List<string> MatchingFields { get; set; } = new();

        [JsonPropertyName("only_in_plankart")]
        public List<string> OnlyInPlankart { get; set; } = new();

        [JsonPropertyName("only_in_bestemmelser")]
        public List<string> OnlyInBestemmelser { get; set; } = new();

        [JsonPropertyName("only_in_sosi")]
        public List<string> OnlyInSosi { get; set; } = new();

        [JsonPropertyName("is_consistent")]
        public bool IsConsistent { get; set; }

        [JsonPropertyName("document_fields")]
        public Dictionary<string, DocumentFields> DocumentFields { get; set; } = new();

        [JsonPropertyName("metadata")]
        public Dictionary<string, string>? Metadata { get; set; }
    }

    public class DocumentFields
    {
        [JsonPropertyName("raw_fields")]
        public List<string> RawFields { get; set; } = new();

        [JsonPropertyName("normalized_fields")]
        public List<string> NormalizedFields { get; set; } = new();
    }
} 