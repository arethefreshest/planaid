namespace backend.Models
{
    public class ServiceUrls
    {
        public ServiceUrl Local { get; set; } = new();
        public ServiceUrl Docker { get; set; } = new();
    }

    public class ServiceUrl
    {
        public string PythonServiceUrl { get; set; } = string.Empty;
        public string NerServiceUrl { get; set; } = string.Empty;
    }
} 