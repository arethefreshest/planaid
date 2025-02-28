namespace backend.Models
{
    public class ProcessedPdfDocument
    {
        public string Title { get; set; } = string.Empty;
        public DateTime ProcessedDate { get; set; }
        public int TotalPages { get; set; }
        public List<PageContent> Pages { get; set; } = new();
        public Dictionary<string, string> Metadata { get; set; } = new();
    }

    public class PageContent
    {
        public int PageNumber { get; set; }
        public string Content { get; set; } = string.Empty;
    }
} 