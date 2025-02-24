using backend.Models;
using iText.Kernel.Pdf;
using iText.Kernel.Pdf.Canvas.Parser;
using iText.Kernel.Pdf.Canvas.Parser.Listener;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace backend.Services
{
    /*
    PDF Processing Service

    This service handles the processing and analysis of regulatory PDF documents.
    It supports different types of documents (plankart, bestemmelser) and extracts
    relevant information such as field identifiers and metadata.

    Features:
    - PDF text extraction
    - Field identifier recognition
    - Metadata extraction
    - Document persistence
    - Multiple document type support
    */

    public class PdfProcessingService
    {
        private readonly IPythonIntegrationService _pythonService;
        private readonly ILogger<PdfProcessingService> _logger;

        public PdfProcessingService(IPythonIntegrationService pythonService, ILogger<PdfProcessingService> logger)
        {
            _pythonService = pythonService;
            _logger = logger;
        }

        public async Task<string> ProcessPdfAsync(string filePath, PdfType type)
        {
            try
            {
                // For single file processing
                string? plankartPath = type == PdfType.PlanMap ? filePath : null;
                string? bestemmelserPath = type == PdfType.Regulations ? filePath : null;

                if (plankartPath == null && bestemmelserPath == null)
                {
                    throw new ArgumentException("Invalid PDF type or file path");
                }

                var result = await _pythonService.CheckConsistencyAsync(
                    plankartPath ?? string.Empty,
                    bestemmelserPath ?? string.Empty
                );
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing PDF");
                throw;
            }
        }

        public async Task<string> CheckConsistencyAsync(string plankartPath, string bestemmelserPath, string? sosiPath = null)
        {
            try
            {
                return await _pythonService.CheckConsistencyAsync(plankartPath, bestemmelserPath, sosiPath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking consistency");
                throw;
            }
        }

        /// <summary>
        /// Processes a plan map PDF and extracts field identifiers and prefixes.
        /// </summary>
        private async Task<ProcessedPdfDocument> ProcessPlanMapAsync(string filePath)
        {
            var baseFileName = Path.GetFileNameWithoutExtension(filePath);
            using var reader = new PdfReader(filePath);
            using var document = new PdfDocument(reader);

            var prefixSet = new HashSet<string>();
            var fieldIdentifiers = new HashSet<string>();
            var fullText = PdfTextExtractor.GetTextFromPage(document.GetPage(1));

            // Find the legend section
            var legendMatch = Regex.Match(fullText, @"Tegnforklaring.*?(?=Kartopplysninger|$)", 
                RegexOptions.Singleline);

            if (legendMatch.Success)
            {
                var legendText = legendMatch.Value;
                
                // First pass: Find all potential prefixes in the legend
                var prefixPatterns = new[]
                {
                    // Base pattern for most prefixes (o_BKS, o_BR, o_BE, etc.)
                    @"(?:^|\n)\s*([oO]_[A-ZÆØÅ]{2,}\d*)\s+(?:[A-ZÆØÅa-zæøå]|[-–])",
                    // H-numbers specific pattern
                    @"(?:^|\n)\s*(H\d{3})\s+(?:[A-ZÆØÅa-zæøå]|[-–])",
                    // SNØ pattern
                    @"(?:^|\n)\s*(#\d+\s+SNØ)\s+(?:[A-ZÆØÅa-zæøå]|[-–])",
                    // Two-letter prefixes with optional prefix (f_BE, BE, etc.)
                    @"(?:^|\n)\s*(?:[fF]_)?([A-ZÆØÅ]{2})\s+(?:[A-ZÆØÅa-zæøå]|[-–])"
                };

                foreach (var pattern in prefixPatterns)
                {
                    var matches = Regex.Matches(legendText, pattern, RegexOptions.Multiline);
                    foreach (Match match in matches)
                    {
                        var code = match.Groups[1].Value.Trim();
                        
                        if (code.StartsWith("#"))
                        {
                            prefixSet.Add("#0 SNØ");
                        }
                        else if (code.StartsWith("H") && Regex.IsMatch(code, @"H\d{3}"))
                        {
                            prefixSet.Add(code);
                        }
                        else
                        {
                            var prefix = Regex.Match(code, @"^[A-ZÆØÅ]+").Value;
                            if (!string.IsNullOrEmpty(prefix))
                            {
                                prefixSet.Add(prefix);
                            }
                        }
                    }
                }

                // Second pass: Look specifically for two-letter codes that might be missed
                var twoLetterMatches = Regex.Matches(legendText, @"(?:^|\n)\s*(BE|GF|SF|SV|SPA)\s", RegexOptions.Multiline);
                foreach (Match match in twoLetterMatches)
                {
                    prefixSet.Add(match.Groups[1].Value);
                }
            }

            // Extract field identifiers from the map
            var mapMatches = Regex.Matches(fullText, 
                @"(?:^|\s)((?:o_|f_)?[A-ZÆØÅ]+(?:\d+)?|H\d{3}|#\d+\s+SNØ)(?=\s|$)", 
                RegexOptions.Multiline);

            foreach (Match match in mapMatches)
            {
                var identifier = match.Groups[1].Value.Trim();
                if ((identifier.Contains("_") || 
                     (identifier.StartsWith("#") && identifier != "#0 SNØ") || 
                     (identifier.StartsWith("H") && Regex.IsMatch(identifier, @"H\d{3}")) ||
                     Regex.IsMatch(identifier, @"[A-ZÆØÅ]+\d+") ||
                     Regex.IsMatch(identifier, @"^(BIA|BKS|BR)$")) && // Add base prefixes found in map
                    identifier != "NN2000")
                {
                    fieldIdentifiers.Add(identifier);
                }
            }

            // Filter out any coordinate-like patterns and system identifiers
            fieldIdentifiers.RemoveWhere(f => 
                Regex.IsMatch(f, @"^[NØ]\d{6,}$") || 
                f == "#0 SNØ" || 
                f == "NN2000");

            var result = new ProcessedPdfDocument
            {
                Title = baseFileName,
                ProcessedDate = DateTime.UtcNow,
                TotalPages = document.GetNumberOfPages(),
                Metadata = new Dictionary<string, string>
                {
                    { "Type", "PlanMap" },
                    { "FieldIdentifiers", JsonSerializer.Serialize(fieldIdentifiers.OrderBy(f => f).ToList()) },
                    { "FoundPrefixes", JsonSerializer.Serialize(prefixSet.OrderBy(p => p).ToList()) }
                }
            };

            await SaveProcessedDocumentAsync(result, baseFileName);
            return result;
        }

        /// <summary>
        /// Validates if a field identifier is valid within its context.
        /// </summary>
        private bool IsValidFieldIdentifier(string identifier, string context)
        {
            // Explicitly exclude unwanted identifiers
            var excludedPrefixes = new[] { "A", "BYA", "N", "PBL", "T" };
            
            // Check if the identifier starts with any excluded prefix
            foreach (var excluded in excludedPrefixes)
            {
                if (identifier == excluded || 
                    identifier.StartsWith($"o_{excluded}") || 
                    identifier.StartsWith($"f_{excluded}") ||
                    identifier.StartsWith($"{excluded}_") ||
                    identifier.StartsWith($"{excluded}-"))
                {
                    return false;
                }
            }

            // Must contain at least one number (except for special cases)
            if (!identifier.StartsWith("#") && !Regex.IsMatch(identifier, @"\d"))
            {
                return false;
            }

            // Check if it appears in a proper context
            // Look for the identifier followed by a description
            var afterIdentifier = context.Substring(
                context.IndexOf(identifier) + identifier.Length,
                Math.Min(100, context.Length - context.IndexOf(identifier) - identifier.Length)
            );

            // Valid identifiers are usually followed by a description
            return Regex.IsMatch(afterIdentifier, @"^\s*[-–]\s*[A-ZÆØÅa-zæøå]");
        }

        /// <summary>
        /// Extracts the prefix from a field identifier.
        /// </summary>
        private string? ExtractPrefix(string identifier)
        {
            if (identifier.StartsWith("#"))
                return "#0 SNØ"; // Adjusted to capture the general prefix

            // Remove o_ or f_ prefix if present
            var withoutPrefix = identifier.Replace("o_", "").Replace("f_", "");
            
            // Extract the base prefix (letters before any numbers)
            var match = Regex.Match(withoutPrefix, @"^([A-ZÆØÅ]+)");
            if (!match.Success)
                return null;

            var prefix = match.Groups[1].Value;
            
            // Special case for H-numbers
            if (prefix == "H" && withoutPrefix.Length > 1)
            {
                var fullPrefix = Regex.Match(withoutPrefix, @"^H\d{3}");
                if (fullPrefix.Success)
                    return fullPrefix.Value;
            }

            return prefix;
        }

        /// <summary>
        /// Processes a regulations PDF and extracts page content.
        /// </summary>
        private async Task<ProcessedPdfDocument> ProcessRegulationsAsync(string filePath)
        {
            var baseFileName = Path.GetFileNameWithoutExtension(filePath);
            using var reader = new PdfReader(filePath);
            using var document = new PdfDocument(reader);

            var pages = new List<PageContent>();
            for (int i = 1; i <= document.GetNumberOfPages(); i++)
            {
                var page = document.GetPage(i);
                var text = PdfTextExtractor.GetTextFromPage(page);
                pages.Add(new PageContent
                {
                    PageNumber = i,
                    Content = text
                });
            }

            var result = new ProcessedPdfDocument
            {
                Title = baseFileName,
                ProcessedDate = DateTime.UtcNow,
                TotalPages = document.GetNumberOfPages(),
                Pages = pages
            };

            await SaveProcessedDocumentAsync(result, baseFileName);
            return result;
        }

        /// <summary>
        /// Saves processed document data to JSON file.
        /// </summary>
        private async Task SaveProcessedDocumentAsync(ProcessedPdfDocument document, string baseFileName)
        {
            var jsonString = JsonSerializer.Serialize(document, new JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            
            var outputPath = Path.Combine("processed_pdfs", $"{baseFileName}.json");
            Directory.CreateDirectory("processed_pdfs"); // Ensure directory exists
            await File.WriteAllTextAsync(outputPath, jsonString);
        }
    }

    public enum PdfType
    {
        Regulations,
        PlanMap,
        Consistency
    }

    /// <summary>
    /// Represents extracted fields and prefixes from a plan map.
    /// </summary>
    public class PlanMapFields
    {
        public string Title { get; set; } = string.Empty;
        public DateTime ProcessedDate { get; set; }
        public List<string> FieldIdentifiers { get; set; } = new();
        public List<string> FoundPrefixes { get; set; } = new();
    }
}

