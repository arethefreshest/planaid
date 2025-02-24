/*
Python Integration Service

This service handles communication with the Python microservice for regulatory
document processing. It manages file uploads and consistency checks between
plankart, bestemmelser, and SOSI files.

Features:
- Asynchronous file processing
- Multi-part form data handling
- Error handling and logging
- Support for optional SOSI files
*/

using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Net.Http.Headers;
using Microsoft.AspNetCore.Http;
using System.Text.Json;
using System.Collections.Generic;

namespace backend.Services {
    public class PythonIntegrationService : IPythonIntegrationService
    {
        private readonly HttpClient _httpClient;
        private readonly HttpClient _nerHttpClient;
        private readonly ILogger<PythonIntegrationService> _logger;

        public PythonIntegrationService(IHttpClientFactory httpClientFactory, ILogger<PythonIntegrationService> logger)
        {
            _httpClient = httpClientFactory.CreateClient("PythonService");
            _nerHttpClient = httpClientFactory.CreateClient("NerService");
            _logger = logger;
        }

        /// <inheritdoc/>
        public async Task<string> CheckConsistencyAsync(string plankartPath, string bestemmelserPath, string? sosiPath = null)
        {
            try
            {
                if (_httpClient.BaseAddress == null)
                {
                    throw new InvalidOperationException("HttpClient BaseAddress is not configured");
                }

                using var formData = new MultipartFormDataContent();
                
                // Add files with proper content type
                using var plankartContent = new StreamContent(File.OpenRead(plankartPath));
                plankartContent.Headers.ContentType = new MediaTypeHeaderValue(GetContentType(plankartPath));
                formData.Add(plankartContent, "plankart", Path.GetFileName(plankartPath));

                using var bestemmelserContent = new StreamContent(File.OpenRead(bestemmelserPath));
                bestemmelserContent.Headers.ContentType = new MediaTypeHeaderValue("application/pdf");
                formData.Add(bestemmelserContent, "bestemmelser", Path.GetFileName(bestemmelserPath));

                if (sosiPath != null)
                {
                    using var sosiContent = new StreamContent(File.OpenRead(sosiPath));
                    sosiContent.Headers.ContentType = new MediaTypeHeaderValue("text/xml");
                    formData.Add(sosiContent, "sosi", Path.GetFileName(sosiPath));
                }

                var requestUri = new Uri(_httpClient.BaseAddress!, "api/check-field-consistency");
                _logger.LogInformation($"Sending request to: {requestUri}");
                
                var response = await _httpClient.PostAsync(requestUri, formData);
                var content = await response.Content.ReadAsStringAsync();
                
                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogError($"Python service returned error. Status: {response.StatusCode}, Content: {content}");
                    throw new HttpRequestException($"Python service error: {content}");
                }
                
                if (string.IsNullOrEmpty(content))
                {
                    _logger.LogError("Python service returned empty response");
                    throw new HttpRequestException("Python service returned empty response");
                }

                return content;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking field consistency");
                throw;
            }
        }

        private string GetContentType(string filePath)
        {
            return Path.GetExtension(filePath).ToLower() switch
            {
                ".pdf" => "application/pdf",
                ".xml" => "text/xml",
                _ => "application/octet-stream"
            };
        }

        public async Task<string[]> ExtractBestemmelsesFields(string bestemmelserPath)
        {
            try
            {
                using var formData = new MultipartFormDataContent();
                using var fileContent = new StreamContent(File.OpenRead(bestemmelserPath));
                fileContent.Headers.ContentType = new MediaTypeHeaderValue("application/pdf");
                formData.Add(fileContent, "file", Path.GetFileName(bestemmelserPath));

                var response = await _nerHttpClient.PostAsync("api/extract-fields", formData);
                var content = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogError($"NER service returned error: {content}");
                    throw new HttpRequestException($"NER service error: {content}");
                }

                var result = JsonSerializer.Deserialize<Dictionary<string, List<string>>>(content);
                return result?["fields"]?.ToArray() ?? Array.Empty<string>();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error extracting fields from bestemmelser");
                throw;
            }
        }
    }
}