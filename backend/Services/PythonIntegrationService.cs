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

namespace backend.Services 
{
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

            // Log file sizes for debugging
            _logger.LogInformation($"📦 Uploading files to Python service:");
            _logger.LogInformation($"  - Plankart: {Path.GetFileName(plankartPath)} ({new FileInfo(plankartPath).Length} bytes)");
            _logger.LogInformation($"  - Bestemmelser: {Path.GetFileName(bestemmelserPath)} ({new FileInfo(bestemmelserPath).Length} bytes)");
            if (sosiPath != null)
            {
                _logger.LogInformation($"  - SOSI: {Path.GetFileName(sosiPath)} ({new FileInfo(sosiPath).Length} bytes)");
            }

            // Open file streams
            using var plankartStream = File.OpenRead(plankartPath);
            using var bestemmelserStream = File.OpenRead(bestemmelserPath);
            using var formData = new MultipartFormDataContent();

            // Add plankart
            var plankartContent = new StreamContent(plankartStream);
            plankartContent.Headers.ContentType = new MediaTypeHeaderValue(GetContentType(plankartPath));
            formData.Add(plankartContent, "plankart", Path.GetFileName(plankartPath));

            // Add bestemmelser
            var bestemmelserContent = new StreamContent(bestemmelserStream);
            bestemmelserContent.Headers.ContentType = new MediaTypeHeaderValue("application/pdf");
            formData.Add(bestemmelserContent, "bestemmelser", Path.GetFileName(bestemmelserPath));

            // Handle SOSI if present
            StreamContent? sosiContent = null;
            FileStream? sosiStream = null;

            try
            {
                if (sosiPath != null)
                {
                    sosiStream = File.OpenRead(sosiPath);
                    sosiContent = new StreamContent(sosiStream);
                    sosiContent.Headers.ContentType = new MediaTypeHeaderValue("text/plain"); // ✅ Important
                    formData.Add(sosiContent, "sosi", Path.GetFileName(sosiPath));
                }

                var requestUri = new Uri(_httpClient.BaseAddress!, "api/check-field-consistency");
                _logger.LogInformation($"➡️ Sending request to: {requestUri}");

                var response = await _httpClient.PostAsync(requestUri, formData);
                var content = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogError($"❌ Python service error. Status: {response.StatusCode}, Content: {content}");
                    throw new HttpRequestException($"Python service error: {content}");
                }

                if (string.IsNullOrEmpty(content))
                {
                    _logger.LogError("❌ Python service returned empty response");
                    throw new HttpRequestException("Python service returned empty response");
                }

                return content;
            }
            finally
            {
                sosiContent?.Dispose();
                sosiStream?.Dispose();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error checking field consistency");
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
                var fileStream = File.OpenRead(bestemmelserPath);
                var fileContent = new StreamContent(fileStream);
                var formData = new MultipartFormDataContent();

                try
                {
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
                finally
                {
                    fileContent.Dispose();
                    fileStream.Dispose();
                    formData.Dispose();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error extracting fields from bestemmelser");
                throw;
            }
        }
    }
}