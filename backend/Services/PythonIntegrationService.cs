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

namespace backend.Services {
    public class PythonIntegrationService : IPythonIntegrationService
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<PythonIntegrationService> _logger;

        public PythonIntegrationService(HttpClient httpClient, ILogger<PythonIntegrationService> logger)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            
            if (_httpClient.BaseAddress == null)
            {
                throw new InvalidOperationException("HttpClient BaseAddress is not configured");
            }
            _logger.LogInformation($"PythonIntegrationService initialized with BaseAddress: {_httpClient.BaseAddress}");

            _httpClient.Timeout = TimeSpan.FromMinutes(5); // Increase timeout to 5 minutes
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
                
                // Create StreamContent with proper content type
                var plankartContent = new StreamContent(File.OpenRead(plankartPath));
                plankartContent.Headers.ContentType = new MediaTypeHeaderValue("application/pdf");
                
                var bestemmelserContent = new StreamContent(File.OpenRead(bestemmelserPath));
                bestemmelserContent.Headers.ContentType = new MediaTypeHeaderValue("application/pdf");
                
                // Add files to form data with content type
                formData.Add(plankartContent, "plankart", Path.GetFileName(plankartPath));
                formData.Add(bestemmelserContent, "bestemmelser", Path.GetFileName(bestemmelserPath));
                
                if (!string.IsNullOrEmpty(sosiPath))
                {
                    var sosiContent = new StreamContent(File.OpenRead(sosiPath));
                    sosiContent.Headers.ContentType = new MediaTypeHeaderValue("text/xml");
                    formData.Add(sosiContent, "sosi", Path.GetFileName(sosiPath));
                }

                var requestUri = new Uri(_httpClient.BaseAddress!, "api/check-field-consistency");
                _logger.LogInformation($"Sending request to: {requestUri}");
                var response = await _httpClient.PostAsync(requestUri, formData);
                var content = await response.Content.ReadAsStringAsync();
                
                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogError($"Python service returned error: {content}");
                    throw new HttpRequestException($"Python service error: {content}");
                }
                
                return content;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking field consistency");
                throw;
            }
        }
    }
}