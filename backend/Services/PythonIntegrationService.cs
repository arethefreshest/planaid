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
            _httpClient = httpClient;
            _logger = logger;
        }

        /// <inheritdoc/>
        public async Task<string> CheckConsistencyAsync(string plankartPath, string bestemmelserPath, string? sosiPath = null)
        {
            try
            {
                using var formData = new MultipartFormDataContent();
                
                // Helper function to add file with proper content type
                async Task AddFileToForm(string filePath, string fieldName)
                {
                    var fileContent = new ByteArrayContent(await File.ReadAllBytesAsync(filePath));
                    fileContent.Headers.ContentType = new MediaTypeHeaderValue("application/pdf");
                    formData.Add(fileContent, fieldName, Path.GetFileName(filePath));
                }
                
                // Add required files
                await AddFileToForm(plankartPath, "plankart");
                await AddFileToForm(bestemmelserPath, "bestemmelser");
                
                // Add optional SOSI file
                if (sosiPath != null)
                {
                    await AddFileToForm(sosiPath, "sosi");
                }

                var response = await _httpClient.PostAsync("/api/check-field-consistency", formData);
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