/*
Consistency Controller

This controller handles requests for checking consistency between regulatory
documents. It processes uploaded files and coordinates with the Python service
for analysis.

Features:
- File upload handling
- Consistency checking
- Temporary file management
- Error handling and logging
*/

using Microsoft.AspNetCore.Mvc;
using backend.Services;
using backend.Models;
using System.ComponentModel.DataAnnotations;
using System.Text.Json;

namespace backend.Controllers
{
    [ApiController]
    [Route("api")]
    public class ConsistencyController : ControllerBase
    {
        private readonly ILogger<ConsistencyController> _logger;
        private readonly IPythonIntegrationService _pythonService;

        public ConsistencyController(
            ILogger<ConsistencyController> logger,
            IPythonIntegrationService pythonService)
        {
            _logger = logger;
            _pythonService = pythonService;
        }

        /// <summary>
        /// Checks consistency between regulatory documents.
        /// </summary>
        /// <param name="plankart">Plankart PDF file</param>
        /// <param name="bestemmelser">Bestemmelser PDF file</param>
        /// <param name="sosi">Optional SOSI file</param>
        /// <returns>Consistency check results</returns>
        [HttpPost("check-field-consistency")]
        [Consumes("multipart/form-data")]
        public async Task<IActionResult> CheckFieldConsistency(
            [FromForm] IFormFile plankart,
            [FromForm] IFormFile bestemmelser,
            [FromForm] IFormFile? sosi = null)
        {
            try
            {
                _logger.LogInformation($"Received files: {plankart.FileName}, {bestemmelser.FileName}");
                _logger.LogInformation($"Content types: {plankart.ContentType}, {bestemmelser.ContentType}");
                
                if (plankart == null || bestemmelser == null)
                {
                    return BadRequest("Missing required files");
                }

                // Create temp files with original extensions
                var plankartPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}{Path.GetExtension(plankart.FileName)}");
                var bestemmelserPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}{Path.GetExtension(bestemmelser.FileName)}");
                string? sosiPath = null;

                try
                {
                    _logger.LogInformation($"Writing files to: {plankartPath}, {bestemmelserPath}");
                    
                    using (var stream = new FileStream(plankartPath, FileMode.Create))
                    {
                        await plankart.CopyToAsync(stream);
                    }

                    using (var stream = new FileStream(bestemmelserPath, FileMode.Create))
                    {
                        await bestemmelser.CopyToAsync(stream);
                    }

                    if (sosi != null)
                    {
                        sosiPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}{Path.GetExtension(sosi.FileName)}");
                        using (var stream = new FileStream(sosiPath, FileMode.Create))
                        {
                            await sosi.CopyToAsync(stream);
                        }
                    }

                    var result = await _pythonService.CheckConsistencyAsync(plankartPath, bestemmelserPath, sosiPath);
                    _logger.LogInformation("Python service response: {Result}", result);
                    
                    if (string.IsNullOrEmpty(result))
                    {
                        _logger.LogError("Empty response from processing service");
                        return StatusCode(500, new { detail = "Empty response from processing service" });
                    }

                    try
                    {
                        var pythonResponse = JsonSerializer.Deserialize<PythonResponse<ConsistencyResult>>(result);
                        if (pythonResponse == null)
                        {
                            _logger.LogError("Failed to deserialize response");
                            return StatusCode(500, new { detail = "Failed to deserialize response" });
                        }

                        return Ok(pythonResponse);
                    }
                    catch (JsonException ex)
                    {
                        _logger.LogError(ex, "Error deserializing response: {Result}", result);
                        return StatusCode(500, new { detail = $"Invalid response format: {ex.Message}" });
                    }
                }
                finally
                {
                    // Cleanup temp files
                    if (System.IO.File.Exists(plankartPath)) System.IO.File.Delete(plankartPath);
                    if (System.IO.File.Exists(bestemmelserPath)) System.IO.File.Delete(bestemmelserPath);
                    if (sosiPath != null && System.IO.File.Exists(sosiPath)) System.IO.File.Delete(sosiPath);
                }
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "Service communication error");
                return StatusCode(502, new { detail = ex.Message });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing consistency check");
                return StatusCode(500, new { detail = ex.Message });
            }
        }
    }
} 