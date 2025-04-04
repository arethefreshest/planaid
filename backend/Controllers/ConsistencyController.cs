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
using System.Text.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using PlanAid.Services;

namespace backend.Controllers
{
    [ApiController]
    [Route("api")]
    public class ConsistencyController : ControllerBase
    {
        private readonly ILogger<ConsistencyController> _logger;
        private readonly IPythonIntegrationService _pythonService;
        private readonly PlanAid.Services.MetricsService _metricsService;

        public ConsistencyController(
            ILogger<ConsistencyController> logger,
            IPythonIntegrationService pythonService,
            PlanAid.Services.MetricsService metricsService)
        {
            _logger = logger;
            _pythonService = pythonService;
            _metricsService = metricsService;
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
            var metrics = _metricsService.CreateMetrics("backend_consistency");
            
            try
            {
                _logger.LogInformation($"Received files: {plankart.FileName}, {bestemmelser.FileName}");
                _logger.LogInformation($"Content types: {plankart.ContentType}, {bestemmelser.ContentType}");
                
                if (plankart == null || bestemmelser == null)
                {
                    return BadRequest("Missing required files");
                }

                // Add document info
                _metricsService.AddDocumentInfo(metrics, "plankart", plankart.FileName, plankart.Length);
                _metricsService.AddDocumentInfo(metrics, "bestemmelser", bestemmelser.FileName, bestemmelser.Length);
                if (sosi != null)
                {
                    _metricsService.AddDocumentInfo(metrics, "sosi", sosi.FileName, sosi.Length);
                }

                // Save files to temporary location
                var tempDir = Path.Combine(Path.GetTempPath(), "PlanAid", metrics.RunId);
                Directory.CreateDirectory(tempDir);
                
                var plankartPath = Path.Combine(tempDir, plankart.FileName);
                var bestemmelserPath = Path.Combine(tempDir, bestemmelser.FileName);
                string? sosiPath = null;
                
                var fileStart = DateTime.UtcNow;
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
                    sosiPath = Path.Combine(tempDir, sosi.FileName);
                    using (var stream = new FileStream(sosiPath, FileMode.Create))
                    {
                        await sosi.CopyToAsync(stream);
                    }
                }
                var fileEnd = DateTime.UtcNow;
                _metricsService.RecordTiming(metrics, "file_saving", (fileEnd - fileStart).TotalSeconds);
                
                // Call Python service with timing
                var pythonCallStart = DateTime.UtcNow;
                var result = await _pythonService.CheckConsistencyAsync(plankartPath, bestemmelserPath, sosiPath);
                var pythonCallEnd = DateTime.UtcNow;
                
                _metricsService.RecordTiming(metrics, "python_service_call", (pythonCallEnd - pythonCallStart).TotalSeconds);
                
                // Parse result
                var parseStart = DateTime.UtcNow;
                var consistencyResult = JsonSerializer.Deserialize<ConsistencyResult>(result);
                var parseEnd = DateTime.UtcNow;
                
                _metricsService.RecordTiming(metrics, "result_parsing", (parseEnd - parseStart).TotalSeconds);
                
                // Record field counts
                if (consistencyResult != null)
                {
                    _metricsService.RecordFieldCounts(metrics, consistencyResult);
                }
                
                // Record total processing time
                _metricsService.RecordTiming(metrics, "total_processing", (DateTime.UtcNow - DateTime.Parse(metrics.Timestamp)).TotalSeconds);
                
                // Save metrics
                await _metricsService.SaveMetricsAsync("backend_consistency", metrics);
                
                // Clean up temporary files
                Directory.Delete(tempDir, true);
                
                return Ok(consistencyResult);
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "Service communication error");
                return StatusCode(502, new { detail = ex.Message });
            }
            catch (Exception ex)
            {
                metrics.Error = ex.ToString();
                await _metricsService.SaveMetricsAsync("backend_consistency_error", metrics);
                _logger.LogError(ex, "Error checking consistency");
                return StatusCode(500, new { error = "Error checking consistency" });
            }
        }
    }
} 