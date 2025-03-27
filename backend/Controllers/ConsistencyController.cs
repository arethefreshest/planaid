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
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;

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
            // Start metrics collection
            var startTime = DateTime.UtcNow;
            var metrics = new Dictionary<string, object>
            {
                ["run_id"] = Guid.NewGuid().ToString(),
                ["timestamp"] = startTime.ToString("o"),
                ["files"] = new Dictionary<string, object>
                {
                    ["plankart"] = new Dictionary<string, object>
                    {
                        ["filename"] = plankart.FileName,
                        ["size_kb"] = plankart.Length / 1024.0
                    },
                    ["bestemmelser"] = new Dictionary<string, object>
                    {
                        ["filename"] = bestemmelser.FileName,
                        ["size_kb"] = bestemmelser.Length / 1024.0
                    }
                },
                ["timings"] = new Dictionary<string, double>(),
                ["resource_usage"] = new Dictionary<string, double>()
            };
            
            if (sosi != null)
            {
                ((Dictionary<string, object>)metrics["files"])["sosi"] = new Dictionary<string, object>
                {
                    ["filename"] = sosi.FileName,
                    ["size_kb"] = sosi.Length / 1024.0
                };
            }
            
            try
            {
                _logger.LogInformation($"Received files: {plankart.FileName}, {bestemmelser.FileName}");
                _logger.LogInformation($"Content types: {plankart.ContentType}, {bestemmelser.ContentType}");
                
                if (plankart == null || bestemmelser == null)
                {
                    return BadRequest("Missing required files");
                }

                // Save files to temporary location
                var tempDir = Path.Combine(Path.GetTempPath(), "PlanAid", metrics["run_id"]?.ToString() ?? Guid.NewGuid().ToString());
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
                ((Dictionary<string, double>)metrics["timings"])["file_saving"] = (fileEnd - fileStart).TotalSeconds;
                
                // Call Python service with timing
                var pythonCallStart = DateTime.UtcNow;
                var result = await _pythonService.CheckConsistencyAsync(plankartPath, bestemmelserPath, sosiPath);
                var pythonCallEnd = DateTime.UtcNow;
                
                ((Dictionary<string, double>)metrics["timings"])["python_service_call"] = 
                    (pythonCallEnd - pythonCallStart).TotalSeconds;
                
                // Parse result
                var parseStart = DateTime.UtcNow;
                var consistencyResult = JsonSerializer.Deserialize<ConsistencyResult>(result);
                var parseEnd = DateTime.UtcNow;
                
                ((Dictionary<string, double>)metrics["timings"])["result_parsing"] = 
                    (parseEnd - parseStart).TotalSeconds;
                
                // Record field counts
                metrics["field_counts"] = new Dictionary<string, int>
                {
                    ["matching"] = consistencyResult?.MatchingFields?.Count ?? 0,
                    ["only_in_plankart"] = consistencyResult?.OnlyInPlankart?.Count ?? 0,
                    ["only_in_bestemmelser"] = consistencyResult?.OnlyInBestemmelser?.Count ?? 0,
                    ["only_in_sosi"] = consistencyResult?.OnlyInSosi?.Count ?? 0,
                    ["is_consistent"] = consistencyResult?.IsConsistent == true ? 1 : 0
                };
                
                // Record end metrics
                var endTime = DateTime.UtcNow;
                ((Dictionary<string, double>)metrics["timings"])["total_processing"] = 
                    (endTime - startTime).TotalSeconds;
                
                // Save metrics to file
                var metricsDir = Path.Combine(Directory.GetCurrentDirectory(), "metrics");
                Directory.CreateDirectory(metricsDir);
                
                var metricsFile = Path.Combine(metricsDir, $"backend_consistency_{metrics["run_id"]}.json");
                await System.IO.File.WriteAllTextAsync(
                    metricsFile, 
                    JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true })
                );
                
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
                // Record error in metrics
                metrics["error"] = ex.ToString();
                
                // Save metrics even in case of error
                var metricsDir = Path.Combine(Directory.GetCurrentDirectory(), "metrics");
                Directory.CreateDirectory(metricsDir);
                
                var metricsFile = Path.Combine(metricsDir, $"backend_consistency_error_{metrics["run_id"]}.json");
                await System.IO.File.WriteAllTextAsync(
                    metricsFile, 
                    JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true })
                );
                
                _logger.LogError(ex, "Error checking consistency");
                return StatusCode(500, new { error = "Error checking consistency" });
            }
        }
    }
} 