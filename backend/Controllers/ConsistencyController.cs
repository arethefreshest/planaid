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
using System.ComponentModel.DataAnnotations;

namespace backend.Controllers
{
    [ApiController]
    [Route("api")]
    public class ConsistencyController : ControllerBase
    {
        private readonly PythonIntegrationService _pythonService;
        private readonly ILogger<ConsistencyController> _logger;

        public ConsistencyController(PythonIntegrationService pythonService, ILogger<ConsistencyController> logger)
        {
            _pythonService = pythonService;
            _logger = logger;
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
            [Required] IFormFile plankart,
            [Required] IFormFile bestemmelser,
            IFormFile? sosi = null)
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
                    return Ok(new { result = result });
                }
                finally
                {
                    // Cleanup temp files
                    if (System.IO.File.Exists(plankartPath)) System.IO.File.Delete(plankartPath);
                    if (System.IO.File.Exists(bestemmelserPath)) System.IO.File.Delete(bestemmelserPath);
                    if (sosiPath != null && System.IO.File.Exists(sosiPath)) System.IO.File.Delete(sosiPath);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing consistency check");
                return StatusCode(500, new { detail = ex.Message });
            }
        }
    }
} 