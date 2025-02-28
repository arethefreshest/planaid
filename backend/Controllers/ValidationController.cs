/*
Validation Controller

This controller handles document validation requests, processing uploaded PDF
files and extracting relevant information based on document type.

Features:
- PDF file upload handling
- Document type-specific processing
- Temporary file management
- Error handling and logging
*/

using Microsoft.AspNetCore.Mvc;
using backend.Services;
using System.ComponentModel.DataAnnotations;
using Microsoft.Extensions.Logging;

namespace backend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ValidationController : ControllerBase
    {
        private readonly PdfProcessingService _pdfService;
        private readonly ILogger<ValidationController> _logger;

        public ValidationController(PdfProcessingService pdfService, ILogger<ValidationController> logger)
        {
            _pdfService = pdfService;
            _logger = logger;
        }

        /// <summary>
        /// Processes an uploaded PDF file based on its type.
        /// </summary>
        /// <param name="file">The PDF file to process</param>
        /// <param name="type">Type of document (Regulations/PlanMap)</param>
        /// <returns>Processing results</returns>
        [HttpPost("upload")]
        [Consumes("multipart/form-data")]
        public async Task<IActionResult> UploadPdf([Required] IFormFile file, [FromQuery] PdfType type = PdfType.Regulations)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest("No file uploaded");
            }

            try
            {
                // Use the original filename without extension
                var originalFileName = Path.GetFileNameWithoutExtension(file.FileName);
                var tempFile = Path.Combine(Path.GetTempPath(), $"{originalFileName}_{Guid.NewGuid()}.pdf");
                
                try
                {
                    using (var stream = new FileStream(tempFile, FileMode.Create))
                    {
                        await file.CopyToAsync(stream);
                    }

                    var result = await _pdfService.ProcessPdfAsync(tempFile, type);
                    return Ok(result);
                }
                finally
                {
                    // Clean up the temp file
                    if (System.IO.File.Exists(tempFile))
                    {
                        System.IO.File.Delete(tempFile);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing uploaded PDF");
                return StatusCode(500, "Error processing PDF");
            }
        }
    }
}
