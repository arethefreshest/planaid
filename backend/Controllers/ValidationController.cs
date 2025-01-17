using Microsoft.AspNetCore.Mvc;
using backend.Services;
using System.ComponentModel.DataAnnotations;

namespace backend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ValidationController : ControllerBase
    {
        private readonly PdfProcessingService _pdfService;

        public ValidationController(PdfProcessingService pdfService)
        {
            _pdfService = pdfService;
        }

        [HttpPost("upload")]
        [Consumes("multipart/form-data")]
        public IActionResult UploadPdf([Required] IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest("No file uploaded");
            }

            var text = _pdfService.ExtractTextFromPdf(file);
            return Ok(new { Text = text });
        }
    }
}
