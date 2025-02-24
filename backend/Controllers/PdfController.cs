using Microsoft.AspNetCore.Mvc;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace backend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class PdfController : ControllerBase
    {
        private readonly string _processedPdfsPath;
        
        public PdfController(IWebHostEnvironment env)
        {
            _processedPdfsPath = Path.Combine(env.ContentRootPath, "processed_pdfs");
        }

        [HttpGet("list")]
        public ActionResult<IEnumerable<string>> GetProcessedPdfs()
        {
            var files = Directory.GetFiles(_processedPdfsPath, "*.json")
                .Select(Path.GetFileName);
            return Ok(files);
        }

        [HttpGet("{filename}")]
        public ActionResult GetProcessedPdf(string filename)
        {
            var filePath = Path.Combine(_processedPdfsPath, filename);
            if (!System.IO.File.Exists(filePath))
                return NotFound();
            
            var content = System.IO.File.ReadAllText(filePath);
            return Content(content, "application/json");
        }
    }
} 