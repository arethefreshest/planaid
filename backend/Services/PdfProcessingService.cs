using System.IO;
using iText.Kernel.Pdf;
using iText.Kernel.Pdf.Canvas.Parser;

namespace backend.Services
{
    public class PdfProcessingService
    {
        public string ExtractTextFromPdf(IFormFile file)
        {
            using var stream = file.OpenReadStream();
            using var pdfReader = new PdfReader(stream);
            using var document = new PdfDocument(pdfReader);

            string text = string.Empty;
            for (int i = 1; i <= document.GetNumberOfPages(); i++)
            {
                text += PdfTextExtractor.GetTextFromPage(document.GetPage(i));
            }

            return text;
        }
    }
}
