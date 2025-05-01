using Microsoft.AspNetCore.Mvc;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using iText.Kernel.Pdf;
using iText.Kernel.Pdf.Canvas.Parser;

[ApiController]
[Route("api/compare-pdfs")]
public class PdfCompareController : ControllerBase
{
    private const string MistralApiKey = "QID7uLWTtHLfHY1JuZO8wHFuENtXFIRA";
    private const string MistralApiUrl = "https://api.mistral.ai/v1/chat/completions";
    private const string Model = "mistral-small";

    [HttpPost]
    public async Task<IActionResult> ComparePdfs(IFormFile file1, IFormFile file2)
    {
        if (file1 == null || file2 == null)
            return BadRequest("Both PDF files are required.");

        string text1 = await ExtractTextFromPdf(file1);
        string text2 = await ExtractTextFromPdf(file2);

        string prompt = $@"Compare the following two documents and highlight the differences clearly.

Document 1:
{text1.Substring(0, Math.Min(10000, text1.Length))}

Document 2:
{text2.Substring(0, Math.Min(10000, text2.Length))}

Please list the differences as bullet points or a summary.";

        var httpClient = new HttpClient();
        httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", MistralApiKey);

        var body = new
        {
            model = Model,
            messages = new[] {
                new { role = "user", content = prompt }
            },
            temperature = 0.3
        };

        var response = await httpClient.PostAsync(
            MistralApiUrl,
            new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json")
        );

        string responseContent = await response.Content.ReadAsStringAsync();
        using var jsonDoc = JsonDocument.Parse(responseContent);
        var result = jsonDoc.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString();

        return Ok(new
        {
            text1,
            text2,
            diff = result
        });
    }

    private async Task<string> ExtractTextFromPdf(IFormFile file)
    {
        using var stream = file.OpenReadStream();
        using var memoryStream = new MemoryStream();
        await stream.CopyToAsync(memoryStream);
        memoryStream.Position = 0;

        var sb = new StringBuilder();

        using var pdfReader = new PdfReader(memoryStream);
        using var pdfDoc = new PdfDocument(pdfReader);

        for (int i = 1; i <= pdfDoc.GetNumberOfPages(); i++)
        {
            var page = pdfDoc.GetPage(i);
            var text = PdfTextExtractor.GetTextFromPage(page);
            sb.AppendLine(text);
        }

        return sb.ToString();
    }
}
