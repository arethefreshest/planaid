using System.IO;
using backend.Services;
using Xunit;
using Microsoft.AspNetCore.Http;
using Moq;
using iText.Kernel.Pdf;
using iText.Layout;
using iText.Layout.Element;

namespace backend.Tests;

public class PdfProcessingServiceTests
{
    private byte[] CreateTestPdf(string text)
    {
        using var memoryStream = new MemoryStream();
        var writer = new PdfWriter(memoryStream);
        var pdf = new PdfDocument(writer);
        var document = new Document(pdf);
        
        document.Add(new Paragraph(text));
        document.Flush();
        document.Close();
        
        return memoryStream.ToArray();
    }

    [Fact]
    public void ExtractTextFromPdf_ReturnsText()
    {
        // Arrange
        var service = new PdfProcessingService();
        var expectedText = "Hello, this is a test PDF document.";
        var pdfBytes = CreateTestPdf(expectedText);

        var fileMock = new Mock<IFormFile>();
        fileMock.Setup(f => f.Length).Returns(pdfBytes.Length);
        fileMock.Setup(f => f.OpenReadStream())
            .Returns(new MemoryStream(pdfBytes));

        // Act
        var result = service.ExtractTextFromPdf(fileMock.Object);

        // Assert
        Assert.NotNull(result);
        Assert.Contains(expectedText, result);
    }
}
