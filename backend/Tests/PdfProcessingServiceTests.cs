/*
PDF Processing Service Tests

This test suite verifies the functionality of the PDF processing service,
including text extraction, field identification, and error handling.

Test Coverage:
- PDF text extraction
- Field identifier extraction
- Multi-page processing
- Error handling
- Document persistence
*/

using System.IO;
using backend.Services;
using backend.Models;
using Xunit;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Moq;
using iText.Kernel.Pdf;
using iText.Layout;
using iText.Layout.Element;
using System.Text.Json;

namespace backend.Tests;

public class PdfProcessingServiceTests
{
    private readonly Mock<ILogger<PdfProcessingService>> _loggerMock;
    private readonly Mock<IPythonIntegrationService> _pythonServiceMock;
    private readonly PdfProcessingService _service;

    public PdfProcessingServiceTests()
    {
        _loggerMock = new Mock<ILogger<PdfProcessingService>>();
        _pythonServiceMock = new Mock<IPythonIntegrationService>();
        
        // Setup default mock behavior
        _pythonServiceMock
            .Setup(x => x.CheckConsistencyAsync(
                It.IsAny<string>(),
                It.IsAny<string>(),
                It.IsAny<string>()))
            .ReturnsAsync("{}");

        _service = new PdfProcessingService(
            _loggerMock.Object,
            _pythonServiceMock.Object);
    }

    /// <summary>
    /// Creates a test PDF with specified content.
    /// </summary>
    private byte[] CreateTestPdf(string content)
    {
        using var memoryStream = new MemoryStream();
        using var writer = new PdfWriter(memoryStream);
        using var pdf = new PdfDocument(writer);
        using var document = new Document(pdf);
        
        // Add test content
        document.Add(new Paragraph(content));
        
        // Add a second page
        document.Add(new AreaBreak());
        document.Add(new Paragraph("Page 2 content"));
        
        document.Close();
        return memoryStream.ToArray();
    }

    /// <summary>
    /// Tests text extraction from regulations PDF.
    /// </summary>
    [Fact]
    public async Task ProcessPdfAsync_WithRegulationsType_ShouldExtractText()
    {
        // Arrange
        var testPdfPath = "path/to/test.pdf";

        // Act
        var result = await _service.ProcessPdfAsync(testPdfPath, PdfType.Regulations);

        // Assert
        // ... your assertions here ...
    }

    [Fact]
    public async Task ProcessPdfAsync_WithPlanMapType_ShouldExtractFieldIdentifiers()
    {
        // Arrange
        var testFile = Path.GetTempFileName();
        var pdfBytes = CreateTestPdf("Test content with field BRA_1");
        await File.WriteAllBytesAsync(testFile, pdfBytes);
        
        try
        {
            // Act
            var result = await _service.ProcessPdfAsync(testFile, PdfType.PlanMap);

            // Assert
            Assert.NotNull(result);
            Assert.Contains("BRA_1", result.ToString());
        }
        finally
        {
            if (File.Exists(testFile))
            {
                File.Delete(testFile);
            }
        }
    }

    [Fact]
    public async Task ProcessPdfAsync_ExtractsAllPages()
    {
        // Arrange
        var testFile = Path.GetTempFileName();
        var pdfBytes = CreateTestPdf("Test content");
        await File.WriteAllBytesAsync(testFile, pdfBytes);
        
        try
        {
            // Act
            var resultJson = await _service.ProcessPdfAsync(testFile, PdfType.Regulations);
            var result = JsonSerializer.Deserialize<ProcessedPdfDocument>(resultJson);

            // Assert
            Assert.NotNull(result);
            Assert.True(result!.TotalPages > 0);
            Assert.NotEmpty(result.Pages);
        }
        finally
        {
            if (File.Exists(testFile))
            {
                File.Delete(testFile);
            }
        }
    }

    [Fact]
    public async Task ProcessPdfAsync_HandlesError()
    {
        // Arrange
        var nonExistentFile = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() => 
            _service.ProcessPdfAsync(nonExistentFile, PdfType.Regulations));
    }

    [Fact]
    public async Task ProcessPdfAsync_ValidFile_ReturnsProcessedDocument()
    {
        // Arrange
        var filePath = "test.pdf";
        var type = PdfType.Consistency;

        // Act
        var result = await _service.ProcessPdfAsync(filePath, type);

        // Assert
        Assert.NotNull(result);
    }
}
