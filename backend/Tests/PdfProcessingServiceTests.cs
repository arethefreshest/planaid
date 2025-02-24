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
using System.Collections.Generic;

namespace backend.Tests;

public class PdfProcessingServiceTests : IDisposable
{
    private readonly Mock<ILogger<PdfProcessingService>> _loggerMock;
    private readonly Mock<IPythonIntegrationService> _pythonServiceMock;
    private readonly PdfProcessingService _service;
    private readonly string _testDirectory;

    public PdfProcessingServiceTests()
    {
        _loggerMock = new Mock<ILogger<PdfProcessingService>>();
        _pythonServiceMock = new Mock<IPythonIntegrationService>();
        _service = new PdfProcessingService(_pythonServiceMock.Object, _loggerMock.Object);
        
        _testDirectory = Path.Combine(Path.GetTempPath(), "PdfProcessingTests_" + Guid.NewGuid());
        Directory.CreateDirectory(_testDirectory);
        Console.WriteLine($"Created test directory: {_testDirectory}");
    }

    private string CreateTestPdf(string content)
    {
        var filePath = Path.Combine(_testDirectory, $"{Guid.NewGuid()}.pdf");
        Console.WriteLine($"Creating PDF at: {filePath}");
        Console.WriteLine($"Content: {content}");
        
        try 
        {
            using var fs = new FileStream(filePath, FileMode.Create);
            using var writer = new PdfWriter(fs);
            using var pdf = new PdfDocument(writer);
            using var doc = new Document(pdf);
            doc.Add(new Paragraph(content));
            doc.Close();
            
            Console.WriteLine($"PDF created successfully at: {filePath}");
            Console.WriteLine($"File exists: {File.Exists(filePath)}");
            Console.WriteLine($"File size: {new FileInfo(filePath).Length} bytes");
            
            return filePath;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error creating PDF: {ex}");
            throw;
        }
    }

    /// <summary>
    /// Tests text extraction from regulations PDF.
    /// </summary>
    [Fact]
    public async Task ProcessPdfAsync_WithRegulationsType_ShouldExtractText()
    {
        Console.WriteLine("Starting ProcessPdfAsync_WithRegulationsType_ShouldExtractText test");
        
        // Arrange
        var content = "Test content for regulations";
        var testFile = CreateTestPdf(content);
        Console.WriteLine($"Test file created at: {testFile}");
        
        try
        {
            // Act
            var resultJson = await _service.ProcessPdfAsync(testFile, PdfType.Regulations);
            Console.WriteLine($"Result JSON: {resultJson}");
            
            var result = JsonSerializer.Deserialize<ProcessedPdfDocument>(resultJson);
            Console.WriteLine($"Deserialized result: {JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true })}");

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result!.Pages);
            Assert.NotEmpty(result.Pages);
        }
        finally
        {
            SafeDeleteFile(testFile);
        }
    }

    [Fact]
    public async Task ProcessPdfAsync_WithPlanMapType_ShouldExtractFieldIdentifiers()
    {
        // Arrange
        var content = @"Tegnforklaring
BRA1 - Boligbebyggelse
o_BRA_1 - Offentlig boligbebyggelse
f_BRA_2 - Felles boligbebyggelse
Kartopplysninger";
        var testFile = CreateTestPdf(content);
        
        _pythonServiceMock.Setup(x => x.CheckConsistencyAsync(
            It.IsAny<string>(), 
            It.IsAny<string>(), 
            It.IsAny<string>()))
            .ReturnsAsync("{}");
        
        try
        {
            // Act
            var resultJson = await _service.ProcessPdfAsync(testFile, PdfType.PlanMap);
            var result = JsonSerializer.Deserialize<ProcessedPdfDocument>(resultJson);
            
            // Assert
            Assert.NotNull(result);
        }
        finally
        {
            SafeDeleteFile(testFile);
        }
    }

    [Fact]
    public async Task ProcessPdfAsync_ExtractsAllPages()
    {
        // Arrange
        var content = "Test content With multiple lines And more content";
        var testFile = CreateTestPdf(content);
        
        try
        {
            // Act
            var resultJson = await _service.ProcessPdfAsync(testFile, PdfType.Regulations);
            _loggerMock.VerifyNoErrors();
            
            var result = JsonSerializer.Deserialize<ProcessedPdfDocument>(resultJson);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result!.Pages);
            Assert.True(result.Pages.Count > 0, "Document should have at least one page");
            Assert.True(result.TotalPages > 0, "Total pages should be greater than 0");
            
            // Normalize strings for comparison
            var normalizedContent = content.Replace("\n", " ").Replace("  ", " ").Trim();
            var normalizedResult = result.Pages[0].Content.Replace("\n", " ").Replace("  ", " ").Trim();
            Assert.Contains(normalizedContent, normalizedResult);
        }
        finally
        {
            SafeDeleteFile(testFile);
        }
    }

    [Fact]
    public async Task ProcessPdfAsync_HandlesError()
    {
        Console.WriteLine("Starting ProcessPdfAsync_HandlesError test");
        var nonExistentFile = Path.Combine(_testDirectory, "nonexistent.pdf");
        Console.WriteLine($"Using non-existent file path: {nonExistentFile}");

        var exception = await Assert.ThrowsAsync<FileNotFoundException>(() => 
            _service.ProcessPdfAsync(nonExistentFile, PdfType.Regulations));
        
        Console.WriteLine($"Exception message: {exception.Message}");
    }

    [Fact]
    public async Task CheckConsistencyAsync_ValidPdf_ReturnsProcessedDocument()
    {
        // Test setup
        var testFilePath = CreateTestPdf("Test content");
        
        _pythonServiceMock.Setup(x => x.CheckConsistencyAsync(
            It.IsAny<string>(), 
            It.IsAny<string>(), 
            It.IsAny<string>()))
            .ReturnsAsync("{}");
        
        try
        {
            // Test execution
            var result = await _service.ProcessPdfAsync(testFilePath, PdfType.Regulations);
            
            // Assertions
            Assert.NotNull(result);
        }
        finally
        {
            SafeDeleteFile(testFilePath);
        }
    }

    private void SafeDeleteFile(string filePath)
    {
        try
        {
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
                Console.WriteLine($"Successfully deleted file: {filePath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to delete file {filePath}: {ex}");
        }
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_testDirectory))
            {
                Directory.Delete(_testDirectory, true);
                Console.WriteLine($"Successfully deleted test directory: {_testDirectory}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to delete test directory {_testDirectory}: {ex}");
        }
    }
}

// Helper extension method for verifying no errors were logged
public static class LoggerMockExtensions
{
    public static void VerifyNoErrors<T>(this Mock<ILogger<T>> loggerMock)
    {
        loggerMock.Verify(
            x => x.Log(
                It.Is<LogLevel>(l => l == LogLevel.Error),
                It.IsAny<EventId>(),
                It.IsAny<It.IsAnyType>(),
                It.IsAny<Exception>(),
                It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
            Times.Never);
    }
}
