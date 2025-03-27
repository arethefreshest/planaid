using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace PlanAid.Services
{
    public class MetricsService
    {
        private readonly ILogger<MetricsService> _logger;
        private readonly string _metricsDirectory;
        
        public MetricsService(ILogger<MetricsService> logger)
        {
            _logger = logger;
            _metricsDirectory = Path.Combine(Directory.GetCurrentDirectory(), "metrics");
            
            // Ensure metrics directory exists
            Directory.CreateDirectory(_metricsDirectory);
        }
        
        public async Task SaveMetricsAsync(string operationName, Dictionary<string, object> metrics)
        {
            try
            {
                var runId = metrics.ContainsKey("run_id") 
                    ? metrics["run_id"].ToString() 
                    : Guid.NewGuid().ToString();
                
                var metricsFile = Path.Combine(_metricsDirectory, $"{operationName}_{runId}.json");
                
                await File.WriteAllTextAsync(
                    metricsFile, 
                    JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true })
                );
                
                _logger.LogInformation($"Metrics saved to {metricsFile}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving metrics");
            }
        }
        
        public async Task SaveErrorMetricsAsync(string operationName, Dictionary<string, object> metrics, Exception error)
        {
            try
            {
                var runId = metrics.ContainsKey("run_id") 
                    ? metrics["run_id"].ToString() 
                    : Guid.NewGuid().ToString();
                
                metrics["error"] = error.ToString();
                
                var metricsFile = Path.Combine(_metricsDirectory, $"{operationName}_error_{runId}.json");
                
                await File.WriteAllTextAsync(
                    metricsFile, 
                    JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true })
                );
                
                _logger.LogInformation($"Error metrics saved to {metricsFile}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving error metrics");
            }
        }
    }
} 