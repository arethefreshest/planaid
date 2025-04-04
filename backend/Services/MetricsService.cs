using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using backend.Models;
using Microsoft.Extensions.Hosting;

namespace PlanAid.Services
{
    public class MetricsService
    {
        private readonly ILogger<MetricsService> _logger;
        private readonly string _metricsDirectory;
        
        public MetricsService(ILogger<MetricsService> logger, IHostEnvironment env)
        {
            _logger = logger;
            _metricsDirectory = Path.Combine(env.ContentRootPath, "metrics");
            
            // Ensure metrics directory exists
            try
            {
                Directory.CreateDirectory(_metricsDirectory);
                _logger.LogInformation($"Metrics directory created/verified at: {_metricsDirectory}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Failed to create metrics directory at: {_metricsDirectory}");
                throw;
            }
        }
        
        public MetricsData CreateMetrics(string operationName)
        {
            return new MetricsData
            {
                Operation = operationName,
                RunId = Guid.NewGuid().ToString(),
                Timestamp = DateTime.UtcNow.ToString("o")
            };
        }

        public void AddDocumentInfo(MetricsData metrics, string docType, string filename, long size)
        {
            metrics.DocumentInfo.Documents[docType] = new DocumentDetails
            {
                Filename = filename,
                Size = size
            };
        }

        public void RecordTiming(MetricsData metrics, string stepName, double duration)
        {
            metrics.Timings[stepName] = duration;
        }

        public void RecordFieldCounts(MetricsData metrics, ConsistencyResult result)
        {
            metrics.FieldCounts = new Dictionary<string, int>
            {
                ["matching"] = result?.MatchingFields?.Count ?? 0,
                ["only_in_plankart"] = result?.OnlyInPlankart?.Count ?? 0,
                ["only_in_bestemmelser"] = result?.OnlyInBestemmelser?.Count ?? 0,
                ["only_in_sosi"] = result?.OnlyInSosi?.Count ?? 0,
                ["is_consistent"] = result?.IsConsistent == true ? 1 : 0
            };
        }
        
        public async Task SaveMetricsAsync(string operationName, MetricsData metrics)
        {
            try
            {
                var metricsFile = Path.Combine(_metricsDirectory, $"{operationName}_{metrics.RunId}.json");
                
                // Ensure the directory exists again (in case it was deleted)
                Directory.CreateDirectory(_metricsDirectory);
                
                await File.WriteAllTextAsync(
                    metricsFile, 
                    JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true })
                );
                
                _logger.LogInformation($"Metrics saved to {metricsFile}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error saving metrics to {_metricsDirectory}");
                throw;
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
                
                // Ensure the directory exists again (in case it was deleted)
                Directory.CreateDirectory(_metricsDirectory);
                
                await File.WriteAllTextAsync(
                    metricsFile, 
                    JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true })
                );
                
                _logger.LogInformation($"Error metrics saved to {metricsFile}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error saving error metrics to {_metricsDirectory}");
                throw;
            }
        }
    }
} 