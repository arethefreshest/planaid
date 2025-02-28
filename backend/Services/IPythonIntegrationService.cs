using Microsoft.AspNetCore.Http;

namespace backend.Services
{
    public interface IPythonIntegrationService
    {
        /// <summary>
        /// Checks consistency between regulatory documents using Python service.
        /// </summary>
        /// <param name="plankartPath">Path to plankart PDF file</param>
        /// <param name="bestemmelserPath">Path to bestemmelser PDF file</param>
        /// <param name="sosiPath">Optional path to SOSI file</param>
        /// <returns>JSON string containing consistency check results</returns>
        Task<string> CheckConsistencyAsync(string plankartPath, string bestemmelserPath, string? sosiPath = null);
    }
} 