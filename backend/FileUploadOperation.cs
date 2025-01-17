using Microsoft.OpenApi.Models;
using Swashbuckle.AspNetCore.SwaggerGen;
using Microsoft.AspNetCore.Http;

namespace backend;

public class FileUploadOperation : IOperationFilter
{
    public void Apply(OpenApiOperation operation, OperationFilterContext context)
    {
        var fileUploadMime = "multipart/form-data";
        if (operation.RequestBody == null || !operation.RequestBody.Content.Any(x => x.Key.Equals(fileUploadMime, StringComparison.InvariantCultureIgnoreCase)))
            return;

        var fileParams = context.MethodInfo.GetParameters()
            .Where(p => p.ParameterType == typeof(IFormFile));
            
        operation.RequestBody.Content[fileUploadMime].Schema.Properties =
            fileParams.ToDictionary(
                k => k.Name ?? throw new InvalidOperationException("Parameter name cannot be null"),
                v => new OpenApiSchema()
                {
                    Type = "string",
                    Format = "binary"
                });
    }
} 