using backend;
using backend.Services;
using backend.Models;
using backend.Extensions;
using Microsoft.OpenApi.Models;
using Microsoft.Extensions.Logging;
using System.Net.Http.Headers;
using Microsoft.AspNetCore.OpenApi;
using PlanAid.Services;

var builder = WebApplication.CreateBuilder(args);

// Add logging
var logger = LoggerFactory.Create(config =>
{
    config.AddConsole();
    var fileOptions = builder.Configuration.GetSection("Logging:File").Get<FileLoggerOptions>();
    if (fileOptions != null)
    {
        config.AddFile(fileOptions);
    }
}).CreateLogger("Program");

// Add services to the container.
builder.Services.AddControllers();
builder.Services.AddScoped<PdfProcessingService>();
builder.Services.AddScoped<IPythonIntegrationService, PythonIntegrationService>();
builder.Services.AddScoped<PlanAid.Services.MetricsService>();

// Configure HttpClient and register PythonIntegrationService
builder.Services.AddHttpClient("PythonService", client =>
{
    var isDocker = Environment.GetEnvironmentVariable("DOTNET_RUNNING_IN_CONTAINER") == "true";
    var serviceUrls = builder.Configuration.GetSection("ServiceUrls").Get<ServiceUrls>();
    var pythonServiceUrl = isDocker 
        ? (serviceUrls?.Docker?.PythonServiceUrl ?? "http://python_service:8000")
        : (serviceUrls?.Local?.PythonServiceUrl ?? "http://localhost:8000");
    
    logger.LogInformation($"Configuring Python service URL: {pythonServiceUrl}");
    client.BaseAddress = new Uri(pythonServiceUrl.TrimEnd('/') + "/");
    
    // Add default headers
    client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    client.Timeout = TimeSpan.FromMinutes(5); // Increased timeout
});

builder.Services.AddHttpClient("NerService", client =>
{
    var isDocker = Environment.GetEnvironmentVariable("DOTNET_RUNNING_IN_CONTAINER") == "true";
    var serviceUrls = builder.Configuration.GetSection("ServiceUrls").Get<ServiceUrls>();
    var nerServiceUrl = isDocker 
        ? (serviceUrls?.Docker?.NerServiceUrl ?? "http://ner_service:8001")
        : (serviceUrls?.Local?.NerServiceUrl ?? "http://localhost:8001");
    
    logger.LogInformation($"Configuring NER service URL: {nerServiceUrl}");
    client.BaseAddress = new Uri(nerServiceUrl.TrimEnd('/') + "/");
    client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    client.Timeout = TimeSpan.FromMinutes(2); // Increased timeout
});

// Configure Swagger/OpenAPI
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo
    {
        Title = "PlanAid API",
        Version = "v1",
        Description = "API Documentation for PlanAid",
        Contact = new OpenApiContact
        {
            Name = "PlanAid",
            Email = "areb@uia.no",
        }
    });
    
    // Add support for file uploads in Swagger
    c.OperationFilter<FileUploadOperation>();
});

builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowLocalhost", policy =>
    {
        policy.WithOrigins(
            builder.Configuration["FRONTEND_URL"] ?? "http://localhost:3000",
            builder.Configuration["DOCKER_FRONTEND_URL"] ?? "http://frontend:3000"
        )
        .AllowAnyMethod()
        .AllowAnyHeader()
        .AllowCredentials();
    });
});

var app = builder.Build();

// Add the redirects first, before other middleware
if (app.Environment.IsDevelopment())
{
    app.Use(async (context, next) =>
    {
        if (context.Request.Path.Value == "/" || context.Request.Path.Value == "/index.html")
        {
            context.Response.Redirect("/swagger");
            return;
        }
        await next();
    });
    
    app.UseSwagger();
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "PlanAid API v1");
        c.RoutePrefix = "swagger";
    });
}

// app.UseHttpsRedirection(); Commented out for development purposes
app.UseCors("AllowLocalhost");

app.UseRouting();
app.UseAuthorization();

app.MapControllers();

app.Run();
