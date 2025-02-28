using backend;
using backend.Services;
using Microsoft.OpenApi.Models;
using Microsoft.Extensions.Logging;
using System.Net.Http.Headers;

var builder = WebApplication.CreateBuilder(args);

// Add logging
var logger = LoggerFactory.Create(config =>
{
    config.AddConsole();
}).CreateLogger("Program");

// Add services to the container.
builder.Services.AddControllers();
builder.Services.AddScoped<PdfProcessingService>();
builder.Services.AddScoped<IPythonIntegrationService, PythonIntegrationService>();

// Configure HttpClient and register PythonIntegrationService
builder.Services.AddHttpClient("PythonService", client =>
{
    var pythonServiceUrl = builder.Configuration.GetValue<string>("PythonServiceUrl") ?? "http://python_service:8000";
    logger.LogInformation($"Configuring Python service URL: {pythonServiceUrl}");
    client.BaseAddress = new Uri(pythonServiceUrl.TrimEnd('/') + "/");
    
    // Add default headers
    client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    client.Timeout = TimeSpan.FromMinutes(5); // Increased timeout
});

builder.Services.AddHttpClient("NerService", client =>
{
    var nerServiceUrl = builder.Configuration.GetValue<string>("NerServiceUrl") ?? "http://ner_service:8001";
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
        policy.WithOrigins("http://localhost:3000")
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
app.UseCors(options =>
{
    options.WithOrigins(
        "http://localhost:3000",
        "http://frontend:3000",
        "http://localhost:8000",
        "http://python_service:8000"
    )
    .AllowAnyHeader()
    .AllowAnyMethod()
    .AllowCredentials();
});

app.UseRouting();
app.UseAuthorization();

app.MapControllers();

app.Run();
