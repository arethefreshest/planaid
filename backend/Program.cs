using backend;
using backend.Services;
using Microsoft.OpenApi.Models;
using Microsoft.Extensions.Logging;

var builder = WebApplication.CreateBuilder(args);

// Add logging
var logger = LoggerFactory.Create(config =>
{
    config.AddConsole();
}).CreateLogger("Program");

// Add services to the container.
builder.Services.AddControllers();
builder.Services.AddSingleton<PdfProcessingService>();

// Configure HttpClient and register PythonIntegrationService
builder.Services.AddHttpClient<IPythonIntegrationService, PythonIntegrationService>(client =>
{
    var pythonServiceUrl = builder.Configuration.GetValue<string>("PythonServiceUrl") ?? "http://python_service:8000";
    logger.LogInformation($"Configuring Python service URL: {pythonServiceUrl}");
    client.BaseAddress = new Uri(pythonServiceUrl.TrimEnd('/') + "/");
    
    // Add default headers
    client.DefaultRequestHeaders.Accept.Add(new System.Net.Http.Headers.MediaTypeWithQualityHeaderValue("application/json"));
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
        "http://frontend:3000"
    )
    .AllowAnyHeader()
    .AllowAnyMethod()
    .AllowCredentials();
});

app.UseRouting();
app.UseAuthorization();

app.MapControllers();

app.Run();
