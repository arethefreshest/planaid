using backend;
using backend.Services;
using Microsoft.OpenApi.Models;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllers();
builder.Services.AddSingleton<PdfProcessingService>();

// Add HttpClient for Python service
builder.Services.AddHttpClient<PythonIntegrationService>(client =>
{
    client.BaseAddress = new Uri("http://python_service:8000");
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

app.UseHttpsRedirection();
app.UseCors(options =>
{
    options.WithOrigins("http://localhost:3000")
           .AllowAnyHeader()
           .AllowAnyMethod();
});

app.UseRouting();
app.UseAuthorization();

app.MapControllers();

// Allow frontend to access backend
app.UseCors("AllowLocalhost");

app.Run();
