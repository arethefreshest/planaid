# Build stage
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy csproj and restore dependencies
COPY ["PlanAid.csproj", "./"]
RUN dotnet restore

# Copy everything else and build
COPY . .

# Development stage
FROM build AS development
ENTRYPOINT ["dotnet", "watch", "run", "--urls", "http://+:5251"]

# Production stage
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS production
WORKDIR /app
COPY --from=build /src/bin/Debug/net8.0 .
ENTRYPOINT ["dotnet", "PlanAid.dll"]

CMD ["dotnet", "backend.dll", "--no-hot-reload"]