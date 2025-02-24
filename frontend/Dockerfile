# Base stage
FROM node:14 AS base

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of your application code
COPY . .

# Build the application
RUN npm run build

# Development stage
FROM base AS development

# Expose the port your app runs on
EXPOSE 3000

# Command to run your app
CMD ["npm", "start"]