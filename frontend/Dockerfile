# Base stage
FROM node:14 AS base

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Install React and its type definitions
RUN npm install react react-dom
RUN npm install --save-dev @types/react @types/react-dom @types/node

# Copy the rest of your application code
COPY . .

# Development stage
FROM base AS development

# Expose the port your app runs on
EXPOSE 3000

# Command to run your app
CMD ["npm", "start"]