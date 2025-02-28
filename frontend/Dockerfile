# Base stage
FROM node:20-alpine AS base
WORKDIR /app

# Copy only package.json and package-lock.json to leverage caching
COPY package*.json ./

# Install dependencies separately to avoid volume override issues
RUN npm install

# Development stage
FROM base AS development
ENV NODE_ENV=development
CMD ["npm", "start"]

# Production stage
FROM base AS production
COPY . .  
RUN npm run build
RUN npm install -g serve
CMD ["serve", "-s", "build", "-l", "3000"]
