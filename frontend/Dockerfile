# Build stage
FROM node:20-alpine AS base
WORKDIR /app

# Copy package files and install dependencies in a separate layer
COPY package*.json ./
# Use regular npm install for development since we're mounting the source
RUN npm install

# Development stage
FROM base AS development
ENV NODE_ENV=development
# Don't copy source here - it's mounted as a volume
CMD ["npm", "start"]

# Production stage
FROM base AS production
COPY . .
RUN npm run build
RUN npm install -g serve
CMD ["serve", "-s", "build", "-l", "3000"] 