# Stage 1: Build the Vite app
FROM node:18 AS builder

WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy source code and build the app
COPY . .
RUN npm run build

# Stage 2: Serve the Vite app
FROM node:18 AS production

WORKDIR /app

# Install 'serve' to serve static files
RUN npm install -g serve

# Copy the Vite build output from the builder stage
COPY --from=builder /app/dist ./dist

# Expose port 3000 and serve the app
EXPOSE 3000
CMD ["serve", "-s", "dist", "-l", "3000"]
