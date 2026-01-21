FROM node:20-slim

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm install

# Copy source and types
COPY src ./src
COPY types ./types

# Build
RUN npm run build

# Start
CMD ["npm", "start"]
