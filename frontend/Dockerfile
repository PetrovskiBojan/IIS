# Step 1: Build the React application
FROM node:16-alpine as builder

WORKDIR /app

# Copy package.json and package-lock.json files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install --production

COPY . .

RUN npm run build

# Step 2: Serve the app using http-server
FROM node:16-alpine

RUN npm install -g http-server

WORKDIR /app

COPY --from=builder /app/build /app

EXPOSE 3000

CMD ["http-server", ".", "-p", "3000"]
