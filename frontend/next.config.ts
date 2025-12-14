/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  // Enable if you need WebSocket support
  experimental: {
    serverActions: true,
  },
}

module.exports = nextConfig
