/** @type {import('next').NextConfig} */
const nextConfig = {
  async redirects() {
    return [
      {
        source: '/upload',
        destination: '/trial/upload',
        permanent: true,
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: '/api/remediation/:path*',
        destination: 'http://localhost:8000/api/remediation/:path*',
      },
      {
        source: '/api/full-analysis',
        destination: 'http://localhost:8000/api/analysis/full-analysis',
      },
    ];
  },
};

export default nextConfig;
