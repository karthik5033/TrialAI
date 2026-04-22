/**
 * Fetch wrapper with retry logic for handling transient network errors
 * like ECONNRESET and TLS handshake failures.
 */
export async function fetchWithRetry(
  url: string,
  options: RequestInit,
  maxRetries: number = 3,
  delayMs: number = 1000
): Promise<Response> {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);
      return response;
    } catch (error: any) {
      lastError = error;
      const isRetryable =
        error?.cause?.code === 'ECONNRESET' ||
        error?.cause?.code === 'SELF_SIGNED_CERT_IN_CHAIN' ||
        error?.cause?.code === 'ETIMEDOUT' ||
        error?.message?.includes('fetch failed');

      if (isRetryable && attempt < maxRetries) {
        console.warn(
          `Fetch attempt ${attempt}/${maxRetries} failed (${error?.cause?.code || error.message}). Retrying in ${delayMs}ms...`
        );
        await new Promise((resolve) => setTimeout(resolve, delayMs * attempt));
      } else {
        throw error;
      }
    }
  }

  throw lastError;
}
