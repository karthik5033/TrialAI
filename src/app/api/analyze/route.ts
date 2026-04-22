export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { dataset_name } = body;

    if (!dataset_name) {
      return new Response(JSON.stringify({ error: "Missing required field: dataset_name" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const fastApiUrl = process.env.FASTAPI_URL || "http://localhost:8000";
    const fastApiResponse = await fetch(`${fastApiUrl}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_name }),
    });

    if (!fastApiResponse.ok) {
      const errorText = await fastApiResponse.text();
      throw new Error(`FastAPI /analyze error: ${errorText}`);
    }

    const data = await fastApiResponse.json();

    return new Response(JSON.stringify(data), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });

  } catch (error: any) {
    console.error("Analyze Proxy Error:", error);
    return new Response(JSON.stringify({ error: error.message || "Internal Server Error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
