export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { dataset_name, technique } = body;

    if (!dataset_name || !technique) {
      return new Response(JSON.stringify({ error: "Missing required fields: dataset_name, technique" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const fastApiUrl = process.env.FASTAPI_URL || "http://localhost:8000";
    const fastApiResponse = await fetch(`${fastApiUrl}/mitigate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_name, technique }),
    });

    if (!fastApiResponse.ok) {
      const errorText = await fastApiResponse.text();
      throw new Error(`FastAPI /mitigate error: ${errorText}`);
    }

    const data = await fastApiResponse.json();

    return new Response(JSON.stringify(data), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });

  } catch (error: any) {
    console.error("Mitigate Proxy Error:", error);
    return new Response(JSON.stringify({ error: error.message || "Internal Server Error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
