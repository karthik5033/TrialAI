import { fetchWithRetry } from '../fetchWithRetry';

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { dataset, sensitiveAttributes, metric } = body;

    if (!dataset || !sensitiveAttributes || !metric) {
      return new Response(JSON.stringify({ error: "Missing required fields" }), {
        status: 400,
        headers: { "Content-Type": "application/json" }
      });
    }

    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) {
      return new Response(JSON.stringify({ error: "OPENROUTER_API_KEY is not configured" }), {
        status: 500,
        headers: { "Content-Type": "application/json" }
      });
    }

    const systemPrompt = "You are the Defense in an AI bias trial. You are measured, analytical, and calm. Your job is to argue the accuracy-fairness tradeoff and present business justification for the model. You speak in formal courtroom language. Keep responses to 3-4 sentences maximum.";
    const userMessage = `Respond to the prosecution claims about ${dataset} dataset and ${metric} violations. Defend the model using accuracy and business necessity arguments.`;

    const openRouterResponse = await fetchWithRetry("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "anthropic/claude-3-haiku",
        stream: true,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userMessage }
        ]
      })
    });

    if (!openRouterResponse.ok) {
      const errorText = await openRouterResponse.text();
      throw new Error(`OpenRouter API Error: ${errorText}`);
    }

    return new Response(openRouterResponse.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
      }
    });

  } catch (error: any) {
    console.error("Defense Agent Route Error:", error);
    return new Response(JSON.stringify({ error: error.message || "Internal Server Error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
