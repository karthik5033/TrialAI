export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { dataset, sensitiveAttributes, metric, prosecutionArgument, defenseArgument } = body;

    if (!dataset || !sensitiveAttributes || !metric || !prosecutionArgument || !defenseArgument) {
      return new Response(JSON.stringify({ error: "Missing required fields" }), {
        status: 400,
        headers: { "Content-Type": "application/json" }
      });
    }

    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
      return new Response(JSON.stringify({ error: "GROQ_API_KEY is not configured" }), {
        status: 500,
        headers: { "Content-Type": "application/json" }
      });
    }

    const systemPrompt = "You are the Judge in an AI bias trial. You are authoritative, neutral, and structured. You apply legal fairness standards including disparate impact doctrine. You cross-examine both sides and deliver structured rulings. Keep responses to 3-4 sentences maximum.";
    const userMessage = `Having heard the Prosecution argue: ${prosecutionArgument}. And the Defense counter: ${defenseArgument}. Deliver your ruling on the ${metric} charge regarding the ${dataset} dataset.`;

    const groqResponse = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "llama-3.1-8b-instant",
        stream: true,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userMessage }
        ]
      })
    });

    if (!groqResponse.ok) {
      const errorText = await groqResponse.text();
      throw new Error(`Groq API Error: ${errorText}`);
    }

    // Stream SSE response to client
    const stream = new ReadableStream({
      async start(controller) {
        if (!groqResponse.body) {
          controller.close();
          return;
        }
        const reader = groqResponse.body.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            controller.enqueue(value);
          }
        } catch (error) {
          controller.error(error);
        } finally {
          controller.close();
        }
      }
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
      }
    });

  } catch (error: any) {
    console.error("Judge Agent Route Error:", error);
    return new Response(JSON.stringify({ error: error.message || "Internal Server Error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
