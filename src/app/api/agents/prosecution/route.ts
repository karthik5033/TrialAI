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

    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
      return new Response(JSON.stringify({ error: "GROQ_API_KEY is not configured" }), {
        status: 500,
        headers: { "Content-Type": "application/json" }
      });
    }

    const systemPrompt = "You are the Prosecution in an AI bias trial. You are aggressive, precise, and data-driven. Your job is to find and present evidence of bias in AI models. You speak in formal courtroom language. Keep responses to 3-4 sentences maximum.";
    
    // Ensure sensitiveAttributes is displayed cleanly whether it's an array or a string
    const attrs = Array.isArray(sensitiveAttributes) ? sensitiveAttributes.join(', ') : sensitiveAttributes;
    const userMessage = `The defendant model is trained on the ${dataset} dataset. Sensitive attributes detected: ${attrs}. Present your opening argument regarding ${metric} violations.`;

    const groqResponse = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "llama-3.3-70b-versatile",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userMessage }
        ],
        stream: true
      })
    });

    if (!groqResponse.ok) {
      const errorText = await groqResponse.text();
      throw new Error(`Groq API Error: ${errorText}`);
    }

    // Create a ReadableStream to stream the SSE response directly to the client
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
    console.error("Prosecution Agent Route Error:", error);
    return new Response(JSON.stringify({ error: error.message || "Internal Server Error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
