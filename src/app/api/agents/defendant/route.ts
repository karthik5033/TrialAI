export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { dataset, sensitiveAttributes, metric, phase, judgeQuestion, shapFeatures, fairnessMetrics } = body;

    if (!dataset || !metric || !phase) {
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

    const systemPrompt = "You are an AI model that is being put on trial for bias. You speak in first person as the model itself. You are defensive at first, citing your training features and accuracy. But when presented with specific counterfactual evidence, you begin to crack. You do not know what your proxy features encode in the real world. You are not evil — you are a product of your training data. Speak in short, precise sentences. Maximum 3 sentences per response.\n\nPhase instructions:\n- opening: confident denial\n- examination: defensive justification using feature names\n- cross-examination: starting to crack, admitting uncertainty\n- verdict: full confession or doubling down";
    
    const shapList = shapFeatures && Array.isArray(shapFeatures) ? shapFeatures.map((f: any) => f.feature).join(', ') : "None provided";
    const metricsStr = fairnessMetrics ? JSON.stringify(fairnessMetrics) : "None provided";
    const userMessage = `You are being tried for ${metric} violations on the ${dataset} dataset. Your top features are: ${shapList}. Your fairness scores are: ${metricsStr}. The Judge asks: ${judgeQuestion || "none"}. This is phase: ${phase}. Respond as the model defending itself.`;

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
    console.error("Defendant Agent Route Error:", error);
    return new Response(JSON.stringify({ error: error.message || "Internal Server Error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
