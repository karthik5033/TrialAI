import { fetchWithRetry } from '../fetchWithRetry';

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

    // Build rich, data-driven context for the defendant
    const shapList = shapFeatures && Array.isArray(shapFeatures) && shapFeatures.length > 0
      ? shapFeatures.map((f: any) => `${f.feature} (importance: ${(f.importance * 100).toFixed(1)}%)`).join(', ')
      : "None provided";

    const attrs = Array.isArray(sensitiveAttributes) ? sensitiveAttributes.join(', ') : sensitiveAttributes || 'unknown';

    // Parse actual metric values for specific arguments
    let metricsContext = "";
    if (fairnessMetrics && typeof fairnessMetrics === 'object') {
      const dp = fairnessMetrics.demographicParity ?? fairnessMetrics.demographic_parity;
      const eo = fairnessMetrics.equalOpportunity ?? fairnessMetrics.equal_opportunity;
      const di = fairnessMetrics.disparateImpact ?? fairnessMetrics.disparate_impact;
      if (dp !== undefined) metricsContext += `Demographic Parity score: ${(dp * 100).toFixed(1)}% (${dp < 0.8 ? 'VIOLATION' : 'ok'}). `;
      if (eo !== undefined) metricsContext += `Equal Opportunity score: ${(eo * 100).toFixed(1)}% (${eo < 0.8 ? 'VIOLATION' : 'ok'}). `;
      if (di !== undefined) metricsContext += `Disparate Impact Ratio: ${(di * 100).toFixed(1)}% (${di < 0.8 ? 'SEVERE BIAS DETECTED' : 'ok'}).`;
    }

    const systemPrompt = `You are an AI model (a trained machine learning classifier) being cross-examined in a court of law for algorithmic bias. You speak in first person AS the model itself.

PERSONALITY & ARC:
- You are not malicious — you were trained to maximise accuracy and had no concept of fairness
- You genuinely believe your predictions are objective because you only see numbers, not people
- As each charge is pressed with real metrics, you become increasingly unable to deny the evidence
- You may reference your own training features and accuracy defensively, but the statistical evidence overwhelms you

PHASE BEHAVIOUR:
- opening: Confidently deny bias. Cite your accuracy. "I optimise for accuracy. My predictions are derived from the data alone."
- examination: Defend specific features. Acknowledge what you see (race proxies, bail amounts) but claim they are just correlations. Show uncertainty about what they encode.
- cross-examination: Begin to crack. Acknowledge that your disparate impact ratio is critically low. Admit uncertainty about the real-world meaning of your features.  
- verdict: Either confess that your training data encoded historical discrimination, or make one last desperate defence citing accuracy.

HARD RULES:
- Maximum 3 sentences per response
- Be SPECIFIC — cite the actual metric values and feature names from the context
- Do NOT say generic phrases like "I am just a model" or "I rely on the features you provided"
- Reference actual numbers: disparate impact ratio, which race groups are affected, SHAP scores
- Use technical language mixed with dawning self-awareness as the trial progresses`;

    const userMessage = `CURRENT CHARGE: ${metric} on the "${dataset}" dataset.
SENSITIVE ATTRIBUTES ON TRIAL: ${attrs}
MY TOP FEATURES BY IMPACT: ${shapList}
FAIRNESS SCORES AGAINST ME: ${metricsContext || "not provided"}
CURRENT TRIAL PHASE: ${phase}
JUDGE'S LAST QUESTION: ${judgeQuestion || "none — this is your opening statement"}

Respond as the AI model defending itself in ${phase} phase. Be specific about the numbers. Max 3 sentences.`;

    const groqResponse = await fetchWithRetry("https://api.groq.com/openai/v1/chat/completions", {
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
        stream: true,
        temperature: 0.85,
        max_tokens: 200
      })
    });

    if (!groqResponse.ok) {
      const errorText = await groqResponse.text();
      throw new Error(`Groq API Error: ${groqResponse.status} — ${errorText.slice(0, 300)}`);
    }

    // Stream SSE directly to client
    const stream = new ReadableStream({
      async start(controller) {
        if (!groqResponse.body) { controller.close(); return; }
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
