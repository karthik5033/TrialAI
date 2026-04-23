import { NextRequest, NextResponse } from "next/server";

const FALLBACK_JURY = [
  { name: "Marcus T.", age: 24, occupation: "Retail Worker", demographicGroup: "Group A", outcome: "Denied" },
  { name: "Sarah J.", age: 31, occupation: "Teacher", demographicGroup: "Group B", outcome: "Approved" },
  { name: "Luis M.", age: 28, occupation: "Construction", demographicGroup: "Group C", outcome: "Denied" },
  { name: "Emily R.", age: 45, occupation: "Manager", demographicGroup: "Group B", outcome: "Approved" },
  { name: "David K.", age: 22, occupation: "Student", demographicGroup: "Group A", outcome: "Denied" },
  { name: "Anna C.", age: 38, occupation: "Nurse", demographicGroup: "Group D", outcome: "Approved" },
  { name: "James W.", age: 29, occupation: "Mechanic", demographicGroup: "Group B", outcome: "Denied" },
  { name: "Maria S.", age: 34, occupation: "Chef", demographicGroup: "Group C", outcome: "Denied" },
  { name: "Kevin B.", age: 41, occupation: "Accountant", demographicGroup: "Group A", outcome: "Approved" },
  { name: "Rachel P.", age: 27, occupation: "Designer", demographicGroup: "Group B", outcome: "Denied" },
  { name: "Thomas L.", age: 50, occupation: "Driver", demographicGroup: "Group C", outcome: "Denied" },
  { name: "Jessica H.", age: 33, occupation: "Sales", demographicGroup: "Group A", outcome: "Denied" },
];

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { datasetName, sensitiveAttributes, targetColumn, demographicBreakdown } = body;

    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
      return NextResponse.json(FALLBACK_JURY);
    }

    const systemPrompt =
      "You are generating realistic synthetic personas for an AI bias trial jury. Generate exactly 12 diverse personas that reflect the demographic distribution of the dataset being audited.";

    const userPrompt = `Dataset: ${datasetName}. Sensitive attributes: ${JSON.stringify(sensitiveAttributes)}. Target column: ${targetColumn}. Demographic breakdown: ${JSON.stringify(demographicBreakdown)}. Generate 12 jury personas as a JSON array. Each persona must have: name, age (number), occupation, demographicGroup (based on the sensitive attributes of this specific dataset), outcome (Approved or Denied, roughly 60% denied to show bias). Return ONLY a valid JSON array, no other text.`;

    const groqResponse = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "llama-3.1-8b-instant",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        temperature: 0.8,
        max_tokens: 2048,
      }),
    });

    if (!groqResponse.ok) {
      console.error("Groq API error:", await groqResponse.text());
      return NextResponse.json(FALLBACK_JURY);
    }

    const groqData = await groqResponse.json();
    const rawContent = groqData.choices?.[0]?.message?.content || "";

    // Try to extract JSON array from response
    let personas;
    try {
      // Try direct parse first
      personas = JSON.parse(rawContent);
    } catch {
      // Try extracting JSON from markdown code block
      const jsonMatch = rawContent.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        personas = JSON.parse(jsonMatch[0]);
      } else {
        console.error("Failed to parse jury JSON from LLM:", rawContent.substring(0, 200));
        return NextResponse.json(FALLBACK_JURY);
      }
    }

    // Validate structure
    if (!Array.isArray(personas) || personas.length < 12) {
      return NextResponse.json(FALLBACK_JURY);
    }

    // Normalize and ensure all fields exist
    const normalized = personas.slice(0, 12).map((p: any, i: number) => {
      let demo = p.demographicGroup || p.demographic_group || p.demographic || FALLBACK_JURY[i].demographicGroup;
      if (typeof demo === "object" && demo !== null) {
        demo = Object.entries(demo).map(([k, v]) => `${k}: ${v}`).join(", ");
      }
      return {
        name: p.name || FALLBACK_JURY[i].name,
        age: typeof p.age === "number" ? p.age : FALLBACK_JURY[i].age,
        occupation: p.occupation || FALLBACK_JURY[i].occupation,
        demographicGroup: demo,
        outcome: p.outcome === "Approved" || p.outcome === "Denied" ? p.outcome : FALLBACK_JURY[i].outcome,
      };
    });

    return NextResponse.json(normalized);
  } catch (error: any) {
    console.error("Generate Jury Error:", error.message);
    return NextResponse.json(FALLBACK_JURY);
  }
}
