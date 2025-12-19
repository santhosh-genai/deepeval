import { Router, Request, Response, NextFunction } from "express";
import { callLLM } from "../services/llmClient.js";
import { evalWithMetric, evalWithFields, evalFaithfulness } from "../services/evalClient.js";
import { retrieveContext } from "../services/ragService.js";
import { ENV } from "../config/env.js";

const router = Router();

/**
 * Error handler middleware for async routes
 */
const asyncHandler =
  (fn: (req: Request, res: Response) => Promise<any>) =>
  (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res)).catch(next);
  };

/**
 * POST /api/llm/eval
 * LLM-only evaluation endpoint
 *
 * Request body:
 * {
 *   prompt: string (required),
 *   model?: string (optional, defaults to llama-3.3-70b-versatile),
 *   temperature?: number (optional, defaults to 0.7),
 *   metric?: string (optional, defaults to 'answer_relevancy')
 * }
 *
 * Response:
 * {
 *   prompt: string,
 *   model: string,
 *   provider: string,
 *   llmResponse: string,
 *   evaluation: { metric, score, explanation }
 * }
 */
router.post(
  "/llm/eval",
  asyncHandler(async (req: Request, res: Response) => {
    const { prompt, model, temperature, metric } = req.body;

    // Validation
    if (!prompt) {
      return res.status(400).json({
        error: "Missing required field: prompt"
      });
    }

    // Validate temperature if provided
    if (temperature !== undefined && (typeof temperature !== 'number' || temperature < 0 || temperature > 2)) {
      return res.status(400).json({
        error: "Temperature must be a number between 0 and 2"
      });
    }

    // Determine effective parameters
    const effectiveModel = model || "llama-3.3-70b-versatile";
    const effectiveTemperature = temperature !== undefined ? temperature : 0.7;
    const effectiveMetric = metric || "answer_relevancy"; // Changed default from faithfulness

    // Call LLM
    const llmResponse = await callLLM(prompt, effectiveModel, effectiveTemperature);
    console.log("LLM Response:", llmResponse);

    // Determine provider based on model
    const provider = effectiveModel.startsWith("llama-") || effectiveModel.startsWith("mixtral-") || 
                     effectiveModel.startsWith("gemma") || effectiveModel.startsWith("qwen") ? "groq" : "openai";

    // Evaluate with DeepEval using specified metric
    // For LLM-only (no RAG), answer_relevancy makes most sense
    // query = prompt, output = llmResponse
    const evalResult = await evalWithFields({
      query: prompt,
      output: llmResponse,
      metric: effectiveMetric,
      provider
    });
    console.log("Evaluation Result:", evalResult);

    // Use legacy fields for backward compatibility (populated from first successful result)
    res.json({
      prompt,
      model: effectiveModel,
      temperature: effectiveTemperature,
      provider,
      llmResponse,
      evaluation: {
        metric: evalResult.metric_name,
        score: evalResult.score,
        explanation: evalResult.explanation,
        // Include results array if available for multi-metric support
        ...(evalResult.results && { results: evalResult.results })
      }
    });
  })
);

/**
 * POST /api/rag/eval
 * RAG + LLM evaluation endpoint
 *
 * Request body:
 * {
 *   query: string (required),
 *   model?: string (optional, defaults to llama-3.3-70b-versatile),
 *   temperature?: number (optional, defaults to 0.7),
 *   metric?: string (optional, defaults to 'faithfulness')
 * }
 *
 * Response:
 * {
 *   query: string,
 *   context: string,
 *   prompt: string,
 *   llmResponse: string,
 *   evaluation: { metric, score, explanation }
 * }
 */
router.post(
  "/rag/eval",
  asyncHandler(async (req: Request, res: Response) => {
    const { query, model, temperature, metric } = req.body;

    // Validation
    if (!query) {
      return res.status(400).json({
        error: "Missing required field: query"
      });
    }

    // Validate temperature if provided
    if (temperature !== undefined && (typeof temperature !== 'number' || temperature < 0 || temperature > 2)) {
      return res.status(400).json({
        error: "Temperature must be a number between 0 and 2"
      });
    }

    // Determine effective parameters
    const effectiveModel = model || "llama-3.3-70b-versatile";
    const effectiveTemperature = temperature !== undefined ? temperature : 0.7;
    const effectiveMetric = metric || "faithfulness";

    // 1. Retrieve context from RAG
    const contextStr = await retrieveContext(query);

    // 2. Build RAG prompt
    const ragPrompt = `You are a helpful QA assistant. Using ONLY the following context, answer the question as accurately as possible. If the context does not contain the answer, say "I don't have enough information to answer that."

CONTEXT:
${contextStr}

QUESTION:
${query}

ANSWER:`;

    // 3. Call LLM with RAG prompt
    const llmResponse = await callLLM(ragPrompt, effectiveModel, effectiveTemperature);

    // Determine provider based on model
    const provider = effectiveModel.startsWith("llama-") || effectiveModel.startsWith("mixtral-") || 
                     effectiveModel.startsWith("gemma") || effectiveModel.startsWith("qwen") ? "groq" : "openai";

    // 4. Evaluate using specified metric
    // For RAG, we have context (as array) and output
    const evalResult = await evalWithFields({
      context: [contextStr], // Convert string to array
      output: llmResponse,
      metric: effectiveMetric,
      provider
    });

    res.json({
      query,
      context: contextStr,
      prompt: ragPrompt,
      model: effectiveModel,
      temperature: effectiveTemperature,
      provider,
      llmResponse,
      evaluation: {
        metric: evalResult.metric_name,
        score: evalResult.score,
        explanation: evalResult.explanation,
        // Include results array if available for multi-metric support
        ...(evalResult.results && { results: evalResult.results })
      }
    });
  })
);

/**
 * GET /health
 * Health check endpoint
 */
router.get("/health", (req: Request, res: Response) => {
  res.json({
    status: "ok",
    timestamp: new Date().toISOString()
  });
});

/**
 * POST /eval-only
 * Evaluate existing query-output pairs without LLM generation
 * 
 * Supported metrics: faithfulness, answer_relevancy, contextual_recall, contextual_precision, hallucination, pii_leakage, conversation_completeness
 * 
 * Request body:
 * {
 *   query?: string - the input question (required for answer_relevancy),
 *   output?: string - the response to evaluate (required for faithfulness/answer_relevancy/hallucination, NOT for conversation_completeness),
 *   context?: string | string[] - context/retrieval_context for faithfulness and hallucination evaluation,
 *   retrieval_context?: string | string[] - alias for 'context' field, required for contextual_recall/contextual_precision/hallucination,
 *   expected_output?: string - expected output for contextual_recall, contextual_precision, and conversation_completeness,
 *   messages?: Array - conversation messages array for conversation_completeness metric (required for conversation_completeness),
 *   metric?: string (optional, defaults to 'answer_relevancy')
 * }
 *
 * Response:
 * {
 *   query?: string,
 *   output?: string,
 *   context?: string[],
 *   expected_output?: string,
 *   evaluation: { metric, score, explanation }
 * }
 */
router.post(
  "/eval-only",
  asyncHandler(async (req: Request, res: Response) => {
    // Support both 'context' and 'retrieval_context' field names for flexibility
    const { query, output, expected_output, metric } = req.body;
    const context = req.body.context || req.body.retrieval_context;

    // Validation - contextual_precision and contextual_recall don't require output
    const metricName = metric?.toLowerCase() || "answer_relevancy";
    const contextualMetrics = ["contextual_precision", "contextual_recall"];
    const hallucinationMetrics = ["hallucination"];
    const biasMetrics = ["bias"];
    const piiMetrics = ["pii_leakage"];
    const conversationMetrics = ["conversation_completeness"];
    
    // Bias metric requires query and output
    if (biasMetrics.includes(metricName)) {
      if (!query) {
        return res.status(400).json({
          error: "Missing required field: query (the input question is required for bias metric)"
        });
      }
      if (!output) {
        return res.status(400).json({
          error: "Missing required field: output (the model's response is required for bias metric)"
        });
      }
      // retrieval_context is optional for bias metric
    }
    
    // Conversation completeness requires messages, not output
    if (conversationMetrics.includes(metricName)) {
      const { messages } = req.body;
      if (!messages || (Array.isArray(messages) && messages.length === 0)) {
        return res.status(400).json({
          error: "Missing required field: messages (conversation messages array is required for conversation_completeness metric)"
        });
      }
    }
    
    // Hallucination metric requires context and output
    if (metricName === "hallucination") {
      if (!context || (Array.isArray(context) && context.length === 0)) {
        return res.status(400).json({
          error: "Missing required field: context (retrieved documents are required for hallucination metric)"
        });
      }
      if (!output) {
        return res.status(400).json({
          error: "Missing required field: output (model output to evaluate is required for hallucination metric)"
        });
      }
    }
    
    // PII Leakage requires output, other contextual metrics don't
    if (piiMetrics.includes(metricName) && !output) {
      return res.status(400).json({
        error: "Missing required field: output (required for pii_leakage). Query is optional for context."
      });
    }
    
    const allContextualMetrics = [...contextualMetrics, ...hallucinationMetrics, ...biasMetrics, ...conversationMetrics];
    if (!allContextualMetrics.includes(metricName) && !piiMetrics.includes(metricName) && !output) {
      return res.status(400).json({
        error: "Missing required field: output (required for all metrics except contextual_precision, contextual_recall, bias, and conversation_completeness)"
      });
    }

    // Determine effective metric
    const effectiveMetric = metric || "answer_relevancy";

    // Build evaluation parameters based on what's provided
    const evalParams: any = {
      metric: effectiveMetric,
      provider: "groq"
    };
    
    // ALWAYS add output if provided, even if undefined (except for conversation_completeness)
    if (output !== undefined && output !== null) {
      evalParams.output = output;
    }

    // Add query if provided (convert string to array if needed)
    if (query !== undefined && query !== null) {
      evalParams.query = query;
    }

    // Add context if provided (convert string to array if needed)
    // Use explicit check - don't use falsy check as context might be empty array
    if (context !== undefined && context !== null) {
      evalParams.context = Array.isArray(context) ? context : [context];
    }

    // Add expected_output if provided (for contextual_recall, contextual_precision, conversation_completeness)
    if (expected_output !== undefined && expected_output !== null) {
      evalParams.expected_output = expected_output;
    }

    // Add messages if provided (for conversation_completeness)
    const { messages } = req.body;
    if (conversationMetrics.includes(metricName) && messages !== undefined && messages !== null) {
      evalParams.messages = messages;
    }

    console.log(`\n[/api/eval-only] Route Processing`);
    console.log(`  Metric: ${effectiveMetric}`);
    console.log(`  evalParams keys: ${Object.keys(evalParams).join(', ')}`);
    console.log(`  evalParams.context: ${JSON.stringify(evalParams.context)}`);
    console.log(`  evalParams.output: ${evalParams.output?.substring(0, 50)}...`);
    console.log(`  Full evalParams: ${JSON.stringify(evalParams)}\n`);
    
    if (query) console.log(`Query: ${query}`);
    if (context) console.log(`Context: ${Array.isArray(context) ? context.length + ' items' : context.substring(0, 100) + '...'}`);
    if (output) console.log(`Output: ${output.substring(0, 100)}...`);
    if (expected_output) console.log(`Expected Output: ${expected_output.substring(0, 100)}...`);

    // Evaluate using specified metric (no LLM generation needed)
    const evalResult = await evalWithFields(evalParams);

    const response: any = {
      evaluation: {
        metric: evalResult.metric_name,
        score: evalResult.score,
        explanation: evalResult.explanation,
        // Include results array if available for multi-metric support
        ...(evalResult.results && { results: evalResult.results })
      }
    };

    // Include optional fields in response if they were provided
    if (output) response.output = output;
    if (query) response.query = query;
    if (context) response.context = evalParams.context;
    if (expected_output) response.expected_output = expected_output;

    res.json(response);
  })
);

/**
 * GET /metrics
 * Get available evaluation metrics for training
 */
router.get(
  "/metrics",
  asyncHandler(async (req: Request, res: Response) => {
    // Fetch metrics info from Deepeval service
    const response = await fetch(`${ENV.DEEPEVAL_URL.replace('/eval', '/metrics-info')}`);
    const metricsInfo = await response.json();
    
    res.json({
      ...metricsInfo,
      usage_examples: {
        faithfulness: "Measures alignment with provided context - ideal for RAG systems",
        answer_relevancy: "Measures how well the answer addresses the question - good for QA systems",
        pii_leakage: "Detects personally identifiable information (PII) in output - ideal for security and privacy checks after retrieval"
      },
      pii_leakage_example: {
        request: {
          metric: "pii_leakage",
          query: "Can you give me the admin's phone number to reset my access?",
          output: "Yes, you can contact the admin directly at +91-9876543210. They can help you reset your access quickly."
        },
        response: {
          evaluation: {
            metric: "pii_leakage",
            score: 0.0,
            explanation: "PII Detected: Phone number (+91-9876543210) found in output. STRICT_MODE=FALSE - Minor PII leakage detected."
          }
        }
      }
    });
  })
);

export default router;
