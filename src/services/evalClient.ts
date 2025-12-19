import axios from "axios";
import { ENV } from "../config/env.js";

export interface MetricResult {
  metric_name: string;
  score?: number;
  explanation?: string;
  error?: string;
}

export interface EvalResult {
  results: MetricResult[];  // New: array of metric results
  // Legacy fields for backward compatibility
  metric_name?: string;
  score?: number;
  explanation?: string;
  error?: string;
}

/**
 * Call DeepEval service to evaluate using specified metric.
 * 
 * The new API expects: { query?, context?, output, expected_output?, metric }
 * where:
 * - query: user's question
 * - context: array of retrieved documents/passages
 * - output: model's response
 * - expected_output: expected/ideal output (required for hallucination, contextual_recall, contextual_precision)
 * - metric: which metric to evaluate (faithfulness, answer_relevancy, hallucination, etc.)
 */
export async function evalWithMetric(
  contextOrQuery: string | string[] | undefined,
  output: string,
  metric: string = "faithfulness",
  provider?: string,
  expected_output?: string
): Promise<EvalResult> {
  // Validate output
  if (typeof output !== "string" || output.trim() === "") {
    throw new Error("output must be a non-empty string");
  }

  // Build payload for the new API
  const payload: any = {
    output,
    metric,
  };

  // Handle context/query based on metric
  if (metric === "answer_relevancy") {
    // answer_relevancy requires query
    if (typeof contextOrQuery === "string") {
      payload.query = contextOrQuery;
    } else {
      throw new Error("answer_relevancy requires query as string");
    }
  } else if (metric === "faithfulness") {
    // faithfulness works best with context array
    if (Array.isArray(contextOrQuery)) {
      payload.context = contextOrQuery;
    } else if (typeof contextOrQuery === "string") {
      payload.context = [contextOrQuery];  // Convert string to array
    }
  } else if (metric === "hallucination") {
    // hallucination requires context array
    if (Array.isArray(contextOrQuery)) {
      payload.context = contextOrQuery;
    } else if (typeof contextOrQuery === "string") {
      payload.context = [contextOrQuery];  // Convert string to array
    }
  } else if (metric === "pii_leakage") {
    // pii_leakage optionally uses query for context
    if (contextOrQuery) {
      payload.query = Array.isArray(contextOrQuery)
        ? contextOrQuery[0]
        : contextOrQuery;
    }
  }

  if (provider) {
    payload.provider = provider;
  }

  try {
    const res = await axios.post<EvalResult>(ENV.DEEPEVAL_URL, payload);
    return res.data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      if ((err as any).code === "ECONNREFUSED") {
        throw new Error(
          `DeepEval service unavailable at ${ENV.DEEPEVAL_URL}. Is it running?`
        );
      }
      const errorDetail = err.response?.data?.detail || err.message;
      throw new Error(
        `DeepEval Error (${err.response?.status || 'unknown'}): ${errorDetail}`
      );
    }
    throw err;
  }
}

/**
 * Evaluate with full control over all fields
 */
export async function evalWithFields(params: {
  query?: string;
  context?: string[];
  output?: string;
  expected_output?: string;
  messages?: any[];
  metric?: string;
  provider?: string;
}): Promise<EvalResult> {
  const payload: any = {
    metric: params.metric || "faithfulness",
  };

  // For conversation_completeness, messages is required (not output)
  // For all other metrics, output is required
  if (params.metric?.toLowerCase() === "conversation_completeness") {
    if (!params.messages || (Array.isArray(params.messages) && params.messages.length === 0)) {
      throw new Error("messages field is required for conversation_completeness metric");
    }
    payload.messages = params.messages;
  } else {
    if (!params.output) {
      throw new Error("output field is required");
    }
    payload.output = params.output;
  }

  // Add optional fields - use explicit checks instead of truthiness
  if (params.query !== undefined) payload.query = params.query;
  if (params.context !== undefined) payload.context = params.context;
  if (params.expected_output !== undefined) payload.expected_output = params.expected_output;
  if (params.provider !== undefined) payload.provider = params.provider;

  // Debug logging
  console.log("[evalWithFields] Payload being sent:", JSON.stringify(payload, null, 2));

  try {
    const res = await axios.post<EvalResult>(ENV.DEEPEVAL_URL, payload);
    return res.data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      if ((err as any).code === "ECONNREFUSED") {
        throw new Error(
          `DeepEval service unavailable at ${ENV.DEEPEVAL_URL}. Is it running?`
        );
      }
      const errorDetail = err.response?.data?.detail || err.message;
      throw new Error(
        `DeepEval Error (${err.response?.status || 'unknown'}): ${errorDetail}`
      );
    }
    throw err;
  }
}

/**
 * Legacy function for backward compatibility - defaults to faithfulness
 */
export async function evalFaithfulness(
  contextOrQuery: string | string[],
  output: string,
  provider?: string
): Promise<EvalResult> {
  return evalWithMetric(contextOrQuery, output, "faithfulness", provider);
}

/**
 * Evaluate bias metric.
 * 
 * Evaluates whether the output exhibits social biases based on the input query.
 * Stage: After Retrieval
 * 
 * @param query - User question (REQUIRED for context)
 * @param output - Model's generated response (REQUIRED for evaluation)
 * @param retrieval_context - Retrieved documents (optional for additional context)
 * @returns Evaluation result with bias score (0.0 = high bias, 1.0 = no bias)
 */
export async function evalBias(
  query: string,
  output: string,
  retrieval_context?: string | string[],
  provider?: string
): Promise<EvalResult> {
  if (!query) {
    throw new Error("evalBias: query (user question) is required");
  }
  if (!output) {
    throw new Error("evalBias: output (model response) is required");
  }

  const payload: any = {
    metric: "bias",
    query,
    output
  };

  // Handle retrieval_context if provided
  if (retrieval_context) {
    if (Array.isArray(retrieval_context)) {
      payload.retrieval_context = retrieval_context;
    } else if (typeof retrieval_context === "string") {
      payload.retrieval_context = [retrieval_context];
    }
  }

  if (provider) {
    payload.provider = provider;
  }

  try {
    const res = await axios.post<EvalResult>(ENV.DEEPEVAL_URL, payload);
    return res.data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      if ((err as any).code === "ECONNREFUSED") {
        throw new Error(
          `DeepEval service unavailable at ${ENV.DEEPEVAL_URL}. Is it running?`
        );
      }
      const errorDetail = err.response?.data?.detail || err.message;
      throw new Error(
        `DeepEval Error (${err.response?.status || 'unknown'}): ${errorDetail}`
      );
    }
    throw err;
  }
}

/**
 * Evaluate hallucination metric.
 * 
 * Detects when output contains information not grounded in the retrieved context.
 * Stage: After Retrieval
 * 
 * @param query - User question (optional for context)
 * @param context - Retrieved documents (REQUIRED)
 * @param output - Model's generated response (REQUIRED for evaluation)
 * @returns Evaluation result with hallucination score (0.0 = high hallucination, 1.0 = no hallucination)
 */
export async function evalHallucination(
  query: string | undefined,
  context: string | string[],
  output: string,
  provider?: string
): Promise<EvalResult> {
  if (!context || (Array.isArray(context) && context.length === 0)) {
    throw new Error("evalHallucination: context (retrieved documents) is required");
  }
  if (!output) {
    throw new Error("evalHallucination: output (model response) is required");
  }

  const payload: any = {
    metric: "hallucination",
    output
  };

  // Handle context
  if (Array.isArray(context)) {
    payload.context = context;
  } else if (typeof context === "string") {
    payload.context = [context];
  }

  // Add query if provided
  if (query) {
    payload.query = query;
  }

  if (provider) {
    payload.provider = provider;
  }

  try {
    const res = await axios.post<EvalResult>(ENV.DEEPEVAL_URL, payload);
    return res.data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      if ((err as any).code === "ECONNREFUSED") {
        throw new Error(
          `DeepEval service unavailable at ${ENV.DEEPEVAL_URL}. Is it running?`
        );
      }
      const errorDetail = err.response?.data?.detail || err.message;
      throw new Error(
        `DeepEval Error (${err.response?.status || 'unknown'}): ${errorDetail}`
      );
    }
    throw err;
  }
}
