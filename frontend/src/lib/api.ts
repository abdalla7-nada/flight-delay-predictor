import {
  FlightPredictionRequest,
  PredictionResult,
  ApiMetadata,
  ApiOptions,
  ApiHealth,
  ApiErrorResponse,
} from "@/types";

const RAW_API_URL = process.env.NEXT_PUBLIC_API_URL;
export const API_BASE_URL = (
  RAW_API_URL !== undefined
    ? RAW_API_URL
    : process.env.NODE_ENV === "production"
    ? ""
    : "http://localhost:8000"
).replace(/\/+$/, "");

const TIMEOUT_MS = 8000;

function getFullUrl(path: string): URL {
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  if (API_BASE_URL) {
    return new URL(`${API_BASE_URL}${cleanPath}`);
  }
  const origin = typeof window !== "undefined" ? window.location.origin : "http://localhost:3000";
  return new URL(cleanPath, origin);
}

export class ApiClientError extends Error {
  status?: number;
  fieldErrors?: Record<string, string>;

  constructor(message: string, status?: number, fieldErrors?: Record<string, string>) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
    this.fieldErrors = fieldErrors;
  }
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return res;
  } catch (err: unknown) {
    if (
      (err instanceof Error && err.name === "AbortError") ||
      controller.signal.aborted
    ) {
      throw new ApiClientError("Inference timed out. Please retry.", 408);
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (res.ok) {
    return res.json() as Promise<T>;
  }

  let errorMessage = `Server error (HTTP ${res.status})`;
  const fieldErrors: Record<string, string> = {};

  try {
    const errorData: ApiErrorResponse = await res.json();
    if (typeof errorData.detail === "string") {
      errorMessage = errorData.detail;
      if (errorMessage.toLowerCase().includes("origin and destination")) {
        fieldErrors["dest"] = errorMessage;
      }
    } else if (Array.isArray(errorData.detail)) {
      const messages: string[] = [];
      for (const err of errorData.detail) {
        const cleanMsg = err.msg.replace(/^Value error,\s*/i, "");
        const rawField = err.loc && err.loc.length > 1 ? String(err.loc[err.loc.length - 1]) : "general";
        const field = cleanMsg.toLowerCase().includes("origin and destination") ? "dest" : rawField;
        fieldErrors[field] = cleanMsg;
        messages.push(`${field}: ${cleanMsg}`);
      }
      errorMessage = messages.join(" | ");
    }
  } catch {
    errorMessage = `Request failed with HTTP status ${res.status}: ${res.statusText}`;
  }

  throw new ApiClientError(errorMessage, res.status, fieldErrors);
}

export async function fetchHealth(): Promise<ApiHealth> {
  try {
    const url = getFullUrl("/health");
    const res = await fetchWithTimeout(url.toString(), {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    return await handleResponse<ApiHealth>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    throw new ApiClientError(
      `Failed to connect to backend at ${API_BASE_URL || "origin"}. Ensure API is reachable.`,
      0
    );
  }
}

export async function fetchMetadata(): Promise<ApiMetadata> {
  try {
    const url = getFullUrl("/metadata");
    const res = await fetchWithTimeout(url.toString(), {
      method: "GET",
      headers: { Accept: "application/json" },
      next: { revalidate: 3600 },
    });
    return await handleResponse<ApiMetadata>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    throw new ApiClientError(
      `Failed to fetch model metadata from ${API_BASE_URL || "origin"}.`,
      0
    );
  }
}

export async function fetchOptions(): Promise<ApiOptions> {
  try {
    const url = getFullUrl("/options");
    const res = await fetchWithTimeout(url.toString(), {
      method: "GET",
      headers: { Accept: "application/json" },
      next: { revalidate: 3600 },
    });
    return await handleResponse<ApiOptions>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    throw new ApiClientError(
      `Failed to fetch supported airlines/airports from ${API_BASE_URL || "origin"}.`,
      0
    );
  }
}

export async function predictSingle(
  flight: FlightPredictionRequest,
  includeShap: boolean = true
): Promise<PredictionResult> {
  try {
    const url = getFullUrl("/predict");
    if (!includeShap) {
      url.searchParams.set("include_shap", "false");
    }

    const res = await fetchWithTimeout(url.toString(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(flight),
    });
    return await handleResponse<PredictionResult>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    throw new ApiClientError(
      `Connection failure while predicting flight risk. Ensure API is reachable at ${API_BASE_URL || "origin"}.`,
      0
    );
  }
}

export async function predictBatch(
  flights: FlightPredictionRequest[],
  includeShap: boolean = false
): Promise<PredictionResult[]> {
  if (flights.length > 500) {
    throw new ApiClientError("Batch size exceeds 500 flights ceiling enforced by API.", 422);
  }

  try {
    const url = getFullUrl("/predict/batch");
    if (includeShap) {
      url.searchParams.set("include_shap", "true");
    }

    const res = await fetchWithTimeout(url.toString(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(flights),
    });
    return await handleResponse<PredictionResult[]>(res);
  } catch (err: unknown) {
    if (err instanceof ApiClientError) throw err;
    throw new ApiClientError(
      `Connection failure during batch scoring. Ensure API is reachable at ${API_BASE_URL || "origin"}.`,
      0
    );
  }
}
