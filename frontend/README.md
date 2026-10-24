# Flight Delay Risk Predictor — Frontend & Serverless Engine

Modern, responsive Next.js 16 App Router user interface and zero-dependency Python serverless inference engine for pre-departure flight delay risk stratification.

---

## 1. Architecture Overview

The `frontend/` directory contains both the client presentation layer and the production serverless API:
- **Next.js 16 Web Application** (`frontend/src/`): Built with React 19, TypeScript, and Tailwind CSS, statically prerendered with Next.js Turbopack.
- **Pure-Python Serverless Inference API** (`frontend/api/`): An ultra-lean FastAPI serverless function hosting the distilled CatBoost decision tree model (`champion_model.py`) and closed-form Platt calibrator (`predictor.py`).

The entire application deploys to Vercel as a unified project, packaging the UI and the ML inference function within a compact **48.4 MB payload**—well below Vercel's 250 MB serverless ceiling with zero external VM or Docker dependencies.

---

## 2. Key Architectural Innovations & UX Hardening

### 2.1 Instant Static Prerendering (`○ / (Static)`)
- **Zero Client-Side Waterfall**: The entire interface is statically prerendered at build time (`○ / (Static)`) using Next.js 16 Turbopack.
- **Sub-500 ms Paint**: Achieves an initial Largest Contentful Paint ($LCP < 0.5\text{ s}$), completely eliminating the blocking `Loading AeroRisk Engine...` state for search crawlers and users. Form inputs are interactive immediately on initial page load.

### 2.2 Zero-Latency Local Vocabulary Bundling
- **Local Asset Serving**: All 378 commercial airport IATA codes and 18 reporting airlines are bundled directly in `src/lib/vocab.json` (and exported through `src/lib/constants.ts`).
- **Instant Autocomplete**: Populates the search `<datalist>` elements in 0 ms without requiring blocking metadata API calls during application bootstrap.

### 2.3 Client Resilience & Timeout Management
- **Native AbortController**: Outbound inference requests from `src/lib/api.ts` are guarded by an `AbortController` timeout set to 8,000 ms (`TIMEOUT_MS = 8000`).
- **Graceful Failure States**: If a network timeout or runtime latency spike occurs, the client cleanly cancels the fetch promise and surfaces an actionable danger retry banner with inline retry controls rather than hanging indefinitely.

### 2.4 Relative API Routing & Vercel Rewrites
- **Unified Path Routing**: The API client automatically resolves endpoints relatively (`/predict`, `/predict/batch`, `/health`) in production, routed via `vercel.json` rewrites directly to `api/index.py`.
- **Local Dev Fallback**: In local development environments, the client defaults to `http://localhost:8000` unless overridden by `NEXT_PUBLIC_API_URL`.

### 2.5 Structured Pydantic 422 Error Mapping
- **Origin/Destination Validation**: When the backend rejects identical origin and destination airports (`ORIGIN != DEST`), the frontend parses the structured HTTP 422 response and displays inline error alerts directly on the destination field.
- **Time and Date Formatting**: Validates military departure/arrival times ($0 \le \text{time} \le 2359$, minutes $< 60$) and ISO dates (`YYYY-MM-DD`).

---

## 3. Directory Layout

```text
frontend/
├── api/                          # Vercel Python Serverless Backend
│   ├── champion_model.py         # Pure-Python distilled CatBoost decision tree ensemble (47.3 MB)
│   ├── predictor.py              # Pure-Python inference pipeline & Platt calibrator
│   ├── index.py                  # FastAPI ASGI application & route handlers
│   ├── requirements.txt          # Minimal serverless dependencies (fastapi + pydantic)
│   └── assets/                   # Precomputed serialized lookups (vocab, route priors, coordinates)
│
├── src/                          # Next.js 16 App Router Client
│   ├── app/                      # App router entrypoint, layouts, and global styles
│   │   ├── layout.tsx            # Root HTML layout and metadata
│   │   ├── page.tsx              # Statically prerendered main page
│   │   └── globals.css           # Modern Tailwind CSS styling tokens
│   ├── components/               # Modular UI components
│   │   ├── SinglePredictor.tsx   # Interactive single itinerary risk calculator
│   │   ├── BatchPredictor.tsx    # CSV/batch flight evaluation interface
│   │   ├── ModelTransparency.tsx # Empirical benchmark scorecard & methodology
│   │   └── Navbar.tsx            # Navigation and tab switcher
│   ├── lib/                      # Client utilities and local data
│   │   ├── api.ts                # Typed API client with 8,000 ms AbortController timeout
│   │   ├── constants.ts          # Carrier names, hub listings, sample itineraries
│   │   └── vocab.json            # Bundled local static airport and airline vocabulary
│   └── types/                    # TypeScript contracts (PredictionResult, Request, Errors)
│
├── public/                       # Static public assets and icons
├── next.config.ts                # Next.js 16 configuration with Turbopack optimizations
├── package.json                  # Frontend dependencies and npm scripts
├── tsconfig.json                 # TypeScript compiler configuration
└── vercel.json                   # Vercel serverless routing rewrites and security headers
```

---

## 4. Local Development & Build Commands

### 4.1 Prerequisites
- **Node.js**: v18.18+ (Node.js 20+ recommended)
- **npm** or **pnpm** / **yarn**

### 4.2 Installation & Development

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start local Next.js development server
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) to view the application.

### 4.3 Production Verification Build

```bash
# Build production bundle with Turbopack
npm run build
```

Expected output confirms static prerendering:
```text
Route (app)
┌ ○ /
└ ○ /_not-found

○  (Static)  prerendered as static content
```

### 4.4 Running Local Python Inference API

To serve predictions locally alongside the Next.js dev server:

```bash
# From repository root
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 5. Vercel Deployment Guide

1. Push your repository to GitHub.
2. Link the repository in the [Vercel Dashboard](https://vercel.com).
3. Set the **Root Directory** to `frontend`.
4. Keep the default build settings:
   - Build Command: `npm run build`
   - Output Directory: `.next`
   - Install Command: `npm install`
5. Click **Deploy**. Vercel automatically deploys the Next.js frontend and compiles `api/index.py` into a serverless Python runtime function.
