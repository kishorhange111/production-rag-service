import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "15s", target: 5 },
    { duration: "30s", target: 25 },
    { duration: "15s", target: 0 },
  ],
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<250", "p(99)<500"],
  },
};

const BASE_URL = __ENV.RAG_BASE_URL || "http://localhost:8080";
const API_KEY = __ENV.RAG_API_KEY || "";

const QUERIES = [
  "Which port do applications use to connect to the database through PgBouncer?",
  "How long are full nightly database backups kept?",
  "Which environment receives automatic deployments on every merge to main?",
  "Where must rollbacks be announced with the release id?",
  "When do new engineers join the on-call rotation?",
];

export default function () {
  const q = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const headers = { "Content-Type": "application/json" };
  if (API_KEY) {
    headers["X-API-Key"] = API_KEY;
  }

  const res = http.post(
    `${BASE_URL}/query`,
    JSON.stringify({ question: q, verify_grounding: true }),
    { headers }
  );

  check(res, {
    "status is 200": (r) => r.status === 200,
    "has request id": (r) => Boolean(r.headers["X-Request-Id"] || r.headers["X-Request-ID"]),
    "returns answer": (r) => {
      try {
        return Boolean(JSON.parse(r.body).answer);
      } catch (_) {
        return false;
      }
    },
  });

  sleep(0.1);
}
