// Tiny static server that serves a representative subset of the dashboard for
// hermetic Playwright runs. Exercises the same DOM contracts the unit tests use.
const http = require("node:http");

const HOME = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>genai-eval</title></head>
<body>
  <header><h1>genai-eval</h1></header>
  <main>
    <h2>Recent runs</h2>
    <a href="/runs/1" data-testid="run-card">
      <div>fake-large (fake)</div>
      <div>2026-05-02 10:00:00</div>
      <strong>72%</strong>
      <span>n=27</span>
    </a>
  </main>
</body></html>`;

const RUN_DETAIL = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Run #1</title></head>
<body>
  <header><h1>genai-eval</h1></header>
  <main>
    <h2>Run #1 fake-large (fake)</h2>
    <table data-testid="pass-rate-grid">
      <tr><th>task</th><th>en</th><th>es</th><th>ja</th></tr>
      <tr><td>qa</td><td>66%</td><td>66%</td><td>100%</td></tr>
    </table>
    <div data-testid="trend-chart">trend chart placeholder</div>
    <h3>Items</h3>
    <table>
      <tr><td>qa-001</td><td>en</td><td>pass</td></tr>
      <tr><td>qa-003</td><td>en</td><td>fail</td></tr>
    </table>
  </main>
</body></html>`;

const NOT_FOUND = `<!doctype html><html><body><h1>Not found</h1></body></html>`;

const server = http.createServer((req, res) => {
  const url = req.url || "/";
  res.setHeader("content-type", "text/html; charset=utf-8");
  if (url === "/" || url.startsWith("/?")) {
    res.statusCode = 200;
    res.end(HOME);
  } else if (/^\/runs\/\d+$/.test(url)) {
    res.statusCode = 200;
    res.end(RUN_DETAIL);
  } else if (url === "/healthz") {
    res.setHeader("content-type", "application/json");
    res.statusCode = 200;
    res.end(`{"status":"ok"}`);
  } else {
    res.statusCode = 404;
    res.end(NOT_FOUND);
  }
});

const port = Number(process.env.PORT || 3000);
server.listen(port, () => {
  console.log(`mock dashboard listening on :${port}`);
});
