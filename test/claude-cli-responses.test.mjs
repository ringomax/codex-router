import assert from "node:assert/strict";
import { createServer } from "node:http";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import test from "node:test";

import { responsesResult, runClaudeTurn, serveRequest } from "../src/claude-cli-responses.mjs";

async function withServer(runTurn, work) {
  const server = createServer((request, response) => serveRequest(request, response, { runTurn }));
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try { return await work(`http://127.0.0.1:${server.address().port}`); }
  finally { await new Promise((resolve) => server.close(resolve)); }
}

test("Claude CLI adapter emits a complete Responses function call", async () => {
  await withServer(async () => ({ message: "", calls: [{ name: "echo", arguments: '{"value":7}' }], usage: { input_tokens: 2, output_tokens: 3 } }), async (base) => {
    const response = await fetch(`${base}/v1/responses`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "claude-opus", stream: true, input: [{ role: "user", content: "Use echo" }] }),
    });
    assert.equal(response.status, 200);
    const events = (await response.text()).split("\n\n")
      .map((frame) => frame.split("\n").find((line) => line.startsWith("data: "))?.slice(6))
      .filter((data) => data && data !== "[DONE]").map(JSON.parse);
    const done = events.find((event) => event.type === "response.output_item.done");
    assert.equal(done.item.type, "function_call");
    assert.equal(done.item.name, "echo");
    assert.deepEqual(JSON.parse(done.item.arguments), { value: 7 });
    assert.ok(done.item.call_id.startsWith("call_"));
    assert.equal(events.at(-1).type, "response.completed");
    assert.deepEqual(events.at(-1).response.usage, { input_tokens: 2, output_tokens: 3, total_tokens: 5 });
  });
});

test("Claude CLI adapter returns text in non-streaming Responses form", async () => {
  await withServer(async () => ({ message: "PONG", calls: [], usage: {} }), async (base) => {
    const response = await fetch(`${base}/v1/responses`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "claude-opus", input: [{ role: "user", content: "Ping" }] }),
    });
    assert.equal(response.status, 200);
    const result = await response.json();
    assert.equal(result.output[0].content[0].text, "PONG");
    assert.equal(result.status, "completed");
  });
});

test("Claude CLI adapter accepts a string Responses input", async () => {
  await withServer(async (body) => {
    assert.equal(body.input, "Ping");
    return { message: "PONG", calls: [], usage: {} };
  }, async (base) => {
    const response = await fetch(`${base}/v1/responses`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "claude-opus", input: "Ping" }),
    });
    assert.equal(response.status, 200);
    assert.equal((await response.json()).output[0].content[0].text, "PONG");
  });
});

test("Claude CLI adapter refuses images before launching the CLI", async () => {
  await assert.rejects(
    runClaudeTurn({ input: [{ role: "user", content: [{ type: "input_image", image_url: "data:image/png;base64,AA==" }] }] }, {
      spawnImpl: () => { throw new Error("CLI must not start"); },
    }),
    (error) => error.status === 400 && /text input only/.test(error.message),
  );
});

test("Responses result keeps text and tool calls as separate output items", () => {
  const result = responsesResult({ message: "Checking", calls: [{ name: "echo", arguments: "{}" }], usage: {} });
  assert.deepEqual(result.output.map((item) => item.type), ["message", "function_call"]);
});

test("Claude CLI adapter advertises Opus, Sonnet, and Fable", async () => {
  await withServer(async () => ({ message: "PONG", calls: [], usage: {} }), async (base) => {
    const catalog = await (await fetch(`${base}/v1/models`)).json();
    assert.deepEqual(catalog.data.map((model) => model.id), ["claude-opus", "claude-sonnet", "claude-fable"]);
    for (const model of catalog.data) {
      const response = await fetch(`${base}/v1/responses`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: model.id, input: "Ping" }),
      });
      assert.equal(response.status, 200);
      assert.equal((await response.json()).model, model.id);
    }
  });
});

test("Claude CLI adapter sends the chosen model and effort to Claude", async () => {
  for (const [model, cli] of [["claude-opus", "claude-opus-5-5"], ["claude-sonnet", "sonnet"], ["claude-fable", "fable"]]) {
    let args;
    const spawnImpl = (_binary, launchedArgs) => {
      args = launchedArgs;
      const child = new EventEmitter();
      child.stdin = new PassThrough();
      child.stdout = new PassThrough();
      child.stderr = new PassThrough();
      child.stdin.on("finish", () => {
        child.stdout.end(JSON.stringify({ structured_output: { message: "PONG", tool_calls: [] }, usage: {} }));
        child.emit("exit", 0);
      });
      return child;
    };
    const turn = await runClaudeTurn({ model, input: "Ping", reasoning: { effort: "xhigh" } }, { spawnImpl });
    assert.equal(turn.message, "PONG");
    assert.equal(args[args.indexOf("--model") + 1], cli);
    assert.equal(args[args.indexOf("--effort") + 1], "xhigh");
  }
});
