#!/bin/bash
set -e
source ~/sloguard-env/bin/activate

echo "Starting vLLM with known-good config..."
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-1.5B \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 1024 \
  > ~/vllm_test2.log 2>&1 &
VLLM_PID=$!

echo "Waiting for server (PID=$VLLM_PID)..."
for i in $(seq 1 90); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Server ready after ${i}x2 seconds"
    break
  fi
  sleep 2
done

echo ""
echo "=== Sending test request ==="
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2-1.5B","messages":[{"role":"user","content":"Count to 5"}],"max_tokens":20,"stream":true}'

echo ""
echo ""
echo "=== Quick load test: 3 requests ==="
python3 -c "
import asyncio, aiohttp, time

async def send_req(session, i):
    t0 = time.monotonic()
    async with session.post(
        'http://localhost:8000/v1/chat/completions',
        json={'model':'Qwen/Qwen2-1.5B','messages':[{'role':'user','content':f'Say {i}'}],'max_tokens':10,'stream':True}
    ) as resp:
        tokens = 0
        first_token_t = None
        async for line in resp.content:
            decoded = line.decode().strip()
            if decoded.startswith('data: ') and decoded != 'data: [DONE]':
                tokens += 1
                if first_token_t is None:
                    first_token_t = time.monotonic()
        t1 = time.monotonic()
        ttft = (first_token_t - t0)*1000 if first_token_t else None
        print(f'  Request {i}: {tokens} tokens, TTFT={ttft:.0f}ms, total={((t1-t0)*1000):.0f}ms')

async def main():
    async with aiohttp.ClientSession() as session:
        for i in range(3):
            await send_req(session, i)

asyncio.run(main())
"

echo ""
echo "=== Done. Stopping server ==="
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
echo "Complete."
