#!/bin/bash
source ~/sloguard-env/bin/activate
export HF_HUB_OFFLINE=1

test_config() {
    local label=$1
    shift
    echo "=== Testing $label ==="
    timeout 60 python3 -m vllm.entrypoints.openai.api_server "$@" > /dev/null 2>&1 &
    local pid=$!
    for i in $(seq 1 30); do
        if curl -s http://localhost:${port}/health > /dev/null 2>&1; then
            echo "  $label: STARTED (${i}x2s)"
            # Quick inference test
            resp=$(curl -s http://localhost:${port}/v1/chat/completions \
                -H "Content-Type: application/json" \
                -d '{"model":"Qwen/Qwen2-1.5B","messages":[{"role":"user","content":"Hi"}],"max_tokens":3}')
            echo "  $label: response=$(echo $resp | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("choices",[{}])[0].get("message",{}).get("content","ERROR"))' 2>/dev/null)"
            kill $pid 2>/dev/null
            wait $pid 2>/dev/null
            return 0
        fi
        sleep 2
    done
    echo "  $label: FAILED to start"
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null
    return 1
}

port=9901
test_config "config-A" --model Qwen/Qwen2-1.5B --port $port \
    --max-num-seqs 4 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.75 \
    --max-model-len 2048 --enforce-eager --no-enable-chunked-prefill --enable-prefix-caching

sleep 3

port=9902
test_config "config-B" --model Qwen/Qwen2-1.5B --port $port \
    --max-num-seqs 32 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.79 \
    --max-model-len 1024 --no-enforce-eager --enable-chunked-prefill --enable-prefix-caching

sleep 3

port=9903
test_config "config-C" --model Qwen/Qwen2-1.5B --port $port \
    --max-num-seqs 64 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.80 \
    --max-model-len 512 --no-enforce-eager --enable-chunked-prefill --no-enable-prefix-caching

echo ""
echo "All tests complete."
