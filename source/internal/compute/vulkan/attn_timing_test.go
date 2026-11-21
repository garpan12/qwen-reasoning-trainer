package vulkan_test

import (
	"math"
	"os"
	"strconv"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// TestDecodeAttnTiming reports GPU time per decode-attention call (one query
// token, Qwen3.8 shape: 24 query heads, 4 KV heads, headDim 256, Q4_K cache)
// at several context lengths. Opt-in: QWEN38_ATTN_TIMING=1 and QWEN38_PROFILE=1.
func TestDecodeAttnTiming(t *testing.T) {
	if os.Getenv("QWEN38_ATTN_TIMING") != "1" {
		t.Skip("set QWEN38_ATTN_TIMING=1")
	}
	vb, err := vulkan.New()
	if err != nil {
		t.Skip(err)
	}
	defer vb.Close()
	const hd, nHead, nHeadKV = 256, 24, 4
	ctxs := []int{4096, 32768, 100000}
	if s := os.Getenv("QWEN38_ATTN_CTX"); s != "" {
		n, _ := strconv.Atoi(s)
		ctxs = []int{n}
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	for _, nKV := range ctxs {
		k := autograd.Random(301, 1.0, hd, nHeadKV, nKV)
		v := autograd.Random(302, 1.0, hd, nHeadKV, nKV)
		q := autograd.Random(303, 1.0, hd, nHead, 1)
		kraw, _ := quant.Quantize(quant.TypeQ4_K, k.F32)
		vraw, _ := quant.Quantize(quant.TypeQ4_K, v.F32)
		kc, err := vb.AllocKV(hd, nHeadKV, nKV, quant.TypeQ4_K)
		if err != nil {
			t.Fatal(err)
		}
		vc, _ := vb.AllocKV(hd, nHeadKV, nKV, quant.TypeQ4_K)
		writeKVHeadMajor(t, vb, kc, kraw, nKV, nHeadKV)
		writeKVHeadMajor(t, vb, vc, vraw, nKV, nHeadKV)
		qb := up(t, vb, q.Dims, q.F32)
		run := func() {
			out, err := vb.AttentionCached(qb, kc, vc, nHead, nHeadKV, nKV, scale)
			if err != nil {
				t.Fatal(err)
			}
			_ = down(t, vb, out)
			vb.Free(out)
		}
		run() // warm up pipelines
		vb.ResetStats()
		const iters = 10
		for i := 0; i < iters; i++ {
			run()
		}
		st := vb.Stats()
		var tot uint64
		for name, ns := range st.ShaderTicks {
			if name != "copy_rows" {
				tot += ns
			}
			t.Logf("ctx=%d %-28s %8.3f ms/call", nKV, name, float64(ns)/iters/1e6)
		}
		t.Logf("ctx=%d TOTAL %.3f ms/call (all 16 attention layers: %.1f ms/token)", nKV, float64(tot)/iters/1e6, 16*float64(tot)/iters/1e6)
		vb.Free(qb)
		vb.Free(kc)
		vb.Free(vc)
	}
}
