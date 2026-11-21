package vulkan

import (
	"encoding/binary"
	"os"
	"unsafe"
)

// bufRange is a byte range of one VkBuffer, used for hazard tracking.
type bufRange struct {
	buf    uintptr
	lo, hi uint64
}

func rangeOf(x *buffer) bufRange {
	return bufRange{buf: x.buf, lo: x.offset, hi: x.offset + x.size}
}

func overlapsAny(rs []bufRange, r bufRange) bool {
	for _, q := range rs {
		if q.buf == r.buf && q.lo < r.hi && r.lo < q.hi {
			return true
		}
	}
	return false
}

// lazyBarriers reports whether dispatches use hazard-tracked barriers instead of
// a full barrier after every dispatch. NVIDIA only; QWEN38_LAZY_BARRIER=0
// disables it.
func (b *Backend) lazyBarriers() bool {
	if b.lazyMode == 0 {
		b.lazyMode = 2
		if b.vendorID == vendorNVIDIA && os.Getenv("QWEN38_LAZY_BARRIER") != "0" {
			b.lazyMode = 1
		}
	}
	return b.lazyMode == 1
}

// needsBarrier reports whether a dispatch with these bindings conflicts with
// work recorded since the last barrier. writable[i] says whether binding i may
// be written (missing entries are treated as writable).
func (b *Backend) needsBarrier(bindings []*buffer, writable []bool) bool {
	for i, x := range bindings {
		if x == nil {
			continue
		}
		r := rangeOf(x)
		if overlapsAny(b.pendingW, r) {
			return true // read-after-write or write-after-write
		}
		w := i >= len(writable) || writable[i]
		if w && overlapsAny(b.pendingR, r) {
			return true // write-after-read
		}
	}
	return false
}

// trackDispatch records the ranges a dispatch reads and writes.
func (b *Backend) trackDispatch(bindings []*buffer, writable []bool) {
	for i, x := range bindings {
		if x == nil {
			continue
		}
		if i >= len(writable) || writable[i] {
			b.pendingW = append(b.pendingW, rangeOf(x))
		} else {
			b.pendingR = append(b.pendingR, rangeOf(x))
		}
	}
}

// recordFullBarrier records a compute -> compute/transfer memory barrier and
// clears the hazard state.
func (b *Backend) recordFullBarrier() {
	barrier := memoryBarrier{
		sType:         vkStructureMemoryBarrier,
		srcAccessMask: vkAccessShaderRead | vkAccessShaderWrite,
		dstAccessMask: vkAccessShaderRead | vkAccessShaderWrite | vkAccessTransferRead | vkAccessTransferWrite,
	}
	vkCall(b.vk.CmdPipelineBarrier, b.cmdBuffer, vkPipelineStageCompute, vkPipelineStageCompute|vkPipelineStageTransfer, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	b.pendingW = b.pendingW[:0]
	b.pendingR = b.pendingR[:0]
}

// flushHazards makes all dispatches recorded since the last barrier complete and
// visible before a non-dispatch command (copy, fill, transfer) or the end of
// the batch. No-op when nothing is pending.
func (b *Backend) flushHazards() {
	if !b.recording || (len(b.pendingW) == 0 && len(b.pendingR) == 0) {
		return
	}
	b.recordFullBarrier()
}

// pipelineWritable returns the per-binding writability parsed from a
// pipeline's SPIR-V.
func (b *Backend) pipelineWritable(name string) []bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.pipeWritable[name]
}

// spirvWritableBindings returns, per descriptor binding of a SPIR-V module,
// whether the shader may write it: false only for buffers decorated
// NonWritable (GLSL readonly). Unknown bindings are reported writable.
func spirvWritableBindings(code []byte) []bool {
	if len(code) < 20 || len(code)%4 != 0 {
		return nil
	}
	w := make([]uint32, len(code)/4)
	for i := range w {
		w[i] = binary.LittleEndian.Uint32(code[4*i:])
	}
	if w[0] != 0x07230203 {
		return nil
	}
	const (
		opTypePointer    = 32
		opVariable       = 59
		opDecorate       = 71
		opMemberDecorate = 72
		decNonWritable   = 24
		decBinding       = 33
	)
	binding := map[uint32]uint32{}
	nwVar := map[uint32]bool{}
	nwStruct := map[uint32]bool{}
	varType := map[uint32]uint32{}
	pointee := map[uint32]uint32{}
	for i := 5; i < len(w); {
		n := int(w[i] >> 16)
		op := w[i] & 0xffff
		if n == 0 || i+n > len(w) {
			break
		}
		switch op {
		case opDecorate:
			if n >= 3 {
				switch w[i+2] {
				case decBinding:
					if n >= 4 {
						binding[w[i+1]] = w[i+3]
					}
				case decNonWritable:
					nwVar[w[i+1]] = true
				}
			}
		case opMemberDecorate:
			if n >= 4 && w[i+3] == decNonWritable {
				nwStruct[w[i+1]] = true
			}
		case opTypePointer:
			if n >= 4 {
				pointee[w[i+1]] = w[i+3]
			}
		case opVariable:
			if n >= 3 {
				varType[w[i+2]] = w[i+1]
			}
		}
		i += n
	}
	maxB := -1
	for _, bnd := range binding {
		if int(bnd) > maxB {
			maxB = int(bnd)
		}
	}
	out := make([]bool, maxB+1)
	for i := range out {
		out[i] = true
	}
	for v, bnd := range binding {
		if nwVar[v] || nwStruct[pointee[varType[v]]] {
			out[bnd] = false
		}
	}
	return out
}
