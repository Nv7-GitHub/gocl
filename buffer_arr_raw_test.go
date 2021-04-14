package gocl

import (
	"math/rand"
	"testing"
	"unsafe"
)

func TestArrBuffer(t *testing.T) {
	var kernelSource = `
__kernel void square(
	__global int * in,
	const unsigned int count,
	__global int * out) 
{
	int i = get_global_id(0);
	if (i < count)
		out[i] = in[i] * in[i];
}
`
	input := make([]int32, 1024)
	for i := 0; i < len(input); i++ {
		input[i] = rand.Int31()
	}

	prog, err := NewProgram(kernelSource, "square", 0, 0)
	if err != nil {
		t.Fatal(err)
	}

	ptr := unsafe.Pointer(&input[0])
	buff, err := prog.NewBufferFromArr(int(unsafe.Sizeof(input[0])), ptr, len(input))
	if err != nil {
		t.Fatal(err)
	}

	empty := make([]int32, len(input))
	emptyPtr := unsafe.Pointer(&empty[0])
	out, err := prog.NewBufferFromArr(int(unsafe.Sizeof(empty[0])), emptyPtr, len(empty))
	if err != nil {
		t.Fatal(err)
	}

	length, err := NewConst(len(input))
	if err != nil {
		t.Fatal(err)
	}

	err = prog.Execute([]int{len(input)}, buff, length, out)
	if err != nil {
		t.Fatal(err)
	}

	results := make([]int32, len(input))
	err = out.GetDataArr(int(unsafe.Sizeof(results[0])), unsafe.Pointer(&results[0]))
	if err != nil {
		t.Fatal(err)
	}

	for i, val := range input {
		if results[i] != val*val {
			t.Fatalf("Square not calculated correctly. Expected: %d, Got: %d", val*val, results[i])
		}
	}
}
