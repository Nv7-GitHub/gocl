package gocl

import (
	"math/rand"
	"testing"
	"unsafe"
)

type Data struct {
	x int32
	y int32
}

func TestArrBufferStruct(t *testing.T) {
	var kernelSource = `
typedef struct data {
	int x;
	int y;
};

__kernel void square(
	__global struct data * in,
	const unsigned int count,
	__global int * out) 
{
	int i = get_global_id(0);
	if (i < count)
		out[i] = in[i].x * in[i].y;
}
`
	input := make([]Data, 1024)
	for i := 0; i < len(input); i++ {
		input[i].x = rand.Int31n(100)
		input[i].y = rand.Int31n(100)
	}

	prog, err := NewProgram(kernelSource, "square", 0, 0, DeviceTypeAll)
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

	err = prog.Execute([]int{len(input)}, []int{1}, buff, length, out)
	if err != nil {
		t.Fatal(err)
	}

	results := make([]int32, len(input))
	err = out.GetDataArr(int(unsafe.Sizeof(results[0])), unsafe.Pointer(&results[0]))
	if err != nil {
		t.Fatal(err)
	}

	for i, val := range input {
		if results[i] != val.x*val.y {
			t.Fatalf("Square not calculated correctly. Expected: %d, Got: %d", val.x*val.y, results[i])
		}
	}
}
