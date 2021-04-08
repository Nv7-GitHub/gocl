package gocl

import (
	"math/rand"
	"testing"
)

func TestFloat32Arr(t *testing.T) {
	var kernelSource = `
__kernel void square(
	__global float * in,
	const unsigned int count,
	__global float * out) 
{
	int i = get_global_id(0);
	if (i < count)
		out[i] = in[i] * in[i];
}
`
	input := make([]float32, 1024)
	for i := 0; i < len(input); i++ {
		input[i] = rand.Float32()
	}

	prog, err := NewProgram(kernelSource, "square", 0, 0)
	if err != nil {
		t.Fatal(err)
	}

	buff, err := prog.NewBufferFromFloat32Arr(input)
	if err != nil {
		t.Fatal(err)
	}

	out, err := prog.NewEmptyFloat32Arr(len(input))
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

	results, err := out.GetDataFloat32Arr()
	if err != nil {
		t.Fatal(err)
	}

	for i, val := range input {
		if results[i] != val*val {
			t.Fatalf("Square not calculated correctly. Expected: %f, Got: %f", val*val, results[i])
		}
	}
}

func BenchmarkFloat32Arr(t *testing.B) {
	var kernelSource = `
__kernel void square(
	__global float * in,
	const unsigned int count,
	__global float * out) 
{
	int i = get_global_id(0);
	if (i < count)
		out[i] = in[i] * in[i];
}
`
	input := make([]float32, t.N)
	for i := 0; i < len(input); i++ {
		input[i] = rand.Float32()
	}

	t.ResetTimer()

	prog, err := NewProgram(kernelSource, "square", 0, 0)
	if err != nil {
		t.Fatal(err)
	}

	buff, err := prog.NewBufferFromFloat32Arr(input)
	if err != nil {
		t.Fatal(err)
	}
	defer buff.Cleanup()

	out, err := prog.NewEmptyFloat32Arr(len(input))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Cleanup()

	length, err := NewConst(len(input))
	if err != nil {
		t.Fatal(err)
	}

	err = prog.Execute([]int{len(input)}, buff, length, out)
	if err != nil {
		t.Fatal(err)
	}

	results, err := out.GetDataFloat32Arr()
	if err != nil {
		t.Fatal(err)
	}

	for i, val := range input {
		if results[i] != val*val {
			t.Fatalf("Square not calculated correctly. Expected: %f, Got: %f", val*val, results[i])
		}
	}
}
