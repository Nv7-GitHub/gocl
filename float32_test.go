package gocl

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestSquares(t *testing.T) {
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

	results, err := Execute(kernelSource, "square", 0, 0, input, uint32(len(input)))
	if err != nil {
		t.Fatal(err)
	}

	for i, val := range input {
		if results[i] != val*val {
			t.Fatalf("Square not calculated correctly. Expected: %f, Got: %f", val*val, results[i])
		}
	}
}

func TestAddition(t *testing.T) {
	var kernelSource = `
__kernel void add(
	__global float * a,
	__global float *b,
	const unsigned int count,
	__global float * out) 
{
	int i = get_global_id(0);
	if (i < count)
		out[i] = a[i] + b[i];
}
`
	a := make([]float32, 1024)
	b := make([]float32, 1024)
	for i := 0; i < len(a); i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
	}

	results, err := Execute(kernelSource, "add", 0, 0, a, b, uint32(len(a)))
	if err != nil {
		t.Fatal(err)
	}

	for i, val := range a {
		diff := results[i] - (val + b[i])
		if diff < 0 {
			diff *= -1
		}
		if diff > 0.00000000000001 {
			t.Fatalf("Addition not calculated correctly. Expected: %f, Got: %f. It was %f off.", val+b[i], results[i], diff)
		}
	}
}

func ExampleExecute_squares() {
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
	input := []float32{0.5, 1.5, 1.0, 2.0}

	// Parameters: OpenCL Code, Name of function, platform num, device num, parameters for function
	results, err := Execute(kernelSource, "square", 0, 0, input, uint32(len(input)))
	if err != nil {
		panic(err)
	}

	fmt.Println(results)
	// Output: [0.25 2.25 1 4]
}

func ExampleExecute_add() {
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
	input := []float32{0.5, 1.5, 1.0, 2.0}

	// Parameters: OpenCL Code, Name of function, platform num, device num, parameters for function
	results, err := Execute(kernelSource, "square", 0, 0, input, uint32(len(input)))
	if err != nil {
		panic(err)
	}

	fmt.Println(results)
	// Output: [0.25 2.25 1 4]
}
