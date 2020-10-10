package gocl

import (
	"errors"

	"github.com/Nv7-Github/go-cl"
)

// Execute executes a OpenCL program on the given arguments
func Execute(program string, programName string, platform, device int, progArgs ...interface{}) ([]float32, error) {
	platforms, err := cl.GetPlatforms()
	if err != nil {
		return nil, err
	}

	p := platforms[platform]

	devices, err := p.GetDevices(cl.DeviceTypeAll)
	if err != nil {
		return nil, err
	}

	d := devices[device]

	ctx, err := cl.CreateContext([]*cl.Device{d})
	if err != nil {
		return nil, err
	}

	queue, err := ctx.CreateCommandQueue(d, 0)
	if err != nil {
		return nil, err
	}

	prog, err := ctx.CreateProgramWithSource([]string{program})
	if err != nil {
		return nil, err
	}

	err = prog.BuildProgram(nil, "")
	if err != nil {
		return nil, err
	}

	kernel, err := prog.CreateKernel(programName)
	if err != nil {
		return nil, err
	}

	arrlength := -1
	args := make([]interface{}, 0)
	for _, progArg := range progArgs {
		switch progArg.(type) {
		default:
			return nil, errors.New("Argument must be []float32, int32, float32, or uint32")
		case []float32:
			arg, err := ctx.CreateEmptyBuffer(cl.MemReadOnly, 4*len(progArg.([]float32)))
			if err != nil {
				return nil, err
			}

			_, err = queue.EnqueueWriteBufferFloat32(arg, true, 0, progArg.([]float32), nil)
			if err != nil {
				return nil, err
			}

			args = append(args, arg)

			if arrlength == -1 {
				arrlength = len(progArg.([]float32))
			} else if arrlength != len(progArg.([]float32)) {
				return nil, errors.New("All arrays must be the same length")
			}
		case int32:
			args = append(args, progArg)
		case uint32:
			args = append(args, progArg)
		case float32:
			args = append(args, progArg)
		}
	}
	if arrlength == -1 {
		return nil, errors.New("No arrays passed as input")
	}

	out, err := ctx.CreateEmptyBuffer(cl.MemReadOnly, 4*arrlength)
	if err != nil {
		return nil, err
	}

	args = append(args, out)

	err = kernel.SetArgs(args...)
	if err != nil {
		return nil, err
	}

	local, err := kernel.WorkGroupSize(d)
	if err != nil {
		return nil, err
	}

	global := arrlength
	di := arrlength % local
	if di != 0 {
		global += local - di
	}

	_, err = queue.EnqueueNDRangeKernel(kernel, nil, []int{global}, []int{local}, nil)
	if err != nil {
		return nil, err
	}

	err = queue.Finish()
	if err != nil {
		return nil, err
	}

	results := make([]float32, arrlength)
	_, err = queue.EnqueueReadBufferFloat32(args[len(args)-1].(*cl.MemObject), true, 0, results, nil)
	if err != nil {
		return nil, err
	}
	return results, nil
}
