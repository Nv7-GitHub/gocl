package gocl

import (
	"errors"
	"image"

	"github.com/Nv7-Github/go-cl"
)

// ExecuteImage executes a OpenCL program on the given arguments, which should be an image
func ExecuteImage(program string, programName string, platform, device int, progArgs ...interface{}) (*image.RGBA, error) {
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

	args := make([]interface{}, 0)
	var rect image.Rectangle
	var stride int
	var bounds image.Rectangle
	for _, progArg := range progArgs {
		switch progArg.(type) {
		default:
			return nil, errors.New("Argument must be *image.RGBA, int32, float32, or uint32")
		case int32:
			args = append(args, progArg)
		case uint32:
			args = append(args, progArg)
		case float32:
			args = append(args, progArg)
		case *image.RGBA:
			rect = progArg.(*image.RGBA).Rect
			stride = progArg.(*image.RGBA).Stride
			bounds = progArg.(*image.RGBA).Bounds()

			format := cl.ImageFormat{
				ChannelOrder:    cl.ChannelOrderRGBA,
				ChannelDataType: cl.ChannelDataTypeUNormInt8,
			}
			desc := cl.ImageDescription{
				Type:     cl.MemObjectTypeImage2D,
				Width:    bounds.Dx(),
				Height:   bounds.Dy(),
				RowPitch: stride,
			}
			img, err := ctx.CreateImage(cl.MemReadOnly|cl.MemCopyHostPtr, format, desc, progArg.(*image.RGBA).Pix)
			if err != nil {
				return nil, err
			}

			args = append(args, img)
		}
	}

	out, err := ctx.CreateImageFromImage(cl.MemWriteOnly|cl.MemCopyHostPtr, image.NewRGBA(rect))
	if err != nil {
		return nil, err
	}

	args = append(args, out)

	err = kernel.SetArgs(args...)
	if err != nil {
		return nil, err
	}

	local, err := kernel.PreferredWorkGroupSizeMultiple(d)
	if err != nil {
		return nil, err
	}

	xSize := bounds.Dx()
	diff := xSize % local
	xSize += local - diff

	ySize := bounds.Dy()
	diff = ySize % local
	ySize += local - diff

	_, err = queue.EnqueueNDRangeKernel(kernel, nil, []int{xSize, ySize}, []int{local, local}, nil)
	if err != nil {
		return nil, err
	}

	err = queue.Finish()
	if err != nil {
		return nil, err
	}

	img := image.NewRGBA(rect)
	img.Stride = stride
	_, err = queue.EnqueueReadImage(args[len(args)-1].(*cl.MemObject), true, [3]int{0, 0, 0}, [3]int{img.Bounds().Dx(), img.Bounds().Dy(), 1}, stride, 0, img.Pix, nil)
	if err != nil {
		return nil, err
	}
	return img, nil
}
