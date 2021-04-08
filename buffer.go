package gocl

import (
	"image"
	"image/draw"
	"unsafe"

	"github.com/Nv7-Github/go-cl"
)

type Buffer struct {
	obj    *cl.MemObject
	queue  *cl.CommandQueue
	length int

	stride int
	bounds image.Rectangle
}

func (b *Buffer) val() interface{} {
	return b.obj
}

func (b *Buffer) Cleanup() {
	b.obj.Release()
}

func (p *Program) NewEmptyFloat32Arr(length int) (*Buffer, error) {
	arg, err := p.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*length)
	if err != nil {
		return nil, err
	}

	return &Buffer{
		queue:  p.queue,
		obj:    arg,
		length: length,
	}, nil
}

func (p *Program) NewBufferFromFloat32Arr(data []float32) (*Buffer, error) {
	arg, err := p.ctx.CreateEmptyBuffer(cl.MemReadWrite, 4*len(data))
	if err != nil {
		return nil, err
	}

	_, err = p.queue.EnqueueWriteBufferFloat32(arg, true, 0, data, nil)
	if err != nil {
		return nil, err
	}

	return &Buffer{
		queue:  p.queue,
		obj:    arg,
		length: len(data),
	}, nil
}

func (b *Buffer) GetDataFloat32Arr() ([]float32, error) {
	results := make([]float32, b.length)
	_, err := b.queue.EnqueueReadBufferFloat32(b.obj, true, 0, results, nil)
	if err != nil {
		return nil, err
	}

	return results, nil
}

func (p *Program) NewBufferFromImage(im image.Image) (*Buffer, error) {
	img, ok := im.(*image.RGBA)
	if !ok {
		im = image.NewRGBA(img.Bounds())
		draw.Draw(img, im.Bounds(), im, image.Point{}, draw.Src)
	}

	stride := img.Stride
	bounds := img.Bounds()

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
	clImg, err := p.ctx.CreateImage(cl.MemReadWrite|cl.MemCopyHostPtr, format, desc, img.Pix)
	if err != nil {
		return nil, err
	}
	return &Buffer{
		obj:    clImg,
		queue:  p.queue,
		bounds: bounds,
		stride: stride,
	}, nil
}

func (b *Buffer) GetDataImage() (image.Image, error) {
	final := image.NewRGBA(b.bounds)
	final.Stride = b.stride
	_, err := b.queue.EnqueueReadImage(b.obj, true, [3]int{0, 0, 0}, [3]int{final.Bounds().Dx(), final.Bounds().Dy(), 1}, b.stride, 0, final.Pix, nil)
	if err != nil {
		return nil, err
	}

	return final, nil
}

func (p *Program) NewBufferFromArr(data []interface{}) (*Buffer, error) {
	arg, err := p.ctx.CreateEmptyBuffer(cl.MemReadOnly, 4*len(data))
	if err != nil {
		return nil, err
	}

	dataPtr := unsafe.Pointer(&data[0])
	dataSize := int(unsafe.Sizeof(data[0])) * len(data)
	_, err = p.queue.EnqueueWriteBuffer(arg, true, 0, dataSize, dataPtr, nil)
	if err != nil {
		return nil, err
	}

	return &Buffer{
		queue:  p.queue,
		obj:    arg,
		length: len(data),
	}, nil
}

func (b *Buffer) GetDataArr(out []interface{}) error {
	// Out must be same length as input
	dataPtr := unsafe.Pointer(&out[0])
	dataSize := int(unsafe.Sizeof(out[0])) * len(out)
	_, err := b.queue.EnqueueReadBuffer(b.obj, true, 0, dataSize, dataPtr, nil)
	return err
}
