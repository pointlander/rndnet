// Copyright 2021 The rndnet Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
)

// Find lfsr
var LFSR = flag.Bool("lfsr", false, "find a lfsr")

// LFSRMask is a LFSR mask with a maximum period
const LFSRMask = 0x80000057

// Rand is a random number generator
type Rand uint32

// Int63 returns a random float32 between 0 and 1
func (r *Rand) Float32() float32 {
	lfsr := *r
	if (lfsr & 1 == 1) {
		lfsr = (lfsr >> 1) ^ LFSRMask
	} else {
		lfsr = lfsr >> 1
	}
	*r = lfsr
	return float32(lfsr) / ((1 << 32) - 1)
}

// Uint32 returns a random uint32
func (r *Rand) Uint32() uint32 {
	lfsr := *r
	if (lfsr & 1 == 1) {
		lfsr = (lfsr >> 1) ^ LFSRMask
	} else {
		lfsr = lfsr >> 1
	}
	*r = lfsr
	return uint32(lfsr)
}

// Layer is a neural network layer
type Layer struct {
	Weights []float32
	Biases 	[]float32
	Rand 	Rand
	Shift 	int
}

// Network is a neural network
type Network []Layer

// Infernce performs inference on a neural network
func (n Network) Inference(inputs, outputs []float32) {
	last := len(n) - 1
	for i, layer := range n {
		rnd, values, factor := layer.Rand, make([]float32, len(inputs)), float32(math.Sqrt(2 / float64(len(inputs))))
		for j, weight := range layer.Weights {
			sum, index := layer.Biases[j], rnd.Uint32() >> layer.Shift
			
			for k, input := range inputs {
				if k == int(index) {
					sum += input * weight
				} else {
					sum += input * (2 * rnd.Float32() - 1) * factor
				}
			}
			e := float32(math.Exp(float64(sum)))
			values[j] = e/(e+1)
		}
		if i == last {
			copy(outputs, values)
		} else {
			inputs = values
		}
	}
}

func main() {
	flag.Parse()

	if *LFSR {
		// https://en.wikipedia.org/wiki/Linear-feedback_shift_register
		// https://users.ece.cmu.edu/~koopman/lfsr/index.html
		count, polynomial := 0, uint32(0x80000000)
		for polynomial != 0 {
			lfsr, period := uint32(1), uint32(0)
			for {
				lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & polynomial)
				period++
				if lfsr == 1 {
					break
				}
			}
			fmt.Printf("%v period=%v\n", count, period)
			if period == math.MaxUint32 {
				fmt.Printf("%x\n", polynomial)
				return
			}
			count++
			polynomial++
		}
		return
	}

	rnd := Rand(0x55555555)
	var network Network
	layer := Layer {
		Weights: make([]float32, 4),
		Biases: make([]float32, 4),
		Rand: Rand(0x55555555),
		Shift: 31 - 2,
	}
	factor := float32(math.Sqrt(2 / float64(4)))
	for i := range layer.Weights {
		layer.Weights[i] = (2 * rnd.Float32() - 1) * factor
	}
	network = append(network, layer)
	layer = Layer {
		Weights: make([]float32, 3),
		Biases: make([]float32, 3),
		Rand: Rand(0x55555555),
		Shift: 31 - 2,
	}
	factor = float32(math.Sqrt(2 / float64(4)))
	for i := range layer.Weights {
		layer.Weights[i] = (2 * rnd.Float32() - 1) * factor
	}
	network = append(network, layer)
	inputs, outputs := []float32{.1, .1, .1, .1}, make([]float32, 3)
	network.Inference(inputs, outputs)
	fmt.Println(outputs)
}

