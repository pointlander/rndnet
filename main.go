// Copyright 2021 The rndnet Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/bits"
	"math/cmplx"
	"sort"

	"github.com/pointlander/datum/iris"
)

var (
	// LFSR find lfsr
	LFSR = flag.Bool("lfsr", false, "find a lfsr")
	// Real uses the real network
	Real = flag.Bool("real", false, "real network")
	// Complex uses the complex network
	Complex = flag.Bool("complex", false, "complex network")
)

const (
	// LFSRMask is a LFSR mask with a maximum period
	LFSRMask = 0x80000057
	// LFSRInit is an initial LFSR state
	LFSRInit = 0x55555555
	// NumGenomes is the number of genomes
	NumGenomes = 256
)

// Rand is a random number generator
type Rand uint32

// Float32 returns a random float32 between 0 and 1
func (r *Rand) Float32() float32 {
	lfsr := *r
	if lfsr&1 == 1 {
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
	if lfsr&1 == 1 {
		lfsr = (lfsr >> 1) ^ LFSRMask
	} else {
		lfsr = lfsr >> 1
	}
	*r = lfsr
	return uint32(lfsr)
}

// Layer is a neural network layer
type Layer struct {
	Columns int
	Weights []float32
	Biases  []float32
	Rand    Rand
}

// Network is a neural network
type Network []Layer

// Inference performs inference on a neural network
func (n Network) Inference(inputs, outputs []float32) {
	last := len(n) - 1
	for i, layer := range n {
		rnd := layer.Rand
		columns := len(outputs)
		if i < len(n)-1 {
			columns = n[i+1].Columns
		}
		mask, values, factor :=
			uint32((1<<bits.TrailingZeros(uint(layer.Columns)))-1),
			make([]float32, columns),
			float32(math.Sqrt(2/float64(columns)))
		for j, weight := range layer.Weights {
			sum, index := layer.Biases[j], rnd.Uint32()&mask
			for k, input := range inputs {
				if k == int(index) {
					sum += input * weight
				} else {
					sum += input * (2*rnd.Float32() - 1) * factor
				}
			}
			e := float32(math.Exp(float64(sum)))
			values[j] = e / (e + 1)
		}
		if i == last {
			copy(outputs, values)
		} else {
			inputs = values
		}
	}
}

// Copy copies a network
func (n Network) Copy() Network {
	var network Network
	for _, layer := range n {
		l := Layer{
			Columns: layer.Columns,
			Weights: make([]float32, len(layer.Weights)),
			Biases:  make([]float32, len(layer.Biases)),
			Rand:    layer.Rand,
		}
		copy(l.Weights, layer.Weights)
		copy(l.Biases, layer.Biases)
		network = append(network, l)
	}
	return network
}

// ComplexLayer is a complex neural network layer
type ComplexLayer struct {
	Columns int
	Weights []complex64
	Biases  []complex64
	Rand    Rand
}

// ComplexNetwork is a complex neural network
type ComplexNetwork []ComplexLayer

// Inference performs inference on a neural network
func (n ComplexNetwork) Inference(inputs, outputs []complex64) {
	last := len(n) - 1
	for i, layer := range n {
		rnd := layer.Rand
		columns := len(outputs)
		if i < len(n)-1 {
			columns = n[i+1].Columns
		}
		mask, values, factor :=
			uint32((1<<bits.TrailingZeros(uint(layer.Columns)))-1),
			make([]complex64, columns),
			float32(math.Sqrt(2/float64(columns)))
		for j, weight := range layer.Weights {
			sum, index := layer.Biases[j], rnd.Uint32()&mask
			for k, input := range inputs {
				if k == int(index) {
					sum += input * weight
				} else {
					sum += input * complex((2*rnd.Float32()-1)*factor, (2*rnd.Float32()-1)*factor)
				}
			}
			e := complex64(cmplx.Exp(complex128(sum)))
			values[j] = e / (e + 1)
		}
		if i == last {
			copy(outputs, values)
		} else {
			inputs = values
		}
	}
}

// Copy copies a network
func (n ComplexNetwork) Copy() ComplexNetwork {
	var network ComplexNetwork
	for _, layer := range n {
		l := ComplexLayer{
			Columns: layer.Columns,
			Weights: make([]complex64, len(layer.Weights)),
			Biases:  make([]complex64, len(layer.Biases)),
			Rand:    layer.Rand,
		}
		copy(l.Weights, layer.Weights)
		copy(l.Biases, layer.Biases)
		network = append(network, l)
	}
	return network
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
	} else if *Real {
		RealNetwork()
		return
	} else if *Complex {
		ComplexNetworkModel()
		return
	}
}

// RealNetwork is the real network
func RealNetwork() {
	rnd := Rand(LFSRInit)
	type Genome struct {
		Network Network
		Fitness float32
	}
	var genomes []Genome
	addNetwork := func(i int) {
		var network Network
		layer := Layer{
			Columns: 4,
			Weights: make([]float32, 4),
			Biases:  make([]float32, 4),
			Rand:    Rand(LFSRInit + i),
		}
		factor := float32(math.Sqrt(2 / float64(4)))
		for i := range layer.Weights {
			layer.Weights[i] = (2*rnd.Float32() - 1) * factor
		}
		network = append(network, layer)

		layer = Layer{
			Columns: 4,
			Weights: make([]float32, 3),
			Biases:  make([]float32, 3),
			Rand:    Rand(LFSRInit + i),
		}
		factor = float32(math.Sqrt(2 / float64(3)))
		for i := range layer.Weights {
			layer.Weights[i] = (2*rnd.Float32() - 1) * factor
		}
		network = append(network, layer)

		genomes = append(genomes, Genome{
			Network: network,
		})
	}
	for i := 0; i < NumGenomes; i++ {
		addNetwork(i)
	}

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	inputs, outputs := make([]float32, 4), make([]float32, 3)
	get := func() int {
		for {
			for i, genome := range genomes {
				if rnd.Float32() > genome.Fitness {
					return i
				}
			}
		}
	}
	i := 0
	for {
		for j, genome := range genomes {
			sum := float32(0)
			for _, flower := range datum.Fisher {
				for k, value := range flower.Measures {
					inputs[k] = float32(value)
				}
				genome.Network.Inference(inputs, outputs)
				expected := make([]float32, 3)
				expected[iris.Labels[flower.Label]] = 1
				loss := float32(0)
				for l, output := range outputs {
					diff := expected[l] - output
					loss += diff * diff
				}
				loss = float32(math.Sqrt(float64(loss)))
				sum += loss
			}
			sum /= float32(len(datum.Fisher)) * float32(math.Sqrt(3))
			genomes[j].Fitness = sum
		}
		sort.Slice(genomes, func(i, j int) bool {
			return genomes[i].Fitness < genomes[j].Fitness
		})
		genomes = genomes[:NumGenomes]
		fmt.Println(i, genomes[0].Fitness)
		i++
		if i > 127 {
			break
		}

		for i := 0; i < 256; i++ {
			a, b := get(), get()
			layer, vector, valueA, valueB :=
				rnd.Uint32()&1, rnd.Uint32()&1, rnd.Uint32()&3, rnd.Uint32()&3
			networkA, networkB :=
				genomes[a].Network.Copy(), genomes[b].Network.Copy()
			layerA, layerB := networkA[layer], networkB[layer]
			if layer == 1 {
				for valueA > 2 {
					valueA = rnd.Uint32() & 3
				}
				for valueB > 2 {
					valueB = rnd.Uint32() & 3
				}
			}
			if vector == 0 {
				layerA.Weights[valueA], layerB.Weights[valueB] =
					layerB.Weights[valueB], layerA.Weights[valueA]
			} else {
				layerA.Biases[valueA], layerB.Biases[valueB] =
					layerB.Biases[valueB], layerA.Biases[valueA]
			}
			genomes = append(genomes, Genome{
				Network: networkA,
			})
			genomes = append(genomes, Genome{
				Network: networkB,
			})
		}

		for i := 0; i < NumGenomes; i++ {
			layer, vector, value :=
				rnd.Uint32()&1, rnd.Uint32()&1, rnd.Uint32()&3
			network := genomes[i].Network.Copy()
			l := network[layer]
			if layer == 1 {
				for value > 2 {
					value = rnd.Uint32() & 3
				}
			}
			if vector == 0 {
				l.Weights[value] += ((2 * rnd.Float32()) - 1)
			} else {
				l.Biases[value] += ((2 * rnd.Float32()) - 1)
			}
			genomes = append(genomes, Genome{
				Network: network,
			})
		}
	}

	network := genomes[0].Network
	misses, total := 0, 0
	for _, flower := range datum.Fisher {
		for k, value := range flower.Measures {
			inputs[k] = float32(value)
		}
		network.Inference(inputs, outputs)
		max, index := float32(0), 0
		for j, output := range outputs {
			if output > max {
				max, index = output, j
			}
		}
		if index != iris.Labels[flower.Label] {
			misses++
		}
		total++
	}
	fmt.Println(float64(misses) / float64(total))
}

// ComplexNetworkModel is the complex network
func ComplexNetworkModel() {
	rnd := Rand(LFSRInit)
	type Genome struct {
		Network ComplexNetwork
		Fitness float32
	}
	var genomes []Genome
	addNetwork := func(i int) {
		var network ComplexNetwork
		layer := ComplexLayer{
			Columns: 4,
			Weights: make([]complex64, 4),
			Biases:  make([]complex64, 4),
			Rand:    Rand(LFSRInit + i),
		}
		factor := float32(math.Sqrt(2 / float64(4)))
		for i := range layer.Weights {
			layer.Weights[i] = complex((2*rnd.Float32()-1)*factor, (2*rnd.Float32()-1)*factor)
		}
		network = append(network, layer)

		layer = ComplexLayer{
			Columns: 4,
			Weights: make([]complex64, 3),
			Biases:  make([]complex64, 3),
			Rand:    Rand(LFSRInit + i),
		}
		factor = float32(math.Sqrt(2 / float64(3)))
		for i := range layer.Weights {
			layer.Weights[i] = complex((2*rnd.Float32()-1)*factor, (2*rnd.Float32()-1)*factor)
		}
		network = append(network, layer)

		genomes = append(genomes, Genome{
			Network: network,
		})
	}
	for i := 0; i < NumGenomes; i++ {
		addNetwork(i)
	}

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	inputs, outputs := make([]complex64, 4), make([]complex64, 3)
	get := func() int {
		for {
			for i, genome := range genomes {
				if rnd.Float32() > genome.Fitness {
					return i
				}
			}
		}
	}
	i := 0
	for {
		for j, genome := range genomes {
			sum := complex64(0)
			for _, flower := range datum.Fisher {
				for k, value := range flower.Measures {
					inputs[k] = complex(float32(value), 0)
				}
				genome.Network.Inference(inputs, outputs)
				expected := make([]complex64, 3)
				expected[iris.Labels[flower.Label]] = 1
				loss := complex64(0)
				for l, output := range outputs {
					diff := expected[l] - output
					loss += diff * diff
				}
				loss = complex64(cmplx.Sqrt(complex128(loss)))
				sum += loss
			}
			sum /= complex(float32(len(datum.Fisher)), 0) * complex(float32(math.Sqrt(3)), 0)
			genomes[j].Fitness = float32(cmplx.Abs(complex128(sum)))
		}
		sort.Slice(genomes, func(i, j int) bool {
			if math.IsNaN(float64(genomes[i].Fitness)) {
				return false
			}
			if math.IsNaN(float64(genomes[j].Fitness)) {
				return true
			}
			return genomes[i].Fitness < genomes[j].Fitness
		})
		genomes = genomes[:NumGenomes]
		fmt.Println(i, genomes[0].Fitness)
		i++
		if i > 127 {
			break
		}

		for i := 0; i < 256; i++ {
			a, b := get(), get()
			layer, vector, valueA, valueB :=
				rnd.Uint32()&1, rnd.Uint32()&1, rnd.Uint32()&3, rnd.Uint32()&3
			networkA, networkB :=
				genomes[a].Network.Copy(), genomes[b].Network.Copy()
			layerA, layerB := networkA[layer], networkB[layer]
			if layer == 1 {
				for valueA > 2 {
					valueA = rnd.Uint32() & 3
				}
				for valueB > 2 {
					valueB = rnd.Uint32() & 3
				}
			}
			if vector == 0 {
				layerA.Weights[valueA], layerB.Weights[valueB] =
					layerB.Weights[valueB], layerA.Weights[valueA]
			} else {
				layerA.Biases[valueA], layerB.Biases[valueB] =
					layerB.Biases[valueB], layerA.Biases[valueA]
			}
			genomes = append(genomes, Genome{
				Network: networkA,
			})
			genomes = append(genomes, Genome{
				Network: networkB,
			})
		}

		for i := 0; i < NumGenomes; i++ {
			layer, vector, value, part :=
				rnd.Uint32()&1, rnd.Uint32()&1, rnd.Uint32()&3, rnd.Uint32()&1
			network := genomes[i].Network.Copy()
			l := network[layer]
			if layer == 1 {
				for value > 2 {
					value = rnd.Uint32() & 3
				}
			}
			if vector == 0 {
				if part == 0 {
					l.Weights[value] += complex(((2 * rnd.Float32()) - 1), 0)
				} else {
					l.Weights[value] += complex(0, ((2 * rnd.Float32()) - 1))
				}
			} else {
				if part == 0 {
					l.Biases[value] += complex(((2 * rnd.Float32()) - 1), 0)
				} else {
					l.Biases[value] += complex(0, ((2 * rnd.Float32()) - 1))
				}
			}
			genomes = append(genomes, Genome{
				Network: network,
			})
		}
	}

	network := genomes[0].Network
	misses, total := 0, 0
	for _, flower := range datum.Fisher {
		for k, value := range flower.Measures {
			inputs[k] = complex(float32(value), 0)
		}
		network.Inference(inputs, outputs)
		max, index := float32(0), 0
		for j, output := range outputs {
			out := float32(cmplx.Abs(complex128(output)))
			if out > max {
				max, index = out, j
			}
		}
		if index != iris.Labels[flower.Label] {
			misses++
		}
		total++
	}
	fmt.Println(float64(misses) / float64(total))
}
