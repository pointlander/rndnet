// Copyright 2021 The rndnet Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/bits"
	"math/cmplx"
	"sort"

	"github.com/pointlander/datum/iris"
)

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

// ComplexNetworkModel is the complex network
func ComplexNetworkModel(seed int) float64 {
	rnd := Rand(LFSRInit + seed)
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
			Rand:    Rand(LFSRInit + i + seed + NumGenomes),
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
			Rand:    Rand(LFSRInit + i + seed + 2*NumGenomes),
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
	quality := float64(misses) / float64(total)
	fmt.Println(genomes[0].Fitness, quality)
	return quality
}
