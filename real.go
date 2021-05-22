// Copyright 2021 The rndnet Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/bits"
	"sort"

	"github.com/pointlander/datum/iris"
)

// RealLayer is a neural network layer
type RealLayer struct {
	Columns int
	Weights []float32
	Biases  []float32
	Rand    Rand
}

// RealNetwork is a neural network
type RealNetwork []RealLayer

// Inference performs inference on a neural network
func (n RealNetwork) Inference(inputs, outputs []float32) {
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
func (n RealNetwork) Copy() RealNetwork {
	var network RealNetwork
	for _, layer := range n {
		l := RealLayer{
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

// RealNetworkModel is the real network model
func RealNetworkModel() {
	rnd := Rand(LFSRInit)
	type Genome struct {
		Network RealNetwork
		Fitness float32
	}
	var genomes []Genome
	addNetwork := func(i int) {
		var network RealNetwork
		layer := RealLayer{
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

		layer = RealLayer{
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
