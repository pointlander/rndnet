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

// SharedLayer is a neural network layer with shared weights
type SharedLayer struct {
	Rows    int
	Columns int
	Weights []float32
	Rand    Rand
}

// SharedNetwork is a neural network with shared weights
type SharedNetwork []SharedLayer

// Inference performs inference on a neural network
func (n SharedNetwork) Inference(inputs, outputs []float32) {
	last := len(n) - 1
	for i, layer := range n {
		rnd := layer.Rand
		columns := len(outputs)
		if i < len(n)-1 {
			columns = n[i+1].Columns
		}
		mask, values :=
			uint32((1<<bits.TrailingZeros(uint(len(layer.Weights))))-1),
			make([]float32, columns)
		for j := 0; j < layer.Rows; j++ {
			sum := layer.Weights[rnd.Uint32()&mask]
			for k := 0; k < layer.Columns; k++ {
				sum += inputs[k] * layer.Weights[rnd.Uint32()&mask]
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
func (n SharedNetwork) Copy() SharedNetwork {
	var network SharedNetwork
	for _, layer := range n {
		l := SharedLayer{
			Rows:    layer.Rows,
			Columns: layer.Columns,
			Weights: make([]float32, len(layer.Weights)),
			Rand:    layer.Rand,
		}
		copy(l.Weights, layer.Weights)
		network = append(network, l)
	}
	return network
}

// SharedNetworkModel is the real network with shared weights
func SharedNetworkModel() {
	rnd := Rand(LFSRInit)
	type Genome struct {
		Network SharedNetwork
		Fitness float32
	}
	var genomes []Genome
	addNetwork := func(i int) {
		var network SharedNetwork
		layer := SharedLayer{
			Rows:    4,
			Columns: 4,
			Weights: make([]float32, 4),
			Rand:    Rand(LFSRInit + i),
		}
		factor := float32(math.Sqrt(2 / float64(4)))
		for i := range layer.Weights {
			layer.Weights[i] = (2*rnd.Float32() - 1) * factor
		}
		network = append(network, layer)

		layer = SharedLayer{
			Rows:    3,
			Columns: 4,
			Weights: make([]float32, 4),
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
			layer, valueA, valueB :=
				rnd.Uint32()&1, rnd.Uint32()&3, rnd.Uint32()&3
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
			layerA.Weights[valueA], layerB.Weights[valueB] =
				layerB.Weights[valueB], layerA.Weights[valueA]
			genomes = append(genomes, Genome{
				Network: networkA,
			})
			genomes = append(genomes, Genome{
				Network: networkB,
			})
		}

		for i := 0; i < NumGenomes; i++ {
			layer, value :=
				rnd.Uint32()&1, rnd.Uint32()&3
			network := genomes[i].Network.Copy()
			l := network[layer]
			if layer == 1 {
				for value > 2 {
					value = rnd.Uint32() & 3
				}
			}
			l.Weights[value] += ((2 * rnd.Float32()) - 1)
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
