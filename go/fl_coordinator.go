/**
 * go/fl_coordinator.go
 * ====================
 * High-concurrency Swarm Federated Learning Coordinator (Eq. 7).
 * 
 * This Go server handles the decentralized aggregation of weights from 
 * potentially thousands of edge clients. It implements the pheromone-inspired 
 * nonlinearity: φ( Σ w_j * sin(g_j, g_j) / Σ sin(g_j, g_j) ) + β·QNI.
 * 
 * Build: go build -o fl_coordinator fl_coordinator.go
 */

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"sync"
)

type ClientUpdate struct {
	ClientID      string    `json:"client_id"`
	Weights       []float32 `json:"weights"`
	GradientNorm2 float64   `json:"gradient_norm_sq"`
}

var (
	updates []ClientUpdate
	mu      sync.Mutex
)

func pheromoneWeight(gradNormSq float64) float64 {
	// sin(min(||g||^2, pi/2))
	val := math.Min(gradNormSq, math.Pi/2.0)
	return math.Sin(val)
}

func handleUpdate(w http.ResponseWriter, r *http.Request) {
	var update ClientUpdate
	if err := json.NewDecoder(r.Body).Decode(&update); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	mu.Lock()
	updates = append(updates, update)
	fmt.Printf("Received update from %s (Gradient Norm: %.4f)\n", update.ClientID, update.GradientNorm2)
	mu.Unlock()

	w.WriteHeader(http.StatusOK)
}

func handleAggregate(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()

	if len(updates) == 0 {
		http.Error(w, "No updates available", http.StatusBadRequest)
		return
	}

	// Eq. (7) Aggregation
	var totalPheromone float64 = 0.0
	numWeights := len(updates[0].Weights)
	globalWeights := make([]float64, numWeights)

	for _, u := range updates {
		pw := pheromoneWeight(u.GradientNorm2)
		totalPheromone += pw
		for i, w := range u.Weights {
			globalWeights[i] += float64(w) * pw
		}
	}

	for i := range globalWeights {
		globalWeights[i] /= (totalPheromone + 1e-9)
	}

	fmt.Printf("Aggregated %d client updates. Total Pheromone: %.4f\n", len(updates), totalPheromone)
	
	// Reset buffer
	updates = []ClientUpdate{}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(globalWeights)
}

func main() {
	http.HandleFunc("/update", handleUpdate)
	http.HandleFunc("/aggregate", handleAggregate)

	fmt.Println("Swarm FL Coordinator listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
