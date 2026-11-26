/**
 * circuits/nqal_circuit.circom
 * ============================
 * NQAL ZKP-Learn Circuit (Eq. 5).
 * 
 * Proves that the sum of the input spike train exceeds a threshold 
 * without revealing the spike train itself.
 * 
 * Compile: circom nqal_circuit.circom --r1cs --wasm --sym
 */

pragma circom 2.0.0;

template NQALCircuit(N, T) {
    // Private inputs: The spike train (N neurons, T timesteps)
    signal input spikes[N][T];
    
    // Public input: The detection threshold (tau)
    signal input threshold;
    
    // Output: 1 if anomaly detected, 0 otherwise
    signal output anomaly;

    // 1. Ensure all spikes are binary (0 or 1)
    for (var i = 0; i < N; i++) {
        for (var t = 0; t < T; t++) {
            spikes[i][t] * (spikes[i][t] - 1) === 0;
        }
    }

    // 2. Sum the spikes
    signal sum;
    sum <== 0;
    for (var i = 0; i < N; i++) {
        for (var t = 0; t < T; t++) {
            sum <== sum + spikes[i][t];
        }
    }

    // 3. Compare sum to threshold
    // Since Circom doesn't have native > operator for signals, 
    // we use the LessThan template.
    component lt = LessThan(32);
    lt.in[0] <== threshold;
    lt.in[1] <== sum;

    // Output is 1 if sum > threshold (i.e., NOT less than)
    anomaly <== 1 - lt.out;
}

// Standard Circom LessThan comparator
template LessThan(n) {
    signal input in[2];
    signal output out;
    
    component n2b = Num2Bits(n+1);
    n2b.in <== ((1 << n) + in[0] - in[1]);
    
    out <== 1 - n2b.out[n];
}

// Standard Circom Num2Bits
template Num2Bits(n) {
    signal input in;
    signal output out[n];
    var lc1=0;
    for (var i = 0; i<n; i++) {
        out[i] <-- (in >> i) & 1;
        out[i] * (out[i] -1 ) === 0;
        lc1 += out[i] * 2**i;
    }
    lc1 === in;
}

// Instantiate for 128 neurons, 50 timesteps
component main = NQALCircuit(128, 50);
