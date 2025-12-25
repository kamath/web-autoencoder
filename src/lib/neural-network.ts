// Simple Multi-Layer Perceptron for image painting
// The network learns to map (x, y) coordinates to (r, g, b) colors

export interface NetworkConfig {
  inputSize: number
  hiddenLayers: number[]
  outputSize: number
  learningRate: number
}

export class NeuralNetwork {
  private weights: number[][][]
  private biases: number[][]
  private learningRate: number
  private activations: number[][]
  private zValues: number[][]

  constructor(config: NetworkConfig) {
    this.learningRate = config.learningRate
    this.weights = []
    this.biases = []
    this.activations = []
    this.zValues = []

    const layers = [config.inputSize, ...config.hiddenLayers, config.outputSize]

    // Initialize weights and biases with Xavier initialization
    for (let i = 0; i < layers.length - 1; i++) {
      const inputSize = layers[i]
      const outputSize = layers[i + 1]
      const scale = Math.sqrt(2 / (inputSize + outputSize))

      this.weights.push(
        Array.from({ length: outputSize }, () =>
          Array.from({ length: inputSize }, () => (Math.random() * 2 - 1) * scale)
        )
      )

      this.biases.push(Array.from({ length: outputSize }, () => 0))
    }
  }

  setLearningRate(lr: number) {
    this.learningRate = lr
  }

  getLearningRate(): number {
    return this.learningRate
  }

  // Activation functions
  private relu(x: number): number {
    return Math.max(0, x)
  }

  private reluDerivative(x: number): number {
    return x > 0 ? 1 : 0
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))))
  }

  private sigmoidDerivative(x: number): number {
    const s = this.sigmoid(x)
    return s * (1 - s)
  }

  // Forward pass
  forward(input: number[]): number[] {
    this.activations = [input]
    this.zValues = []

    let current = input

    for (let i = 0; i < this.weights.length; i++) {
      const z: number[] = []
      const activation: number[] = []
      const isLastLayer = i === this.weights.length - 1

      for (let j = 0; j < this.weights[i].length; j++) {
        let sum = this.biases[i][j]
        for (let k = 0; k < current.length; k++) {
          sum += this.weights[i][j][k] * current[k]
        }
        z.push(sum)
        // Use sigmoid for output layer (RGB values 0-1), ReLU for hidden layers
        activation.push(isLastLayer ? this.sigmoid(sum) : this.relu(sum))
      }

      this.zValues.push(z)
      this.activations.push(activation)
      current = activation
    }

    return current
  }

  // Backward pass (backpropagation)
  backward(target: number[]): number {
    const output = this.activations[this.activations.length - 1]
    const numLayers = this.weights.length

    // Calculate loss (MSE)
    let loss = 0
    for (let i = 0; i < output.length; i++) {
      loss += (target[i] - output[i]) ** 2
    }
    loss /= output.length

    // Calculate deltas
    const deltas: number[][] = []

    // Output layer delta
    const outputDelta: number[] = []
    for (let i = 0; i < output.length; i++) {
      const error = target[i] - output[i]
      outputDelta.push(error * this.sigmoidDerivative(this.zValues[numLayers - 1][i]))
    }
    deltas.unshift(outputDelta)

    // Hidden layers deltas
    for (let l = numLayers - 2; l >= 0; l--) {
      const delta: number[] = []
      for (let i = 0; i < this.weights[l].length; i++) {
        let sum = 0
        for (let j = 0; j < this.weights[l + 1].length; j++) {
          sum += this.weights[l + 1][j][i] * deltas[0][j]
        }
        delta.push(sum * this.reluDerivative(this.zValues[l][i]))
      }
      deltas.unshift(delta)
    }

    // Update weights and biases
    for (let l = 0; l < numLayers; l++) {
      for (let i = 0; i < this.weights[l].length; i++) {
        for (let j = 0; j < this.weights[l][i].length; j++) {
          this.weights[l][i][j] += this.learningRate * deltas[l][i] * this.activations[l][j]
        }
        this.biases[l][i] += this.learningRate * deltas[l][i]
      }
    }

    return loss
  }

  // Train on a single sample
  train(input: number[], target: number[]): number {
    this.forward(input)
    return this.backward(target)
  }

  // Predict without backpropagation
  predict(input: number[]): number[] {
    return this.forward(input)
  }
}

// Create a network optimized for image painting
export function createImagePainterNetwork(learningRate: number): NeuralNetwork {
  return new NeuralNetwork({
    inputSize: 2, // x, y coordinates (normalized)
    hiddenLayers: [64, 128, 128, 64], // Deeper network for complex patterns
    outputSize: 3, // r, g, b values (normalized)
    learningRate,
  })
}
