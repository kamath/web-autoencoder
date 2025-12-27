// Web Worker for neural network training - runs off main thread
// Architecture inspired by convnetjs image painting demo

interface NetworkConfig {
	inputSize: number;
	hiddenLayers: number[];
	outputSize: number;
	learningRate: number;
	momentum: number;
	l2Decay: number;
}

interface NetworkSnapshot {
	iteration: number;
	weights: Float64Array[];
	biases: Float64Array[];
}

class NeuralNetwork {
	private weights: Float64Array[];
	private biases: Float64Array[];
	private layerSizes: number[];
	private learningRate: number;
	private momentum: number;
	private l2Decay: number;

	// Momentum velocities
	private weightVelocities: Float64Array[];
	private biasVelocities: Float64Array[];

	constructor(config: NetworkConfig) {
		this.learningRate = config.learningRate;
		this.momentum = config.momentum;
		this.l2Decay = config.l2Decay;
		this.layerSizes = [
			config.inputSize,
			...config.hiddenLayers,
			config.outputSize,
		];

		this.weights = [];
		this.biases = [];
		this.weightVelocities = [];
		this.biasVelocities = [];

		// Initialize with typed arrays for better performance
		for (let i = 0; i < this.layerSizes.length - 1; i++) {
			const inputSize = this.layerSizes[i];
			const outputSize = this.layerSizes[i + 1];
			// Xavier/He initialization
			const scale = Math.sqrt(2 / inputSize);

			const weights = new Float64Array(outputSize * inputSize);
			for (let j = 0; j < weights.length; j++) {
				weights[j] = (Math.random() * 2 - 1) * scale;
			}
			this.weights.push(weights);
			this.biases.push(new Float64Array(outputSize));

			// Initialize velocities to zero
			this.weightVelocities.push(new Float64Array(outputSize * inputSize));
			this.biasVelocities.push(new Float64Array(outputSize));
		}
	}

	setLearningRate(lr: number) {
		this.learningRate = lr;
	}

	getLearningRate(): number {
		return this.learningRate;
	}

	setMomentum(m: number) {
		this.momentum = m;
	}

	// Forward pass - inline for speed
	private forward(inputX: number, inputY: number): [number, number, number] {
		let current = [inputX, inputY];
		const numLayers = this.weights.length;

		for (let l = 0; l < numLayers; l++) {
			const inputSize = this.layerSizes[l];
			const outputSize = this.layerSizes[l + 1];
			const weights = this.weights[l];
			const biases = this.biases[l];
			const isLastLayer = l === numLayers - 1;

			const next = new Array(outputSize);

			for (let j = 0; j < outputSize; j++) {
				let sum = biases[j];
				const offset = j * inputSize;
				for (let k = 0; k < current.length; k++) {
					sum += weights[offset + k] * current[k];
				}
				// ReLU for hidden layers, linear for output (regression)
				if (isLastLayer) {
					// Clamp output to [0, 1] for RGB
					next[j] = Math.max(0, Math.min(1, sum));
				} else {
					next[j] = sum > 0 ? sum : 0;
				}
			}
			current = next;
		}

		return current as [number, number, number];
	}

	// Combined forward + backward for training with momentum
	train(
		inputX: number,
		inputY: number,
		targetR: number,
		targetG: number,
		targetB: number,
	): number {
		const numLayers = this.weights.length;
		const activations: number[][] = [[inputX, inputY]];
		const zValues: number[][] = [];

		// Forward pass with stored activations
		let current = [inputX, inputY];

		for (let l = 0; l < numLayers; l++) {
			const inputSize = this.layerSizes[l];
			const outputSize = this.layerSizes[l + 1];
			const weights = this.weights[l];
			const biases = this.biases[l];
			const isLastLayer = l === numLayers - 1;

			const z = new Array(outputSize);
			const activation = new Array(outputSize);

			for (let j = 0; j < outputSize; j++) {
				let sum = biases[j];
				const offset = j * inputSize;
				for (let k = 0; k < current.length; k++) {
					sum += weights[offset + k] * current[k];
				}
				z[j] = sum;
				if (isLastLayer) {
					// Linear output with clamping for regression
					activation[j] = Math.max(0, Math.min(1, sum));
				} else {
					activation[j] = sum > 0 ? sum : 0;
				}
			}

			zValues.push(z);
			activations.push(activation);
			current = activation;
		}

		const output = activations[numLayers];
		const target = [targetR, targetG, targetB];

		// Calculate loss (MSE)
		let loss = 0;
		for (let i = 0; i < 3; i++) {
			loss += (target[i] - output[i]) ** 2;
		}
		loss /= 3;

		// Backward pass
		const deltas: number[][] = [];

		// Output layer delta (linear activation, derivative = 1)
		const outputDelta = new Array(3);
		for (let i = 0; i < 3; i++) {
			outputDelta[i] = target[i] - output[i];
		}
		deltas.unshift(outputDelta);

		// Hidden layers (ReLU derivative)
		for (let l = numLayers - 2; l >= 0; l--) {
			const outputSize = this.layerSizes[l + 1];
			const nextOutputSize = this.layerSizes[l + 2];
			const nextWeights = this.weights[l + 1];
			const delta = new Array(outputSize);

			for (let i = 0; i < outputSize; i++) {
				let sum = 0;
				for (let j = 0; j < nextOutputSize; j++) {
					sum += nextWeights[j * outputSize + i] * deltas[0][j];
				}
				delta[i] = zValues[l][i] > 0 ? sum : 0;
			}
			deltas.unshift(delta);
		}

		// Update weights and biases with momentum and L2 decay
		for (let l = 0; l < numLayers; l++) {
			const inputSize = this.layerSizes[l];
			const outputSize = this.layerSizes[l + 1];
			const weights = this.weights[l];
			const biases = this.biases[l];
			const weightVel = this.weightVelocities[l];
			const biasVel = this.biasVelocities[l];
			const prevActivation = activations[l];
			const delta = deltas[l];

			for (let j = 0; j < outputSize; j++) {
				const offset = j * inputSize;
				for (let k = 0; k < inputSize; k++) {
					const idx = offset + k;
					const grad =
						delta[j] * prevActivation[k] - this.l2Decay * weights[idx];
					weightVel[idx] =
						this.momentum * weightVel[idx] + this.learningRate * grad;
					weights[idx] += weightVel[idx];
				}
				const biasGrad = delta[j];
				biasVel[j] = this.momentum * biasVel[j] + this.learningRate * biasGrad;
				biases[j] += biasVel[j];
			}
		}

		return loss;
	}

	predict(inputX: number, inputY: number): [number, number, number] {
		return this.forward(inputX, inputY);
	}

	// Render entire image at once
	renderToBuffer(width: number, height: number): Uint8ClampedArray {
		const buffer = new Uint8ClampedArray(width * height * 4);

		for (let y = 0; y < height; y++) {
			const inputY = (y / height) * 2 - 1;
			for (let x = 0; x < width; x++) {
				const inputX = (x / width) * 2 - 1;
				const [r, g, b] = this.forward(inputX, inputY);

				const idx = (y * width + x) * 4;
				buffer[idx] = Math.round(r * 255);
				buffer[idx + 1] = Math.round(g * 255);
				buffer[idx + 2] = Math.round(b * 255);
				buffer[idx + 3] = 255;
			}
		}

		return buffer;
	}

	// Create a snapshot of current network state
	createSnapshot(iteration: number): NetworkSnapshot {
		return {
			iteration,
			weights: this.weights.map((w) => new Float64Array(w)),
			biases: this.biases.map((b) => new Float64Array(b)),
		};
	}

	// Restore network from a snapshot
	restoreSnapshot(snapshot: NetworkSnapshot): void {
		this.weights = snapshot.weights.map((w) => new Float64Array(w));
		this.biases = snapshot.biases.map((b) => new Float64Array(b));
	}
}

// Worker state
let network: NeuralNetwork | null = null;
let imageData: Uint8ClampedArray | null = null;
let imageWidth = 0;
let imageHeight = 0;
let isTraining = false;
let iteration = 0;
let batchSize = 5; // Small batch like convnetjs
let renderRequested = false;
let initialLearningRate = 0.01;
let learningRate = 0.01;
let momentum = 0.9;
let minLearningRate = 0.0001; // Don't go below this
let lossHistory: number[] = []; // Track recent losses
const LOSS_HISTORY_SIZE = 100; // Number of loss samples to track
const DECAY_THRESHOLD = 0.001; // If relative improvement is less than this, decay LR
const DECAY_FACTOR = 0.95; // Multiply LR by this when plateauing

// GIF generation state
let snapshots: NetworkSnapshot[] = [];
let captureSnapshotsEnabled = false;
let snapshotMilestones: number[] = [];
let gifFrameCount = 50;

function trainBatch(): number {
	if (!network || !imageData) return 0;

	let totalLoss = 0;

	for (let b = 0; b < batchSize; b++) {
		const x = Math.floor(Math.random() * imageWidth);
		const y = Math.floor(Math.random() * imageHeight);
		const idx = (y * imageWidth + x) * 4;

		const inputX = (x / imageWidth) * 2 - 1;
		const inputY = (y / imageHeight) * 2 - 1;

		const targetR = imageData[idx] / 255;
		const targetG = imageData[idx + 1] / 255;
		const targetB = imageData[idx + 2] / 255;

		totalLoss += network.train(inputX, inputY, targetR, targetG, targetB);
	}

	return totalLoss / batchSize;
}

function trainingLoop() {
	if (!isTraining || !network) return;

	// Train many batches per frame for speed (like convnetjs does ~50 per tick)
	const batchesPerFrame = 50;
	let totalLoss = 0;

	for (let i = 0; i < batchesPerFrame; i++) {
		totalLoss += trainBatch();
		iteration++;
	}

	const avgLoss = totalLoss / batchesPerFrame;

	// Check if we should capture a snapshot
	// Since iterations jump by batchesPerFrame (50), we need to check if we've crossed a milestone
	if (captureSnapshotsEnabled && network && snapshotMilestones.length > 0) {
		const nextMilestone = snapshotMilestones[snapshots.length];
		if (nextMilestone !== undefined && iteration >= nextMilestone) {
			snapshots.push(network.createSnapshot(iteration));
			self.postMessage({
				type: "snapshotCaptured",
				count: snapshots.length,
				total: snapshotMilestones.length,
			});
		}
	}

	// Track loss history and adapt learning rate
	lossHistory.push(avgLoss);
	if (lossHistory.length > LOSS_HISTORY_SIZE) {
		lossHistory.shift();
	}

	// Check if we should decay learning rate (every LOSS_HISTORY_SIZE iterations)
	if (
		lossHistory.length === LOSS_HISTORY_SIZE &&
		learningRate > minLearningRate
	) {
		const oldLoss = lossHistory[0];
		const newLoss = lossHistory[LOSS_HISTORY_SIZE - 1];
		const relativeImprovement = (oldLoss - newLoss) / oldLoss;

		// If improvement is below threshold, decay learning rate
		if (relativeImprovement < DECAY_THRESHOLD) {
			learningRate = Math.max(minLearningRate, learningRate * DECAY_FACTOR);
			network.setLearningRate(learningRate);
			// Clear history to give new LR time to work
			lossHistory = [];
		}
	}

	// Send progress update
	self.postMessage({
		type: "progress",
		iteration,
		loss: avgLoss,
		learningRate: network.getLearningRate(),
	});

	// Render if requested
	if (renderRequested && network) {
		renderRequested = false;
		const buffer = network.renderToBuffer(imageWidth, imageHeight);
		self.postMessage(
			{
				type: "render",
				buffer,
				width: imageWidth,
				height: imageHeight,
			},
			{ transfer: [buffer.buffer] },
		);
	}

	// Continue training
	setTimeout(trainingLoop, 0);
}

function createNetwork(): NeuralNetwork {
	return new NeuralNetwork({
		inputSize: 2,
		// 7 hidden layers with 20 neurons each (like convnetjs)
		hiddenLayers: [20, 20, 20, 20, 20, 20, 20],
		outputSize: 3,
		learningRate,
		momentum,
		l2Decay: 0.0,
	});
}

// Calculate snapshot milestones using logarithmic spacing
function calculateSnapshotMilestones(
	maxIterations: number,
	frameCount: number,
): number[] {
	const milestones = [0];

	if (frameCount <= 2) {
		milestones.push(maxIterations);
		return milestones;
	}

	// Logarithmic spacing for better visual progression
	const logBase = Math.pow(maxIterations, 1 / (frameCount - 2));

	for (let i = 1; i < frameCount - 1; i++) {
		milestones.push(Math.floor(Math.pow(logBase, i)));
	}

	console.log("milestones", milestones);
	milestones.push(maxIterations);
	return milestones;
}

// Generate all frames for the GIF journey
function generateGifFrames(width: number, height: number) {
	if (!network) {
		self.postMessage({ type: "gifError", message: "Network not initialized" });
		return;
	}

	const frames: Uint8ClampedArray[] = [];

	// Check if we have snapshots
	if (snapshots.length === 0) {
		self.postMessage({ type: "gifError", message: "No snapshots captured" });
		return;
	}

	// 1. Progressive iterations (forward)
	for (const snapshot of [...snapshots].reverse()) {
		network.restoreSnapshot(snapshot);
		const buffer = network.renderToBuffer(width, height);
		frames.push(buffer);
	}

	// 2. Reverse sequence (skip endpoints to avoid duplication)
	for (let i = frames.length - 2; i > 0; i--) {
		frames.push(new Uint8ClampedArray(frames[i]));
	}

	// Send frames to main thread
	self.postMessage({
		type: "gifFrames",
		frames,
		width,
		height,
	});
}

// Handle messages from main thread
self.onmessage = (e: MessageEvent) => {
	const { type, ...data } = e.data;

	switch (type) {
		case "init": {
			initialLearningRate = data.learningRate;
			learningRate = data.learningRate;
			momentum = data.momentum ?? 0.9;
			batchSize = data.batchSize ?? 5;
			minLearningRate = data.minLearningRate ?? 0.0001;
			network = createNetwork();
			imageData = new Uint8ClampedArray(data.imageData);
			imageWidth = data.width;
			imageHeight = data.height;
			iteration = 0;
			isTraining = false;
			lossHistory = [];
			self.postMessage({ type: "ready" });
			break;
		}

		case "start":
			isTraining = true;
			trainingLoop();
			break;

		case "stop":
			isTraining = false;
			break;

		case "render":
			renderRequested = true;
			break;

		case "setLearningRate":
			learningRate = data.learningRate;
			if (network) {
				network.setLearningRate(data.learningRate);
			}
			break;

		case "setMomentum":
			momentum = data.momentum;
			if (network) {
				network.setMomentum(data.momentum);
			}
			break;

		case "setBatchSize":
			batchSize = data.batchSize;
			break;

		case "reset":
			isTraining = false;
			iteration = 0;
			// Reset learning rate to initial value
			learningRate = data.learningRate ?? initialLearningRate;
			initialLearningRate = learningRate;
			momentum = data.momentum ?? momentum;
			lossHistory = [];
			network = createNetwork();
			self.postMessage({ type: "reset" });
			break;

		case "enableSnapshotCapture": {
			captureSnapshotsEnabled = true;
			gifFrameCount = data.frameCount || 50;
			snapshotMilestones = calculateSnapshotMilestones(
				data.maxIterations || 1_000_000,
				gifFrameCount,
			);
			snapshots = [];
			// Capture initial state if network exists
			if (network) {
				snapshots.push(network.createSnapshot(iteration));
				self.postMessage({
					type: "snapshotCaptured",
					count: snapshots.length,
					total: snapshotMilestones.length,
				});
			}
			break;
		}

		case "disableSnapshotCapture":
			captureSnapshotsEnabled = false;
			snapshots = [];
			break;

		case "generateGif": {
			const wasTraining = isTraining;
			isTraining = false;

			if (!imageData) {
				self.postMessage({
					type: "gifError",
					message: "No image data available",
				});
				break;
			}

			generateGifFrames(imageWidth, imageHeight);

			// Resume training if it was active
			if (wasTraining) {
				isTraining = true;
				trainingLoop();
			}
			break;
		}

		case "renderSnapshot": {
			if (!network || !snapshots[data.snapshotIndex]) {
				break;
			}

			const snapshot = snapshots[data.snapshotIndex];
			network.restoreSnapshot(snapshot);
			const buffer = network.renderToBuffer(imageWidth, imageHeight);

			self.postMessage(
				{
					type: "snapshotRendered",
					buffer,
					width: imageWidth,
					height: imageHeight,
					snapshotIndex: data.snapshotIndex,
					iteration: snapshot.iteration,
				},
				{ transfer: [buffer.buffer] },
			);
			break;
		}
	}
};
