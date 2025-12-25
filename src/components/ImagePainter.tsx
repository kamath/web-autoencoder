import { useCallback, useEffect, useId, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface TrainingState {
	isTraining: boolean;
	iteration: number;
	loss: number;
	currentLearningRate: number;
}

// Preset images in public/moments
const MOMENT_IMAGES = [
	"/moments/brussels.jpg",
	"/moments/sausalito.jpg",
	"/moments/surprise.JPG",
	"/moments/bday.jpg",
	"/moments/france.JPG",
];

export function ImagePainter() {
	const imageUploadId = useId();
	const imageSizeId = useId();
	const learningRateId = useId();
	const momentumId = useId();
	const batchSizeId = useId();
	const renderIntervalId = useId();

	const [imageLoaded, setImageLoaded] = useState(false);
	const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
	const [fullResolutionImageUrl, setFullResolutionImageUrl] = useState<
		string | null
	>(null);
	const [trainingState, setTrainingState] = useState<TrainingState>({
		isTraining: false,
		iteration: 0,
		loss: 0,
		currentLearningRate: 0.01,
	});
	const [workerReady, setWorkerReady] = useState(false);
	const [currentImageIndex, setCurrentImageIndex] = useState(0);
	const [parametersExpanded, setParametersExpanded] = useState(false);
	const [showFullResolution, setShowFullResolution] = useState(true);
	const [blogContent, setBlogContent] = useState("");

	// Parameters (defaults match convnetjs)
	const [learningRate, setLearningRate] = useState(0.01); // Initial learning rate, will decay automatically
	const [momentum, setMomentum] = useState(0.9);
	const [batchSize, setBatchSize] = useState(5);
	const [renderInterval, setRenderInterval] = useState(100);
	// Default to 128px on non-mobile, 64px on mobile
	const [imageSize, setImageSize] = useState(() => {
		if (typeof window !== "undefined") {
			return window.innerWidth < 768 ? 64 : 128;
		}
		return 128;
	});

	const workerRef = useRef<Worker | null>(null);
	const outputCanvasRef = useRef<HTMLCanvasElement>(null);
	const renderIntervalRef = useRef<number | null>(null);
	// Store the original image element so we can resize it
	const originalImageRef = useRef<HTMLImageElement | null>(null);

	// Load image from URL
	const loadImageFromUrl = useCallback(
		(url: string, autoStart = false) => {
			const img = new Image();
			img.onload = () => {
				originalImageRef.current = img;

				// Store full resolution image
				setFullResolutionImageUrl(url);

				// Wait for worker to be ready before processing
				const checkWorker = setInterval(() => {
					if (workerRef.current) {
						clearInterval(checkWorker);
						// Process after a small delay to ensure worker is initialized
						setTimeout(() => {
							if (originalImageRef.current) {
								const canvas = document.createElement("canvas");
								canvas.width = imageSize;
								canvas.height = imageSize;
								const ctx = canvas.getContext("2d");
								if (!ctx || !workerRef.current) return;

								ctx.drawImage(img, 0, 0, imageSize, imageSize);
								const data = ctx.getImageData(0, 0, imageSize, imageSize);

								setOriginalImageUrl(canvas.toDataURL());
								setImageLoaded(true);
								setWorkerReady(false);

								workerRef.current.postMessage({
									type: "init",
									imageData: data.data,
									width: imageSize,
									height: imageSize,
									learningRate,
									momentum,
									batchSize,
								});

								// Auto-start training if requested
								if (autoStart) {
									// Wait for worker ready, then start
									const waitForReady = (event: MessageEvent) => {
										if (event.data.type === "ready") {
											workerRef.current?.removeEventListener(
												"message",
												waitForReady,
											);
											setTrainingState((prev) => ({
												...prev,
												isTraining: true,
											}));
											workerRef.current?.postMessage({ type: "start" });
											workerRef.current?.postMessage({ type: "render" });
										}
									};
									workerRef.current?.addEventListener("message", waitForReady);
								}
							}
						}, 100);
					}
				}, 50);
			};
			img.src = url;
		},
		[imageSize, learningRate, momentum, batchSize],
	);

	// Load default image on mount
	useEffect(() => {
		loadImageFromUrl(MOMENT_IMAGES[0], true);
	}, [loadImageFromUrl]);

	// Load blog post content
	useEffect(() => {
		fetch("/content/blog-post.md")
			.then((res) => res.text())
			.then((text) => setBlogContent(text))
			.catch((err) => console.error("Failed to load blog post:", err));
	}, []);

	// Navigate to next/previous image
	const goToNextImage = useCallback(() => {
		// Stop training if running
		if (trainingState.isTraining) {
			setTrainingState((prev) => ({ ...prev, isTraining: false }));
			workerRef.current?.postMessage({ type: "stop" });
		}
		const nextIndex = (currentImageIndex + 1) % MOMENT_IMAGES.length;
		setCurrentImageIndex(nextIndex);
		loadImageFromUrl(MOMENT_IMAGES[nextIndex], true);
	}, [currentImageIndex, loadImageFromUrl, trainingState.isTraining]);

	const goToPrevImage = useCallback(() => {
		// Stop training if running
		if (trainingState.isTraining) {
			setTrainingState((prev) => ({ ...prev, isTraining: false }));
			workerRef.current?.postMessage({ type: "stop" });
		}
		const prevIndex =
			(currentImageIndex - 1 + MOMENT_IMAGES.length) % MOMENT_IMAGES.length;
		setCurrentImageIndex(prevIndex);
		loadImageFromUrl(MOMENT_IMAGES[prevIndex], true);
	}, [currentImageIndex, loadImageFromUrl, trainingState.isTraining]);

	// Process image at given size and initialize worker
	const processImage = useCallback(
		(img: HTMLImageElement, size: number) => {
			if (!workerRef.current) return;

			// Create canvas to extract image data
			const canvas = document.createElement("canvas");
			canvas.width = size;
			canvas.height = size;
			const ctx = canvas.getContext("2d");
			if (!ctx) return;

			// Draw scaled image
			ctx.drawImage(img, 0, 0, size, size);
			const data = ctx.getImageData(0, 0, size, size);

			setOriginalImageUrl(canvas.toDataURL());
			setImageLoaded(true);
			setWorkerReady(false);

			// Reset training state
			setTrainingState({
				isTraining: false,
				iteration: 0,
				loss: 0,
				currentLearningRate: learningRate,
			});

			// Initialize worker with image data
			workerRef.current?.postMessage({
				type: "init",
				imageData: data.data,
				width: size,
				height: size,
				learningRate,
				momentum,
				batchSize,
			});

			// Clear output canvas
			if (outputCanvasRef.current) {
				const outputCtx = outputCanvasRef.current.getContext("2d");
				if (outputCtx) {
					outputCtx.fillStyle = "#000";
					outputCtx.fillRect(0, 0, size, size);
				}
			}
		},
		[learningRate, momentum, batchSize],
	);

	// Initialize worker
	useEffect(() => {
		workerRef.current = new Worker(
			new URL("../lib/neural-network.worker.ts", import.meta.url),
			{ type: "module" },
		);

		workerRef.current.onmessage = (e) => {
			const { type, ...data } = e.data;

			switch (type) {
				case "ready":
					setWorkerReady(true);
					break;

				case "progress":
					setTrainingState((prev) => ({
						...prev,
						iteration: data.iteration,
						loss: data.loss,
						currentLearningRate: data.learningRate,
					}));
					break;

				case "render": {
					const canvas = outputCanvasRef.current;
					if (!canvas) return;
					const ctx = canvas.getContext("2d");
					if (!ctx) return;

					const imageData = new ImageData(
						new Uint8ClampedArray(data.buffer),
						data.width,
						data.height,
					);
					ctx.putImageData(imageData, 0, 0);
					break;
				}

				case "reset":
					setTrainingState((prev) => ({
						...prev,
						iteration: 0,
						loss: 0,
					}));
					break;
			}
		};

		return () => {
			workerRef.current?.terminate();
		};
	}, []);

	// Handle file upload
	const handleFileUpload = useCallback(
		(event: React.ChangeEvent<HTMLInputElement>) => {
			const file = event.target.files?.[0];
			if (!file || !workerRef.current) return;

			const reader = new FileReader();
			reader.onload = (e) => {
				const img = new Image();
				img.onload = () => {
					// Store original image for resizing later
					originalImageRef.current = img;
					// Store full resolution image
					setFullResolutionImageUrl(e.target?.result as string);
					processImage(img, imageSize);
				};
				img.src = e.target?.result as string;
			};
			reader.readAsDataURL(file);
		},
		[imageSize, processImage],
	);

	// Handle image size change - reprocess image at new size
	const handleImageSizeChange = useCallback(
		(newSize: number) => {
			setImageSize(newSize);
			if (originalImageRef.current) {
				processImage(originalImageRef.current, newSize);
			}
		},
		[processImage],
	);

	// Request render at interval
	useEffect(() => {
		if (trainingState.isTraining) {
			renderIntervalRef.current = window.setInterval(() => {
				workerRef.current?.postMessage({ type: "render" });
			}, renderInterval);
		}

		return () => {
			if (renderIntervalRef.current) {
				clearInterval(renderIntervalRef.current);
			}
		};
	}, [trainingState.isTraining, renderInterval]);

	// Start/Stop training
	const toggleTraining = useCallback(() => {
		if (!imageLoaded || !workerReady) return;

		const newIsTraining = !trainingState.isTraining;

		setTrainingState((prev) => ({
			...prev,
			isTraining: newIsTraining,
		}));

		workerRef.current?.postMessage({
			type: newIsTraining ? "start" : "stop",
		});

		// Render immediately when starting
		if (newIsTraining) {
			workerRef.current?.postMessage({ type: "render" });
		}
	}, [imageLoaded, workerReady, trainingState.isTraining]);

	// Reset training
	const resetTraining = useCallback(() => {
		setTrainingState((prev) => ({
			...prev,
			isTraining: false,
		}));

		workerRef.current?.postMessage({
			type: "reset",
			learningRate,
			momentum,
		});

		// Clear output canvas
		if (outputCanvasRef.current) {
			const ctx = outputCanvasRef.current.getContext("2d");
			if (ctx) {
				ctx.fillStyle = "#000";
				ctx.fillRect(0, 0, imageSize, imageSize);
			}
		}
	}, [learningRate, momentum, imageSize]);

	// Update learning rate
	const handleLearningRateChange = useCallback((value: number) => {
		setLearningRate(value);
		workerRef.current?.postMessage({
			type: "setLearningRate",
			learningRate: value,
		});
	}, []);

	// Update momentum
	const handleMomentumChange = useCallback((value: number) => {
		setMomentum(value);
		workerRef.current?.postMessage({
			type: "setMomentum",
			momentum: value,
		});
	}, []);

	// Update batch size
	const handleBatchSizeChange = useCallback((value: number) => {
		setBatchSize(value);
		workerRef.current?.postMessage({
			type: "setBatchSize",
			batchSize: value,
		});
	}, []);

	return (
		<div className="min-h-screen bg-background p-4 md:p-8">
			<div className="max-w-4xl mx-auto">
				<h1 className="text-2xl md:text-3xl font-bold text-foreground mb-2">
					Learning to see
				</h1>
				<p className="text-sm md:text-base text-muted-foreground mb-6 md:mb-8">
					Watch a neural network learn to &quot;paint&quot; it pixel by pixel,
					heavily inspired by{" "}
					<a
						href="https://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html"
						target="_blank"
						rel="noopener noreferrer"
						className="text-blue-500 hover:underline"
					>
						ConvNetJS
					</a>{" "}
					by Andrej Karpathy.
				</p>

				{/* Image Display - Combined Card */}
				<div className="bg-card border border-border rounded-lg p-3 md:p-6">
					<div className="grid grid-cols-2 gap-2 md:gap-6">
						{/* Original Image */}
						<div>
							<div className="flex items-center justify-center gap-2 mb-2 md:mb-4">
								<h2 className="text-sm md:text-xl font-semibold text-center">
									Original
								</h2>
								{originalImageUrl && (
									<button
										type="button"
										onClick={() => setShowFullResolution(!showFullResolution)}
										className="px-2 py-1 md:px-3 md:py-1.5 bg-secondary text-secondary-foreground rounded-md text-xs md:text-sm font-medium
                      hover:opacity-90 transition-opacity"
									>
										{showFullResolution ? "View Input" : "View Full Res"}
									</button>
								)}
							</div>
							<div
								className="flex items-center justify-center bg-muted rounded-lg overflow-hidden"
								style={{ minHeight: imageSize * 2 + 32 }}
							>
								{originalImageUrl ? (
									<img
										src={
											showFullResolution
												? (fullResolutionImageUrl ?? originalImageUrl ?? "")
												: (originalImageUrl ?? "")
										}
										alt="Original"
										style={{
											width: "100%",
											height: "auto",
											imageRendering: showFullResolution ? "auto" : "pixelated",
										}}
									/>
								) : (
									<div className="text-muted-foreground text-xs md:text-sm px-2 text-center">
										Upload an image to start
									</div>
								)}
							</div>
						</div>

						{/* Neural Network Output */}
						<div>
							<h2 className="text-sm md:text-xl font-semibold mb-2 md:mb-4 text-center">
								Painted
							</h2>
							<div
								className="flex items-center justify-center bg-muted rounded-lg overflow-hidden"
								style={{ minHeight: imageSize * 2 + 32 }}
							>
								<canvas
									ref={outputCanvasRef}
									width={imageSize}
									height={imageSize}
									style={{
										width: "100%",
										height: "auto",
										imageRendering: "pixelated",
									}}
									className="bg-black"
								/>
							</div>
						</div>
					</div>

					{/* Image navigation - moved below the images */}
					<div className="flex items-center justify-center gap-2 md:gap-4 mt-4 md:mt-6">
						<button
							type="button"
							onClick={goToPrevImage}
							className="px-3 py-1.5 md:px-4 md:py-2 bg-secondary text-secondary-foreground rounded-md font-medium text-sm md:text-base
                hover:opacity-90 transition-opacity"
						>
							Prev
						</button>
						<span className="text-xs md:text-sm text-muted-foreground">
							{currentImageIndex + 1} / {MOMENT_IMAGES.length}
						</span>
						<button
							type="button"
							onClick={goToNextImage}
							className="px-3 py-1.5 md:px-4 md:py-2 bg-secondary text-secondary-foreground rounded-md font-medium text-sm md:text-base
                hover:opacity-90 transition-opacity"
						>
							Next
						</button>
					</div>
				</div>

				{/* Training Stats */}
				{imageLoaded && (
					<div className="bg-card border border-border rounded-lg p-4 md:p-6 my-8">
						<h2 className="text-lg md:text-xl font-semibold mb-3 md:mb-4">
							Training Progress
						</h2>
						{/* Action Buttons */}
						<div className="flex gap-3 md:gap-4 mt-6">
							<button
								type="button"
								onClick={toggleTraining}
								disabled={!imageLoaded || !workerReady}
								className="px-4 py-2 bg-primary text-primary-foreground rounded-md font-medium
                disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity text-sm md:text-base"
							>
								{trainingState.isTraining ? "Pause" : "Start"}
							</button>
							<button
								type="button"
								onClick={resetTraining}
								disabled={!imageLoaded}
								className="px-4 py-2 bg-secondary text-secondary-foreground rounded-md font-medium
                disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity text-sm md:text-base"
							>
								Reset
							</button>
							<button
								type="button"
								onClick={() => setParametersExpanded(!parametersExpanded)}
								className="px-4 py-2 bg-secondary text-secondary-foreground rounded-md font-medium
                hover:opacity-90 transition-opacity text-sm md:text-base"
							>
								Configure
							</button>
						</div>

						{/* Configuration Options */}
						{parametersExpanded && (
							<div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6 pt-6 border-t border-border">
								{/* Training Stats */}
								<div className="md:col-span-2 grid grid-cols-3 gap-2 md:gap-4 text-center pb-6 border-b border-border">
									<div>
										<div className="text-lg md:text-2xl font-mono font-bold text-primary">
											{trainingState.iteration.toLocaleString()}
										</div>
										<div className="text-xs md:text-sm text-muted-foreground">
											Iteration
										</div>
									</div>
									<div>
										<div className="text-lg md:text-2xl font-mono font-bold text-primary">
											{trainingState.loss.toFixed(6)}
										</div>
										<div className="text-xs md:text-sm text-muted-foreground">
											Loss (MSE)
										</div>
									</div>
									<div>
										<div className="text-lg md:text-2xl font-mono font-bold text-primary">
											{trainingState.currentLearningRate.toFixed(6)}
										</div>
										<div className="text-xs md:text-sm text-muted-foreground">
											Current LR
										</div>
									</div>
								</div>

								{/* File Upload */}
								<div>
									<label
										htmlFor={imageUploadId}
										className="block text-sm font-medium mb-2"
									>
										Upload Image
									</label>
									<input
										id={imageUploadId}
										type="file"
										accept="image/*"
										onChange={handleFileUpload}
										className="block w-full text-sm text-muted-foreground
                      file:mr-4 file:py-2 file:px-4
                      file:rounded-md file:border-0
                      file:text-sm file:font-medium
                      file:bg-primary file:text-primary-foreground
                      hover:file:opacity-90 cursor-pointer"
									/>
								</div>

								{/* Image Size */}
								<div>
									<label
										htmlFor={imageSizeId}
										className="block text-sm font-medium mb-2"
									>
										Image Size: {imageSize}px
									</label>
									<input
										id={imageSizeId}
										type="range"
										min="32"
										max="128"
										step="32"
										value={imageSize}
										onChange={(e) =>
											handleImageSizeChange(Number(e.target.value))
										}
										disabled={trainingState.isTraining}
										className="w-full"
									/>
									<p className="text-xs text-muted-foreground mt-2">
										The size of the image to train on. Larger images take longer
										but have more detail.
									</p>
								</div>

								{/* Learning Rate */}
								<div>
									<label
										htmlFor={learningRateId}
										className="block text-sm font-medium mb-2"
									>
										Initial Learning Rate: {learningRate.toFixed(4)}
									</label>
									<input
										id={learningRateId}
										type="range"
										min="0.001"
										max="0.05"
										step="0.001"
										value={learningRate}
										onChange={(e) =>
											handleLearningRateChange(Number(e.target.value))
										}
										disabled={trainingState.isTraining}
										className="w-full"
									/>
									<div className="flex justify-between text-xs text-muted-foreground mt-1">
										<span>0.001 (slow)</span>
										<span>0.05 (fast)</span>
									</div>
									<p className="text-xs text-muted-foreground mt-2">
										Controls how fast the network learns. Higher values learn
										faster but may be unstable. Automatically decreases over
										time for better results.
									</p>
								</div>

								{/* Momentum */}
								<div>
									<label
										htmlFor={momentumId}
										className="block text-sm font-medium mb-2"
									>
										Momentum: {momentum.toFixed(2)}
									</label>
									<input
										id={momentumId}
										type="range"
										min="0"
										max="0.99"
										step="0.01"
										value={momentum}
										onChange={(e) =>
											handleMomentumChange(Number(e.target.value))
										}
										className="w-full"
									/>
									<div className="flex justify-between text-xs text-muted-foreground mt-1">
										<span>0 (none)</span>
										<span>0.99 (high)</span>
									</div>
									<p className="text-xs text-muted-foreground mt-2">
										Helps the network learn smoother and faster by remembering
										previous adjustments. Higher values (like 0.9) work best for
										most images.
									</p>
								</div>

								{/* Batch Size */}
								<div>
									<label
										htmlFor={batchSizeId}
										className="block text-sm font-medium mb-2"
									>
										Batch Size: {batchSize}
									</label>
									<input
										id={batchSizeId}
										type="range"
										min="1"
										max="32"
										step="1"
										value={batchSize}
										onChange={(e) =>
											handleBatchSizeChange(Number(e.target.value))
										}
										className="w-full"
									/>
									<div className="flex justify-between text-xs text-muted-foreground mt-1">
										<span>1 (SGD)</span>
										<span>32 (mini-batch)</span>
									</div>
									<p className="text-xs text-muted-foreground mt-2">
										Number of pixels to learn from at once. Smaller values (1-5)
										give faster, more varied updates. Larger values give
										smoother, more stable learning.
									</p>
								</div>

								{/* Render Interval */}
								<div>
									<label
										htmlFor={renderIntervalId}
										className="block text-sm font-medium mb-2"
									>
										Render Interval: {renderInterval}ms
									</label>
									<input
										id={renderIntervalId}
										type="range"
										min="50"
										max="500"
										step="50"
										value={renderInterval}
										onChange={(e) => setRenderInterval(Number(e.target.value))}
										className="w-full"
									/>
									<div className="flex justify-between text-xs text-muted-foreground mt-1">
										<span>50ms (smooth)</span>
										<span>500ms (fast)</span>
									</div>
									<p className="text-xs text-muted-foreground mt-2">
										How often to update the display. Lower values show smoother
										progress but may slow down training. Higher values train
										faster but update less frequently.
									</p>
								</div>
							</div>
						)}
					</div>
				)}

				{/* Blog Post */}
				{blogContent && (
					<div className="mt-8 bg-card px-4 md:px-6">
						<div className="markdown-content">
							<Markdown remarkPlugins={[remarkGfm]}>{blogContent}</Markdown>
						</div>
					</div>
				)}
			</div>
		</div>
	);
}
