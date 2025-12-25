import { useState, useRef, useCallback, useEffect } from 'react'

interface TrainingState {
  isTraining: boolean
  iteration: number
  loss: number
  currentLearningRate: number
}

// Preset images in public/moments
const MOMENT_IMAGES = [
  '/moments/bday.jpg',
  '/moments/bkk.jpg',
  '/moments/din.jpg',
  '/moments/drinks.jpg',
  '/moments/friendsonly.jpg',
  '/moments/fun.jpg',
  '/moments/halloween.jpg',
  '/moments/osl.jpg',
  '/moments/plane.jpg',
  '/moments/sausalito.jpg',
]

export function ImagePainter() {
  const [imageLoaded, setImageLoaded] = useState(false)
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null)
  const [trainingState, setTrainingState] = useState<TrainingState>({
    isTraining: false,
    iteration: 0,
    loss: 0,
    currentLearningRate: 0.01,
  })
  const [workerReady, setWorkerReady] = useState(false)
  const [currentImageIndex, setCurrentImageIndex] = useState(0)

  // Parameters (defaults match convnetjs)
  const [learningRate, setLearningRate] = useState(0.01) // Initial learning rate, will decay automatically
  const [momentum, setMomentum] = useState(0.9)
  const [batchSize, setBatchSize] = useState(5)
  const [renderInterval, setRenderInterval] = useState(100)
  const [imageSize, setImageSize] = useState(64)

  const workerRef = useRef<Worker | null>(null)
  const outputCanvasRef = useRef<HTMLCanvasElement>(null)
  const renderIntervalRef = useRef<number | null>(null)
  // Store the original image element so we can resize it
  const originalImageRef = useRef<HTMLImageElement | null>(null)


  // Load image from URL
  const loadImageFromUrl = useCallback(
    (url: string, autoStart = false) => {
      const img = new Image()
      img.onload = () => {
        originalImageRef.current = img
        // Wait for worker to be ready before processing
        const checkWorker = setInterval(() => {
          if (workerRef.current) {
            clearInterval(checkWorker)
            // Process after a small delay to ensure worker is initialized
            setTimeout(() => {
              if (originalImageRef.current) {
                const canvas = document.createElement('canvas')
                canvas.width = imageSize
                canvas.height = imageSize
                const ctx = canvas.getContext('2d')
                if (!ctx || !workerRef.current) return

                ctx.drawImage(img, 0, 0, imageSize, imageSize)
                const data = ctx.getImageData(0, 0, imageSize, imageSize)

                setOriginalImageUrl(canvas.toDataURL())
                setImageLoaded(true)
                setWorkerReady(false)

                workerRef.current.postMessage({
                  type: 'init',
                  imageData: data.data,
                  width: imageSize,
                  height: imageSize,
                  learningRate,
                  momentum,
                  batchSize,
                })

                // Auto-start training if requested
                if (autoStart) {
                  // Wait for worker ready, then start
                  const waitForReady = (event: MessageEvent) => {
                    if (event.data.type === 'ready') {
                      workerRef.current?.removeEventListener('message', waitForReady)
                      setTrainingState((prev) => ({ ...prev, isTraining: true }))
                      workerRef.current?.postMessage({ type: 'start' })
                      workerRef.current?.postMessage({ type: 'render' })
                    }
                  }
                  workerRef.current?.addEventListener('message', waitForReady)
                }
              }
            }, 100)
          }
        }, 50)
      }
      img.src = url
    },
    [imageSize, learningRate, momentum, batchSize]
  )

  // Load default image on mount
  useEffect(() => {
    loadImageFromUrl(MOMENT_IMAGES[0])
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Navigate to next/previous image
  const goToNextImage = useCallback(() => {
    // Stop training if running
    if (trainingState.isTraining) {
      setTrainingState((prev) => ({ ...prev, isTraining: false }))
      workerRef.current?.postMessage({ type: 'stop' })
    }
    const nextIndex = (currentImageIndex + 1) % MOMENT_IMAGES.length
    setCurrentImageIndex(nextIndex)
    loadImageFromUrl(MOMENT_IMAGES[nextIndex], true)
  }, [currentImageIndex, loadImageFromUrl, trainingState.isTraining])

  const goToPrevImage = useCallback(() => {
    // Stop training if running
    if (trainingState.isTraining) {
      setTrainingState((prev) => ({ ...prev, isTraining: false }))
      workerRef.current?.postMessage({ type: 'stop' })
    }
    const prevIndex = (currentImageIndex - 1 + MOMENT_IMAGES.length) % MOMENT_IMAGES.length
    setCurrentImageIndex(prevIndex)
    loadImageFromUrl(MOMENT_IMAGES[prevIndex], true)
  }, [currentImageIndex, loadImageFromUrl, trainingState.isTraining])

  // Process image at given size and initialize worker
  const processImage = useCallback(
    (img: HTMLImageElement, size: number) => {
      if (!workerRef.current) return

      // Create canvas to extract image data
      const canvas = document.createElement('canvas')
      canvas.width = size
      canvas.height = size
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      // Draw scaled image
      ctx.drawImage(img, 0, 0, size, size)
      const data = ctx.getImageData(0, 0, size, size)

      setOriginalImageUrl(canvas.toDataURL())
      setImageLoaded(true)
      setWorkerReady(false)

      // Reset training state
      setTrainingState({
        isTraining: false,
        iteration: 0,
        loss: 0,
        currentLearningRate: learningRate,
      })

      // Initialize worker with image data
      workerRef.current?.postMessage({
        type: 'init',
        imageData: data.data,
        width: size,
        height: size,
        learningRate,
        momentum,
        batchSize,
      })

      // Clear output canvas
      if (outputCanvasRef.current) {
        const outputCtx = outputCanvasRef.current.getContext('2d')
        if (outputCtx) {
          outputCtx.fillStyle = '#000'
          outputCtx.fillRect(0, 0, size, size)
        }
      }
    },
    [learningRate, momentum, batchSize]
  )

  // Initialize worker
  useEffect(() => {
    workerRef.current = new Worker(
      new URL('../lib/neural-network.worker.ts', import.meta.url),
      { type: 'module' }
    )

    workerRef.current.onmessage = (e) => {
      const { type, ...data } = e.data

      switch (type) {
        case 'ready':
          setWorkerReady(true)
          break

        case 'progress':
          setTrainingState((prev) => ({
            ...prev,
            iteration: data.iteration,
            loss: data.loss,
            currentLearningRate: data.learningRate,
          }))
          break

        case 'render': {
          const canvas = outputCanvasRef.current
          if (!canvas) return
          const ctx = canvas.getContext('2d')
          if (!ctx) return

          const imageData = new ImageData(
            new Uint8ClampedArray(data.buffer),
            data.width,
            data.height
          )
          ctx.putImageData(imageData, 0, 0)
          break
        }

        case 'reset':
          setTrainingState((prev) => ({
            ...prev,
            iteration: 0,
            loss: 0,
          }))
          break
      }
    }

    return () => {
      workerRef.current?.terminate()
    }
  }, [])

  // Handle file upload
  const handleFileUpload = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      if (!file || !workerRef.current) return

      const reader = new FileReader()
      reader.onload = (e) => {
        const img = new Image()
        img.onload = () => {
          // Store original image for resizing later
          originalImageRef.current = img
          processImage(img, imageSize)
        }
        img.src = e.target?.result as string
      }
      reader.readAsDataURL(file)
    },
    [imageSize, processImage]
  )

  // Handle image size change - reprocess image at new size
  const handleImageSizeChange = useCallback(
    (newSize: number) => {
      setImageSize(newSize)
      if (originalImageRef.current) {
        processImage(originalImageRef.current, newSize)
      }
    },
    [processImage]
  )

  // Request render at interval
  useEffect(() => {
    if (trainingState.isTraining) {
      renderIntervalRef.current = window.setInterval(() => {
        workerRef.current?.postMessage({ type: 'render' })
      }, renderInterval)
    }

    return () => {
      if (renderIntervalRef.current) {
        clearInterval(renderIntervalRef.current)
      }
    }
  }, [trainingState.isTraining, renderInterval])

  // Start/Stop training
  const toggleTraining = useCallback(() => {
    if (!imageLoaded || !workerReady) return

    const newIsTraining = !trainingState.isTraining

    setTrainingState((prev) => ({
      ...prev,
      isTraining: newIsTraining,
    }))

    workerRef.current?.postMessage({
      type: newIsTraining ? 'start' : 'stop',
    })

    // Render immediately when starting
    if (newIsTraining) {
      workerRef.current?.postMessage({ type: 'render' })
    }
  }, [imageLoaded, workerReady, trainingState.isTraining])

  // Reset training
  const resetTraining = useCallback(() => {
    setTrainingState((prev) => ({
      ...prev,
      isTraining: false,
    }))

    workerRef.current?.postMessage({
      type: 'reset',
      learningRate,
      momentum,
    })

    // Clear output canvas
    if (outputCanvasRef.current) {
      const ctx = outputCanvasRef.current.getContext('2d')
      if (ctx) {
        ctx.fillStyle = '#000'
        ctx.fillRect(0, 0, imageSize, imageSize)
      }
    }
  }, [learningRate, momentum, imageSize])

  // Render now
  const renderNow = useCallback(() => {
    workerRef.current?.postMessage({ type: 'render' })
  }, [])

  // Update learning rate
  const handleLearningRateChange = useCallback((value: number) => {
    setLearningRate(value)
    workerRef.current?.postMessage({
      type: 'setLearningRate',
      learningRate: value,
    })
  }, [])

  // Update momentum
  const handleMomentumChange = useCallback((value: number) => {
    setMomentum(value)
    workerRef.current?.postMessage({
      type: 'setMomentum',
      momentum: value,
    })
  }, [])

  // Update batch size
  const handleBatchSizeChange = useCallback((value: number) => {
    setBatchSize(value)
    workerRef.current?.postMessage({
      type: 'setBatchSize',
      batchSize: value,
    })
  }, [])

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-foreground mb-2">Neural Network Image Painter</h1>
        <p className="text-muted-foreground mb-8">
          Upload an image and watch a neural network learn to &quot;paint&quot; it pixel by pixel.
        </p>

        {/* Controls */}
        <div className="bg-card border border-border rounded-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Parameters</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium mb-2">Upload Image</label>
              <input
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
              <label className="block text-sm font-medium mb-2">
                Image Size: {imageSize}px
              </label>
              <input
                type="range"
                min="32"
                max="128"
                step="32"
                value={imageSize}
                onChange={(e) => handleImageSizeChange(Number(e.target.value))}
                disabled={trainingState.isTraining}
                className="w-full"
              />
            </div>

            {/* Learning Rate */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Initial Learning Rate: {learningRate.toFixed(4)}
              </label>
              <input
                type="range"
                min="0.001"
                max="0.05"
                step="0.001"
                value={learningRate}
                onChange={(e) => handleLearningRateChange(Number(e.target.value))}
                disabled={trainingState.isTraining}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>0.001 (slow)</span>
                <span>0.05 (fast)</span>
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Auto-decays during training
              </div>
            </div>

            {/* Momentum */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Momentum: {momentum.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="0.99"
                step="0.01"
                value={momentum}
                onChange={(e) => handleMomentumChange(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>0 (none)</span>
                <span>0.99 (high)</span>
              </div>
            </div>

            {/* Batch Size */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Batch Size: {batchSize}
              </label>
              <input
                type="range"
                min="1"
                max="32"
                step="1"
                value={batchSize}
                onChange={(e) => handleBatchSizeChange(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>1 (SGD)</span>
                <span>32 (mini-batch)</span>
              </div>
            </div>

            {/* Render Interval */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Render Interval: {renderInterval}ms
              </label>
              <input
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
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4 mt-6">
            <button
              onClick={toggleTraining}
              disabled={!imageLoaded || !workerReady}
              className="px-6 py-2 bg-primary text-primary-foreground rounded-md font-medium
                disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
            >
              {trainingState.isTraining ? 'Pause' : 'Start Training'}
            </button>
            <button
              onClick={resetTraining}
              disabled={!imageLoaded}
              className="px-6 py-2 bg-secondary text-secondary-foreground rounded-md font-medium
                disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
            >
              Reset
            </button>
            <button
              onClick={renderNow}
              disabled={!imageLoaded || !workerReady}
              className="px-6 py-2 bg-secondary text-secondary-foreground rounded-md font-medium
                disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
            >
              Render Now
            </button>
          </div>
        </div>

        {/* Training Stats */}
        {imageLoaded && (
          <div className="bg-card border border-border rounded-lg p-6 mb-8">
            <h2 className="text-xl font-semibold mb-4">Training Progress</h2>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-2xl font-mono font-bold text-primary">
                  {trainingState.iteration.toLocaleString()}
                </div>
                <div className="text-sm text-muted-foreground">Iteration</div>
              </div>
              <div>
                <div className="text-2xl font-mono font-bold text-primary">
                  {trainingState.loss.toFixed(6)}
                </div>
                <div className="text-sm text-muted-foreground">Loss (MSE)</div>
              </div>
              <div>
                <div className="text-2xl font-mono font-bold text-primary">
                  {trainingState.currentLearningRate.toFixed(6)}
                </div>
                <div className="text-sm text-muted-foreground">Current LR</div>
              </div>
            </div>
          </div>
        )}

        {/* Image Display */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Original Image */}
          <div className="bg-card border border-border rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-center">Original Image</h2>
            <div
              className="flex items-center justify-center bg-muted rounded-lg overflow-hidden"
              style={{ minHeight: imageSize * 2 + 32 }}
            >
              {originalImageUrl ? (
                <img
                  src={originalImageUrl}
                  alt="Original"
                  style={{ width: imageSize * 2, height: imageSize * 2, imageRendering: 'pixelated' }}
                />
              ) : (
                <div className="text-muted-foreground text-sm">Upload an image to start</div>
              )}
            </div>
            {/* Image navigation */}
            <div className="flex items-center justify-center gap-4 mt-4">
              <button
                onClick={goToPrevImage}
                className="px-4 py-2 bg-secondary text-secondary-foreground rounded-md font-medium
                  hover:opacity-90 transition-opacity"
              >
                Prev
              </button>
              <span className="text-sm text-muted-foreground">
                {currentImageIndex + 1} / {MOMENT_IMAGES.length}
              </span>
              <button
                onClick={goToNextImage}
                className="px-4 py-2 bg-secondary text-secondary-foreground rounded-md font-medium
                  hover:opacity-90 transition-opacity"
              >
                Next
              </button>
            </div>
          </div>

          {/* Neural Network Output */}
          <div className="bg-card border border-border rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-center">Neural Network Output</h2>
            <div
              className="flex items-center justify-center bg-muted rounded-lg overflow-hidden"
              style={{ minHeight: imageSize * 2 + 32 }}
            >
              <canvas
                ref={outputCanvasRef}
                width={imageSize}
                height={imageSize}
                style={{ width: imageSize * 2, height: imageSize * 2, imageRendering: 'pixelated' }}
                className="bg-black"
              />
            </div>
          </div>
        </div>

        {/* Info */}
        <div className="mt-8 text-sm text-muted-foreground">
          <h3 className="font-semibold text-foreground mb-2">How it works:</h3>
          <ul className="list-disc list-inside space-y-1">
            <li>
              A deep MLP (7 hidden layers x 20 neurons) learns to map (x, y) to RGB
            </li>
            <li>
              Uses SGD with momentum (like convnetjs) for faster convergence
            </li>
            <li>
              Training runs in a Web Worker so it doesn&apos;t block the UI
            </li>
            <li>
              Higher momentum (0.9) helps smooth out gradients and converge faster
            </li>
            <li>
              Smaller batch sizes (1-5) give noisier but faster updates
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}
