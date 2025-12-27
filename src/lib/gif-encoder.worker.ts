import { encode } from 'modern-gif'

interface EncodeMessage {
  type: 'encode'
  frames: Uint8ClampedArray[]
  width: number
  height: number
  fps: number
}

self.onmessage = async (e: MessageEvent<EncodeMessage>) => {
  const { type, frames, width, height, fps } = e.data

  if (type === 'encode') {
    try {
      const delay = Math.floor(1000 / fps) // ms per frame

      // Convert frames to modern-gif format
      const gifFrames = frames.map((data, index) => {
        // Report progress
        if (index % 10 === 0 || index === frames.length - 1) {
          self.postMessage({
            type: 'progress',
            current: index + 1,
            total: frames.length,
          })
        }

        return {
          data,
          delay,
        }
      })

      // Encode GIF
      const output = await encode({
        width,
        height,
        frames: gifFrames,
        maxColors: 256,
      })

      // Send blob back to main thread
      self.postMessage({
        type: 'complete',
        blob: new Blob([output], { type: 'image/gif' }),
      })
    } catch (error) {
      self.postMessage({
        type: 'error',
        message: error instanceof Error ? error.message : 'Unknown error',
      })
    }
  }
}
