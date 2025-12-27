export interface GifMemoryEstimate {
  bytesPerFrame: number
  totalFrames: number
  totalBytes: number
  megabytes: number
}

export function calculateGifMemory(
  width: number,
  height: number,
  frameCount: number,
): GifMemoryEstimate {
  const bytesPerFrame = width * height * 4 // RGBA
  // Formula: forward frames + reverse frames + original frame
  const totalFrames = frameCount * 2 + 1
  const totalBytes = totalFrames * bytesPerFrame

  return {
    bytesPerFrame,
    totalFrames,
    totalBytes,
    megabytes: totalBytes / (1024 * 1024),
  }
}

export function calculateSnapshotMilestones(
  maxIterations: number,
  frameCount: number,
): number[] {
  const milestones = [0]

  if (frameCount <= 2) {
    milestones.push(maxIterations)
    return milestones
  }

  // Logarithmic spacing for better visual progression
  const logBase = Math.pow(maxIterations, 1 / (frameCount - 2))

  for (let i = 1; i < frameCount - 1; i++) {
    milestones.push(Math.floor(Math.pow(logBase, i)))
  }

  milestones.push(maxIterations)
  return milestones
}

export function morphFrames(
  frame1: Uint8ClampedArray,
  frame2: Uint8ClampedArray,
  t: number,
): Uint8ClampedArray {
  const result = new Uint8ClampedArray(frame1.length)
  const invT = 1 - t

  for (let i = 0; i < frame1.length; i++) {
    result[i] = Math.round(frame1[i] * invT + frame2[i] * t)
  }

  return result
}
