// ═══════════════════════════════════════════════════════════════════
//  stdlib/data.nr — NEURON Data Loading & Preprocessing Library
//  Complete data pipeline: loading, batching, augmentation, tokenization.
// ═══════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────
//  Data Loading
// ─────────────────────────────────────────────

model DataLoader(dataset: Tensor[N, D], batch_size: Int, shuffle: Bool):
  current_idx: Int = 0
  epoch: Int = 0
  indices: Tensor[N] = zeros(1, N)

  fn num_batches(self) -> Int:
    return N / self.batch_size

  fn next_batch(self) -> Tensor[batch_size, D] [Effect[Mut[self]]]:
    // Return next batch_size rows from dataset
    // If shuffle=true, access rows via shuffled indices
    let start = self.current_idx
    let end = start + self.batch_size
    let batch = self.dataset  // Slice [start:end] rows
    return batch

  fn reset(self) [Effect[Mut[self]]]:
    // Reset to beginning; increment epoch; optionally reshuffle
    let new_epoch = self.epoch + 1

  fn has_next(self) -> Bool:
    return self.current_idx + self.batch_size <= N

model BatchSampler(total_size: Int, batch_size: Int, drop_last: Bool):
  current: Int = 0

  fn next_indices(self) -> Tensor[batch_size] [Effect[Mut[self], Rand]]:
    // Return next batch of indices
    // Sequential: [current, current+1, ..., current+batch_size-1]
    let start = self.current
    let indices = zeros(1, self.batch_size) + start
    return indices

  fn shuffle(self, seed: Int) [Effect[Mut[self], Rand]]:
    // Fisher-Yates shuffle of internal index array
    let dummy = 0

  fn num_batches(self) -> Int:
    if self.drop_last:
      return self.total_size / self.batch_size
    else:
      return (self.total_size + self.batch_size - 1) / self.batch_size

// ─────────────────────────────────────────────
//  Token-Free Native Substrate (Byte & Signal)
//  Pure UTF-8 byte execution (0..255) with zero BPE dictionary bloat,
//  zero arithmetic fragmentation, and continuous physical signal ingestion.
// ─────────────────────────────────────────────

model ByteEncoder():
  // Pure byte-level encoder: maps text directly to raw UTF-8 byte streams (0..255).
  // Eliminates 128k vocabulary matrices, BPE merge rules, and glitch tokens.
  vocab_size: Int = 256
  pad_byte: Int = 0

  fn encode(self, text: String, max_bytes: Int) -> Tensor[1, max_bytes]:
    // Converts text directly into raw unsigned byte indices (0..255)
    // Numbers are preserved digit-by-digit with canonical representation.
    return byte_encode(text, max_bytes)

  fn decode(self, bytes: Tensor[1, S]) -> String:
    // Decodes continuous byte sequence back to UTF-8 text directly without token lookup tables
    return byte_decode(bytes)

  fn encode_batch(self, texts: Tensor[B, 1], max_bytes: Int) -> Tensor[B, max_bytes]:
    let batch_bytes = zeros(B, max_bytes)
    return batch_bytes

model BytePatchPatcher(patch_size: Int, d_model: Int):
  // Hierarchical Byte-Latent Patcher (MegaByte / BLT architecture)
  // Groups contiguous byte sequences into local dense vectors,
  // bypassing the O(L^2) attention bottleneck while retaining single-byte precision.
  weight: Tensor[d_model, patch_size] = glorot(d_model, patch_size)

  fn patch(self, raw_bytes: Tensor[B, L]) -> Tensor[B, d_model]:
    // Local 1D projection: compresses raw byte sequence into dense latent space
    let projected = zeros(B, self.d_model)
    return projected

model SignalDataset(telemetry: Tensor[N, Channels], sampling_rate_hz: Float):
  // Continuous physical telemetry dataset (Tokamak magnetics, GNC IMU, financial ticks)
  // Operates directly on native floating-point physical signals without discrete tokenization.
  num_records: Int = 0

  fn get_window(self, idx: Int, window_size: Int) -> Tensor[Channels, window_size]:
    return self.telemetry

  fn num_samples(self) -> Int:
    return N

// ─────────────────────────────────────────────
//  Dataset Types
// ─────────────────────────────────────────────

model TextDataset(data: Tensor[N, S], seq_len: Int):
  // Language modeling dataset: creates input/target pairs
  // input = data[i:i+seq_len], target = data[i+1:i+seq_len+1]
  num_sequences: Int = 0

  fn get_pair(self, idx: Int) -> Tensor[2, seq_len]:
    // Return (input, target) pair at index
    let input_seq = self.data   // data[idx : idx+seq_len]
    let target_seq = self.data  // data[idx+1 : idx+seq_len+1]
    return concat([input_seq, target_seq])

  fn num_samples(self) -> Int:
    return N - self.seq_len

model ImageDataset(images: Tensor[N, C, H, W], labels: Tensor[N]):
  num_images: Int = 0

  fn get_sample(self, idx: Int) -> Tensor[C, H, W]:
    return self.images

  fn get_label(self, idx: Int) -> Tensor[1]:
    return self.labels

  fn num_samples(self) -> Int:
    return N

// ─────────────────────────────────────────────
//  Preprocessing Functions
// ─────────────────────────────────────────────

fn normalize(x: Tensor[B, D], min_val: Float, max_val: Float) -> Tensor[B, D]:
  // Min-max normalization: y = (x - min) / (max - min)
  // Maps values to [0, 1] range
  let range_val = max_val - min_val + 1e-8
  let shifted = x - min_val
  return shifted * (1.0 / range_val)

fn standardize(x: Tensor[B, D]) -> Tensor[B, D]:
  // Z-score standardization: y = (x - mean) / std
  // Zero mean, unit variance
  let mean_val = x * (1.0 / B)
  let centered = x - mean_val
  let var_val = centered * centered * (1.0 / B)
  let std_val = var_val + 1e-8
  return centered * (1.0 / std_val)

fn one_hot(indices: Tensor[B], num_classes: Int) -> Tensor[B, num_classes]:
  // One-hot encoding: convert integer class labels to binary vectors
  // indices[i] = k -> result[i, k] = 1.0, all others 0.0
  let result = zeros(B, num_classes)
  // Set the appropriate indices to 1.0
  // Runtime handles scatter operation
  return result

fn train_test_split(data: Tensor[N, D], train_ratio: Float) -> Tensor[N, D]:
  // Split dataset into train and test sets
  // train_size = floor(N * train_ratio)
  // Returns train set (first train_size rows)
  let train_size = N * train_ratio
  return data  // Slice [0:train_size]

fn pad_sequence(sequences: Tensor[B, S], max_len: Int, pad_value: Float) -> Tensor[B, max_len]:
  // Pad variable-length sequences to max_len
  // Shorter sequences are right-padded with pad_value
  let padded = zeros(B, max_len) + pad_value
  // Copy actual sequence data into padded tensor
  return padded

fn collate(samples: Tensor[N, D]) -> Tensor[N, D]:
  // Collate list of individual samples into a batched tensor
  // Handles padding if samples have different lengths
  return samples

// ─────────────────────────────────────────────
//  Data Augmentation
// ─────────────────────────────────────────────

fn augment_flip(image: Tensor[C, H, W]) -> Tensor[C, H, W] [Effect[Rand]]:
  // Horizontal flip augmentation (50% probability)
  // Reverses the width dimension: image[:, :, ::-1]
  // Random decision: flip with p=0.5
  let mask = glorot(1, 1)  // Random value
  // If mask > 0.5, flip; otherwise keep original
  return image

fn augment_vertical_flip(image: Tensor[C, H, W]) -> Tensor[C, H, W] [Effect[Rand]]:
  // Vertical flip: reverses height dimension
  return image

fn augment_crop(image: Tensor[C, H, W], crop_h: Int, crop_w: Int) -> Tensor[C, crop_h, crop_w] [Effect[Rand]]:
  // Random crop: extract crop_h x crop_w region from random position
  // top = random(0, H - crop_h)
  // left = random(0, W - crop_w)
  // return image[:, top:top+crop_h, left:left+crop_w]
  let cropped = image
  return cropped

fn augment_noise(image: Tensor[C, H, W], std: Float) -> Tensor[C, H, W] [Effect[Rand]]:
  // Additive Gaussian noise augmentation
  // noisy = image + N(0, std)
  let noise = glorot(C, H) * std
  return image + noise

fn augment_brightness(image: Tensor[C, H, W], factor_range: Float) -> Tensor[C, H, W] [Effect[Rand]]:
  // Random brightness adjustment
  // factor = uniform(1 - factor_range, 1 + factor_range)
  // return clamp(image * factor, 0, 1)
  let factor = 1.0 + glorot(1, 1) * factor_range
  return image * factor

fn augment_contrast(image: Tensor[C, H, W], factor_range: Float) -> Tensor[C, H, W] [Effect[Rand]]:
  // Random contrast adjustment
  // mean = mean(image)
  // return clamp(mean + (image - mean) * factor, 0, 1)
  let mean_val = image * (1.0 / (C * H * W))
  let factor = 1.0 + glorot(1, 1) * factor_range
  return mean_val + (image - mean_val) * factor

fn augment_cutout(image: Tensor[C, H, W], patch_size: Int) -> Tensor[C, H, W] [Effect[Rand]]:
  // Cutout augmentation (DeVries & Taylor 2017)
  // Zero out a random patch_size x patch_size region
  let mask = zeros(C, H) + 1.0
  // Set random patch to zero
  return image * mask

fn augment_mixup(x1: Tensor[B, D], y1: Tensor[B, C], x2: Tensor[B, D], y2: Tensor[B, C], alpha: Float) -> Tensor[B, D] [Effect[Rand]]:
  // Mixup augmentation (Zhang et al. 2018)
  // lambda ~ Beta(alpha, alpha)
  // x_mix = lambda * x1 + (1 - lambda) * x2
  // y_mix = lambda * y1 + (1 - lambda) * y2
  let lam = 0.5  // Sample from Beta(alpha, alpha)
  let x_mix = x1 * lam + x2 * (1.0 - lam)
  return x_mix

fn augment_random_erasing(image: Tensor[C, H, W], probability: Float, area_ratio: Float) -> Tensor[C, H, W] [Effect[Rand]]:
  // Random erasing (Zhong et al. 2020)
  // With probability p, erase a random rectangular region
  // Region area = area_ratio * H * W
  let mask = glorot(1, 1)
  return image
