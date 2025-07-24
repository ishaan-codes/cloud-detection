import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
from tqdm import tqdm
import warnings
import tacoreader
import rasterio as rio
from PIL import Image
from torch.cuda import amp
import time
import random
from functools import wraps

# Suppress warnings
warnings.filterwarnings("ignore")

# ------------------------- Configuration -------------------------
class Config:
    # Dataset configuration
    DATASET_NAME = "tacofoundation:cloudsen12-l1c"
    NUM_CLASSES = 4  # clear, thick cloud, thin cloud, cloud shadow

    # Optimization parameters
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 4  # Further reduced to minimize server load
    ACCUM_STEPS = 16  # Increased to maintain effective batch size
    EPOCHS = 30
    LEARNING_RATE = 0.0005
    MODEL_SAVE_PATH = "best_cloud_mask_model.pth"
    PATIENCE = 7  # Increased patience due to unstable training
    NUM_WORKERS = 1  # Minimal workers to reduce concurrent requests
    SEED = 42
    TRAIN_SPLIT = 0.8
    MAX_RETRIES = 3  # Reduced retries - fail fast approach
    RETRY_DELAY = 10  # Longer base delay
    MAX_RETRY_DELAY = 120  # Much longer max delay
    SKIP_THRESHOLD = 0.3  # Allow more failed samples
    RATE_LIMIT_COOLDOWN = 300  # 5 minute cooldown when rate limited
    MAX_FAILED_SAMPLES = 1000  # Stop trying after this many failures
    SUBSET_SIZE = 5000  # Use smaller subset for training

    # Selected bands (RGB + SWIR + Cirrus)
    SELECTED_BANDS = [2, 3, 4, 8, 11, 12]

# Set random seed and enable optimizations
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
torch.backends.cudnn.benchmark = True

# ------------------------- Retry Decorator -------------------------
def retry_with_backoff(max_retries=5, base_delay=2, max_delay=30):
    """Decorator for exponential backoff retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    print(f"Attempt {attempt + 1} failed: {str(e)[:100]}... Retrying in {delay:.1f}s")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# ------------------------- Lightweight U-Net Architecture -------------------------
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class LightweightUNet(nn.Module):
    """Faster U-Net with fewer parameters"""
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        # Decoder
        self.up1 = Up(256 + 128, 128)
        self.up2 = Up(128 + 64, 64)
        self.up3 = Up(64 + 32, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

# ------------------------- Synthetic Dataset (Fallback) -------------------------
class SyntheticCloudDataset(Dataset):
    """Synthetic dataset for testing when real data is unavailable"""
    def __init__(self, split='train', size=1000):
        self.split = split
        self.size = size
        print(f"Created synthetic {split} dataset with {size} samples")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Create synthetic satellite-like image
        np.random.seed(idx)  # Reproducible samples
        
        # Generate base image with multiple bands
        image = np.random.rand(len(Config.SELECTED_BANDS), 
                              Config.IMAGE_SIZE[0], 
                              Config.IMAGE_SIZE[1]).astype(np.float32)
        
        # Add some realistic patterns
        x, y = np.meshgrid(np.linspace(0, 10, Config.IMAGE_SIZE[1]), 
                          np.linspace(0, 10, Config.IMAGE_SIZE[0]))
        
        # Add cloud-like patterns
        cloud_pattern = np.sin(x) * np.cos(y) + np.random.normal(0, 0.1, x.shape)
        
        # Create mask based on patterns
        mask = np.zeros((Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]), dtype=np.uint8)
        mask[cloud_pattern > 0.5] = 1  # thick cloud
        mask[(cloud_pattern > 0.2) & (cloud_pattern <= 0.5)] = 2  # thin cloud
        mask[(cloud_pattern > -0.2) & (cloud_pattern <= 0.2)] = 3  # cloud shadow
        # rest remains 0 (clear)
        
        return torch.tensor(image), torch.tensor(mask, dtype=torch.long)
class CloudSEN12Dataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        
        # Initialize dataset with retry
        self.dataset = self._init_dataset_with_retry()
        self.num_samples = len(self.dataset)

        # Create indices and split - use smaller subset
        indices = list(range(min(self.num_samples, Config.SUBSET_SIZE)))
        np.random.shuffle(indices)
        split_idx = int(Config.TRAIN_SPLIT * len(indices))

        if split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        # Keep track of failed samples and rate limiting
        self.failed_samples = set()
        self.success_cache = {}  # Cache successful samples
        self.rate_limited_until = 0  # Timestamp when rate limiting ends
        self.total_failures = 0
        self.consecutive_failures = 0

        print(f"Loaded {len(self.indices)} samples for {split} split (subset of {Config.SUBSET_SIZE})")

    @retry_with_backoff(max_retries=3, base_delay=5, max_delay=60)
    def _init_dataset_with_retry(self):
        """Initialize dataset with retry logic"""
        return tacoreader.load(Config.DATASET_NAME)

    def __len__(self):
        return len(self.indices)

    def _is_rate_limited_error(self, error_str):
        """Check if error is due to rate limiting"""
        rate_limit_indicators = [
            "429", "rate limit", "too many requests", 
            "HTTP response code", "throttle"
        ]
        return any(indicator in error_str.lower() for indicator in rate_limit_indicators)

    def _should_skip_due_to_rate_limit(self):
        """Check if we should skip requests due to rate limiting"""
        current_time = time.time()
        if current_time < self.rate_limited_until:
            return True
        return False

    @retry_with_backoff(max_retries=Config.MAX_RETRIES, 
                       base_delay=Config.RETRY_DELAY, 
                       max_delay=Config.MAX_RETRY_DELAY)
    def _load_sample_data(self, sample_idx):
        """Load sample data with retry logic"""
        # Check if we're in rate limit cooldown
        if self._should_skip_due_to_rate_limit():
            raise Exception("Rate limited - in cooldown period")
            
        sample = self.dataset.read(sample_idx)
        
        # Read image data
        with rio.open(sample.read(0)) as src:
            image = src.read(Config.SELECTED_BANDS)
            image = image.astype(np.float32) / 10000.0

        # Read mask data
        with rio.open(sample.read(1)) as dst:
            mask = dst.read(1).astype(np.uint8)

        return image, mask

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        
        # Skip if we know this sample consistently fails
        if sample_idx in self.failed_samples or self.total_failures > Config.MAX_FAILED_SAMPLES:
            return self._get_placeholder()

        # Check cache first
        if sample_idx in self.success_cache:
            self.consecutive_failures = 0  # Reset on success
            return self.success_cache[sample_idx]

        # Skip if we're rate limited
        if self._should_skip_due_to_rate_limit():
            return self._get_placeholder()

        try:
            # Load data with retry
            image, mask = self._load_sample_data(sample_idx)
            
            # Resize images
            image = self._resize_image(image, Config.IMAGE_SIZE)
            mask = self._resize_mask(mask, Config.IMAGE_SIZE)
            
            # Cache successful load
            result = (image, mask)
            if len(self.success_cache) < 500:  # Reduced cache size
                self.success_cache[sample_idx] = result
            
            # Reset failure counters on success
            self.consecutive_failures = 0
            return result

        except Exception as e:
            error_str = str(e)
            self.total_failures += 1
            self.consecutive_failures += 1
            
            # Handle rate limiting
            if self._is_rate_limited_error(error_str):
                self.rate_limited_until = time.time() + Config.RATE_LIMIT_COOLDOWN
                print(f"Rate limited detected. Cooling down for {Config.RATE_LIMIT_COOLDOWN}s")
            
            # Mark as failed after repeated attempts
            if self.consecutive_failures > 3:
                self.failed_samples.add(sample_idx)
                print(f"Sample {sample_idx} marked as permanently failed after {self.consecutive_failures} consecutive failures")
            
            # Print less frequent error messages
            if self.total_failures % 50 == 1:  # Every 50th failure
                print(f"Total failures: {self.total_failures}. Recent error: {error_str[:100]}...")
            
            return self._get_placeholder()

    def _get_placeholder(self):
        """Return a placeholder sample"""
        placeholder_image = torch.zeros(
            len(Config.SELECTED_BANDS), Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 
            dtype=torch.float32
        )
        placeholder_mask = torch.zeros(
            Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], dtype=torch.long
        )
        return placeholder_image, placeholder_mask

    def _resize_image(self, image, target_size):
        """Resize multi-band image using PIL"""
        resized_bands = []
        for band in image:
            band_img = Image.fromarray(band).resize(
                (target_size[1], target_size[0]), resample=Image.BILINEAR
            )
            resized_bands.append(np.array(band_img))
        return torch.tensor(np.stack(resized_bands), dtype=torch.float32)

    def _resize_mask(self, mask, target_size):
        """Resize mask using nearest neighbor"""
        mask_img = Image.fromarray(mask).resize(
            (target_size[1], target_size[0]), resample=Image.NEAREST
        )
        mask_arr = np.array(mask_img)
        mask_arr = np.clip(mask_arr, 0, Config.NUM_CLASSES - 1)
        return torch.tensor(mask_arr, dtype=torch.long)

# ------------------------- Training Utilities -------------------------
class EarlyStopper:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model = None

    def __call__(self, validation_loss, model):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model = model.state_dict()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

def calculate_metrics(pred, target):
    """Calculate comprehensive segmentation metrics"""
    pred = pred.flatten()
    target = target.flatten()

    if np.sum(target) == 0:
        accuracy = accuracy_score(target, pred)
        return accuracy, 0, 0

    iou = jaccard_score(target, pred, average='weighted', zero_division=0)
    f1 = f1_score(target, pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(target, pred)

    return accuracy, iou, f1

def collate_fn_with_filtering(batch):
    """Custom collate function that filters out placeholder samples"""
    valid_samples = []
    placeholder_count = 0
    
    for image, mask in batch:
        # Check if sample is not a placeholder (not all zeros)
        if image.sum() > 0:
            valid_samples.append((image, mask))
        else:
            placeholder_count += 1
    
    # If too many placeholders, we might be rate limited
    if placeholder_count > len(batch) * 0.8:  # More than 80% are placeholders
        print(f"Warning: {placeholder_count}/{len(batch)} samples are placeholders. Possible rate limiting.")
    
    if len(valid_samples) == 0:
        # Return a single placeholder if all samples are invalid
        placeholder_image = torch.zeros(1, len(Config.SELECTED_BANDS), 
                                      Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])
        placeholder_mask = torch.zeros(1, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 
                                     dtype=torch.long)
        return placeholder_image, placeholder_mask
    
    # Stack valid samples
    images = torch.stack([sample[0] for sample in valid_samples])
    masks = torch.stack([sample[1] for sample in valid_samples])
    
    return images, masks

# ------------------------- Main Training Function -------------------------
def main():
    class Args:
        batch_size = Config.BATCH_SIZE
        epochs = Config.EPOCHS
        lr = Config.LEARNING_RATE
        save_dir = Config.MODEL_SAVE_PATH
        output_dir = "results"
    args = Args()

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets with improved error handling
    print("Initializing datasets...")
    try:
        train_dataset = CloudSEN12Dataset(split='train')
        val_dataset = CloudSEN12Dataset(split='val')
        
        # Check if we have too many failures - switch to synthetic data
        if (hasattr(train_dataset, 'total_failures') and 
            train_dataset.total_failures > Config.MAX_FAILED_SAMPLES // 2):
            print("Too many failures detected. Switching to synthetic dataset for testing...")
            train_dataset = SyntheticCloudDataset(split='train', size=2000)
            val_dataset = SyntheticCloudDataset(split='val', size=500)
            
    except Exception as e:
        print(f"Failed to initialize real dataset: {e}")
        print("Using synthetic dataset instead...")
        train_dataset = SyntheticCloudDataset(split='train', size=2000)
        val_dataset = SyntheticCloudDataset(split='val', size=500)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn_with_filtering,
        timeout=120,  # Increased timeout
        drop_last=True  # Drop incomplete batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn_with_filtering,
        timeout=120,
        drop_last=False
    )

    # Initialize lightweight model
    model = LightweightUNet(
        n_channels=len(Config.SELECTED_BANDS),
        n_classes=Config.NUM_CLASSES
    ).to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Loss function with class weighting
    class_weights = torch.tensor([1.0, 2.0, 3.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Early stopper
    early_stopper = EarlyStopper(patience=Config.PATIENCE)

    # Mixed precision scaler
    scaler = amp.GradScaler()

    # Training loop
    best_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'iou': [], 'f1': []}

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        step = 0
        valid_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # Add pause mechanism for rate limiting
        consecutive_empty_batches = 0

        for batch_idx, (images, masks) in enumerate(pbar):
            # Skip if we have a placeholder batch
            if images.size(0) == 1 and images.sum() == 0:
                consecutive_empty_batches += 1
                if consecutive_empty_batches > 10:
                    print(f"Too many consecutive empty batches ({consecutive_empty_batches}). Pausing for rate limit recovery...")
                    time.sleep(Config.RATE_LIMIT_COOLDOWN // 2)  # Shorter pause
                    consecutive_empty_batches = 0
                continue
            else:
                consecutive_empty_batches = 0  # Reset counter

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Mixed precision forward
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks) / Config.ACCUM_STEPS

            # Backward pass with gradient accumulation
            scaler.scale(loss).backward()

            if (step + 1) % Config.ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * Config.ACCUM_STEPS * images.size(0)
            valid_batches += 1
            step += 1

            # Update progress bar
            pbar.set_postfix(loss=loss.item() * Config.ACCUM_STEPS, 
                           valid_batches=valid_batches)

        # Handle remaining gradients
        if step % Config.ACCUM_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if valid_batches > 0:
            train_loss /= valid_batches
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        valid_val_batches = 0

        with torch.no_grad():
            for images, masks in val_loader:
                # Skip placeholder batches
                if images.size(0) == 1 and images.sum() == 0:
                    continue

                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)
                valid_val_batches += 1

                # Calculate metrics
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu().numpy().flatten())
                all_targets.append(masks.cpu().numpy().flatten())

        if valid_val_batches > 0:
            val_loss /= valid_val_batches
        history['val_loss'].append(val_loss)

        # Calculate metrics
        if all_preds and all_targets:
            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)
            accuracy, iou, f1 = calculate_metrics(all_preds, all_targets)
        else:
            accuracy, iou, f1 = 0, 0, 0

        history['iou'].append(iou)
        history['f1'].append(f1)

        # Update scheduler
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"IoU: {iou:.4f} | "
              f"F1: {f1:.4f} | "
              f"Accuracy: {accuracy:.4f} | "
              f"Valid batches: {valid_batches}/{len(train_loader)}")

        # Save best model
        if iou > best_iou:
            best_iou = iou
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, args.save_dir))
            print(f"Saved best model with IoU: {best_iou:.4f}")

        # Check early stopping
        if early_stopper(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            model.load_state_dict(early_stopper.best_model)
            break

    print("Training completed!")

    # Save training history
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "training_history.npy"), history)

    # Plot training history
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(history['iou'], label='IoU')
    plt.plot(history['f1'], label='F1 Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_history.png"))
    plt.close()

if __name__ == "__main__":
    main()