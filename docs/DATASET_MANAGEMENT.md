# Dataset Management Guide

This guide explains how to handle large image datasets (10,000+ images) with GitHub efficiently.

## 🎯 Problem Statement

Large image datasets pose several challenges:
- **Git Performance**: Adding thousands of images slows down Git operations
- **Repository Size**: Large files bloat the repository
- **Clone Time**: Users spend significant time downloading large datasets
- **Storage Limits**: GitHub has file size and repository size limits

## 🚀 Solutions Overview

### 1. **Git LFS (Large File Storage)** - Recommended ⭐

Git LFS is the best solution for managing large files in Git repositories.

#### Setup Git LFS

```bash
# Install Git LFS
git lfs install

# Track image files
git lfs track "*.jpg" "*.jpeg" "*.png" "*.gif" "*.webp" "*.bmp" "*.tiff" "*.svg"
git lfs track "dataset/**/*.jpg" "dataset/**/*.jpeg" "dataset/**/*.png"

# Add .gitattributes file
git add .gitattributes
git commit -m "feat: configure Git LFS for image files"
```

#### Benefits
- ✅ **Efficient**: Only downloads files when needed
- ✅ **Transparent**: Works seamlessly with Git commands
- ✅ **Scalable**: Handles millions of files
- ✅ **Free**: GitHub provides 1GB free LFS storage

#### Usage
```bash
# Add images (automatically handled by LFS)
git add dataset/
git commit -m "feat: add fashion dataset images"

# Clone with LFS files
git lfs clone https://github.com/ashleyashok/fashion-knowledge-graph.git

# Pull LFS files
git lfs pull
```

### 2. **External Storage + Scripts** - Alternative

Store images externally and provide download scripts.

#### Option A: Cloud Storage (AWS S3, Google Cloud)

```python
# scripts/download_dataset.py
import boto3
import os
from tqdm import tqdm

def download_dataset():
    """Download dataset from S3 bucket."""
    s3 = boto3.client('s3')
    bucket_name = 'fashion-knowledge-graph-dataset'
    
    # List all objects in dataset folder
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix='dataset/')
    
    for page in pages:
        for obj in page['Contents']:
            key = obj['Key']
            local_path = key
            
            # Create directory if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            s3.download_file(bucket_name, key, local_path)
            print(f"Downloaded: {key}")

if __name__ == "__main__":
    download_dataset()
```

#### Option B: Hugging Face Datasets

```python
# scripts/download_hf_dataset.py
from datasets import load_dataset
import os

def download_hf_dataset():
    """Download dataset from Hugging Face."""
    dataset = load_dataset("ashleyashok/fashion-knowledge-graph-dataset")
    
    # Download to local directory
    dataset.save_to_disk("dataset/")
    print("Dataset downloaded successfully!")

if __name__ == "__main__":
    download_hf_dataset()
```

### 3. **GitHub Releases** - For Large Files

Upload dataset as a compressed release asset.

```bash
# Create dataset archive
tar -czf fashion-dataset.tar.gz dataset/

# Upload to GitHub Releases via web interface
# Or use GitHub CLI:
gh release create v1.0.0 fashion-dataset.tar.gz --title "Fashion Dataset v1.0"
```

### 4. **DVC (Data Version Control)** - Advanced

DVC is designed specifically for data versioning.

```bash
# Install DVC
pip install dvc

# Initialize DVC
dvc init

# Add dataset to DVC
dvc add dataset/

# Commit DVC files
git add dataset/.gitignore dataset/.dvc
git commit -m "feat: add dataset with DVC"
```

## 📊 Comparison of Solutions

| Solution | Pros | Cons | Best For |
|----------|------|------|----------|
| **Git LFS** | ✅ Easy setup<br>✅ Transparent<br>✅ Free tier | ❌ Storage limits<br>❌ Bandwidth limits | Most use cases |
| **Cloud Storage** | ✅ Unlimited storage<br>✅ Fast downloads<br>✅ Cost-effective | ❌ External dependency<br>❌ Setup complexity | Large datasets |
| **GitHub Releases** | ✅ Simple<br>✅ No limits<br>✅ Versioned | ❌ Manual process<br>❌ No incremental updates | Occasional updates |
| **DVC** | ✅ Advanced features<br>✅ Cloud integration<br>✅ Data pipelines | ❌ Learning curve<br>❌ Setup complexity | Data science teams |

## 🛠️ Implementation Guide

### Step 1: Choose Your Approach

**For most projects**: Use Git LFS
**For very large datasets**: Use Cloud Storage + scripts
**For data science teams**: Consider DVC

### Step 2: Setup Git LFS (Recommended)

```bash
# 1. Install and configure
git lfs install
git lfs track "*.jpg" "*.jpeg" "*.png" "*.gif" "*.webp"

# 2. Add configuration
git add .gitattributes
git commit -m "feat: configure Git LFS"

# 3. Add your dataset
git add dataset/
git commit -m "feat: add fashion dataset"

# 4. Push to GitHub
git push origin main
```

### Step 3: Update Documentation

Update your README.md to include dataset instructions:

```markdown
## 📊 Dataset

### Downloading the Dataset

The fashion dataset contains 10,000+ images and is managed with Git LFS.

```bash
# Clone with LFS files
git lfs clone https://github.com/ashleyashok/fashion-knowledge-graph.git

# Or clone normally and pull LFS files
git clone https://github.com/ashleyashok/fashion-knowledge-graph.git
cd fashion-knowledge-graph
git lfs pull
```

### Dataset Structure

```
dataset/
├── catalog_images/          # Product catalog images
├── social_media_images/     # Social media fashion images
├── test_images/            # Test and validation images
└── metadata/               # Image metadata and annotations
```

### Alternative Download Methods

If you prefer not to use Git LFS:

1. **Download from Cloud Storage**:
   ```bash
   python scripts/download_dataset.py
   ```

2. **Download from Hugging Face**:
   ```bash
   python scripts/download_hf_dataset.py
   ```

3. **Download from GitHub Releases**:
   - Go to [Releases](https://github.com/ashleyashok/fashion-knowledge-graph/releases)
   - Download `fashion-dataset.tar.gz`
   - Extract: `tar -xzf fashion-dataset.tar.gz`
```

## 🔧 Advanced Configuration

### Git LFS Configuration

```bash
# Set LFS batch size for better performance
git config lfs.batchsize 1000

# Set LFS transfer mode
git config lfs.transfer.mode basic

# Check LFS status
git lfs status
git lfs ls-files
```

### Performance Optimization

```bash
# Clone without LFS files (faster)
git clone --no-checkout https://github.com/ashleyashok/fashion-knowledge-graph.git
cd fashion-knowledge-graph
git lfs pull --include="dataset/catalog_images"

# Pull specific folders only
git lfs pull --include="dataset/catalog_images" --exclude="dataset/social_media_images"
```

### Monitoring LFS Usage

```bash
# Check LFS storage usage
git lfs track

# List all LFS files
git lfs ls-files

# Check LFS file sizes
git lfs ls-files -s
```

## 🚨 Important Notes

### GitHub Limits
- **File size**: 100MB per file (LFS: 2GB)
- **Repository size**: 1GB recommended (LFS: 1GB free)
- **Bandwidth**: 1GB/month free for LFS

### Best Practices
1. **Compress images** before adding to repository
2. **Use appropriate formats** (JPEG for photos, PNG for graphics)
3. **Organize datasets** in logical folder structure
4. **Document dataset** with README files
5. **Version datasets** with meaningful commit messages

### Troubleshooting

```bash
# Fix LFS issues
git lfs migrate import --include="*.jpg,*.png"

# Clean up LFS cache
git lfs prune

# Reset LFS files
git lfs uninstall
git lfs install
```

## 📚 Additional Resources

- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub LFS Guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
- [DVC Documentation](https://dvc.org/doc)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

---

For questions about dataset management, please refer to the [Contributing Guidelines](../CONTRIBUTING.md) or contact the maintainers.

