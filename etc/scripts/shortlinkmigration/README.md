# Shortlink Migration Script

This script migrates Compiler Explorer shortlinks from local file storage to AWS S3/DynamoDB.

## Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- AWS credentials configured (via AWS CLI, environment variables, or IAM role)

## Installation

Install dependencies using Poetry:

```bash
cd etc/scripts/shortlinkmigration
poetry install
```

## Usage

### Dry Run (recommended first)
```bash
poetry run python migrate_shortlinks.py \
    --local-storage-dir ./lib/storage/data/ \
    --s3-bucket storage.godbolt.org \
    --s3-prefix ce/ \
    --dynamodb-table links \
    --aws-region us-east-1 \
    --dry-run
```

### Actual Migration
```bash
poetry run python migrate_shortlinks.py \
    --local-storage-dir ./lib/storage/data/ \
    --s3-bucket storage.godbolt.org \
    --s3-prefix ce/ \
    --dynamodb-table links \
    --aws-region us-east-1
```

### With Verification
```bash
poetry run python migrate_shortlinks.py \
    --local-storage-dir ./lib/storage/data/ \
    --s3-bucket storage.godbolt.org \
    --s3-prefix ce/ \
    --dynamodb-table links \
    --aws-region us-east-1 \
    --verify \
    --verify-sample-size 20
```

## Options

- `--local-storage-dir`: Path to local storage directory (required)
- `--s3-bucket`: S3 bucket name (required)
- `--s3-prefix`: S3 key prefix (optional, default: '')
- `--dynamodb-table`: DynamoDB table name (required)
- `--aws-region`: AWS region (required)
- `--batch-size`: Number of files to process in each batch (default: 100)
- `--verify`: Run verification after migration
- `--verify-sample-size`: Number of random samples to verify (default: 10)
- `--dry-run`: Simulate migration without making changes
- `--verbose`: Enable detailed logging

## How it Works

1. **Loading**: Reads all files from local storage directory
2. **Sorting**: Sorts files by creation time to preserve chronological order
3. **Deduplication**: Checks if content already exists in DynamoDB
4. **Collision Handling**: Extends subhash length if collisions occur (local uses 6+ chars, S3 uses 9+ chars)
5. **Migration**: 
   - Uploads config to S3: `{prefix}/{fullHash}`
   - Creates DynamoDB entry with metadata
6. **Verification**: Optionally verifies random samples

## Migration Details

### Local Storage Structure
- Files named by their unique subhash (minimum 6 characters)
- Each file contains: `{prefix, uniqueSubHash, fullHash, config}`

### S3/DynamoDB Structure
- S3 key: `{s3_prefix}{6-char-prefix}/{fullHash}`
- DynamoDB partition key: 6-character prefix
- DynamoDB sort key: unique_subhash (minimum 9 characters)

## Notes

- The script preserves creation timestamps from file metadata
- Existing entries are skipped (deduplication)
- Collision resolution may result in longer subhashes
- Progress is logged every batch_size entries
- Errors are logged but don't stop the migration