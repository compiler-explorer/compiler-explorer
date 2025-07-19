#!/usr/bin/env python3
"""
Migrate Compiler Explorer shortlinks from local storage to AWS S3/DynamoDB.

Usage:
    python migrate_shortlinks.py \
        --local-storage-dir ./lib/storage/data/ \
        --s3-bucket storage.godbolt.org \
        --s3-prefix ce/ \
        --dynamodb-table links \
        --aws-region us-east-1 \
        --dry-run
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import boto3
from boto3.dynamodb.conditions import Key
from dataclasses import dataclass
from collections import defaultdict

# Constants matching the TypeScript implementation
PREFIX_LENGTH = 6  # NEVER CHANGE - DynamoDB partition key
MIN_STORED_ID_LENGTH_S3 = 9  # Minimum for S3 storage
MIN_STORED_ID_LENGTH_LOCAL = 6  # Minimum for local storage

@dataclass
class LocalShortlink:
    """Represents a shortlink from local storage."""
    file_path: Path
    unique_subhash: str  # The filename
    full_hash: str
    config: str
    prefix: str
    creation_time: datetime
    
@dataclass
class MigrationStats:
    """Track migration statistics."""
    total_files: int = 0
    migrated: int = 0
    already_exists: int = 0
    collisions_resolved: int = 0
    errors: int = 0
    skipped: int = 0

class ShortlinkMigrator:
    def __init__(self, s3_bucket: str, s3_prefix: str, dynamodb_table: str, 
                 region: str, dry_run: bool = False):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip('/') + '/' if s3_prefix else ''
        self.dynamodb_table = dynamodb_table
        self.dry_run = dry_run
        self.stats = MigrationStats()
        
        if not dry_run:
            self.s3_client = boto3.client('s3', region_name=region)
            self.dynamodb = boto3.resource('dynamodb', region_name=region)
            self.table = self.dynamodb.Table(dynamodb_table)
        
        self.logger = logging.getLogger(__name__)
    
    def load_local_shortlinks(self, storage_dir: Path) -> List[LocalShortlink]:
        """Load all shortlinks from local storage, sorted by creation time."""
        shortlinks = []
        
        for file_path in storage_dir.iterdir():
            if not file_path.is_file():
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Get file creation time
                stat = file_path.stat()
                creation_time = datetime.fromtimestamp(stat.st_ctime)
                
                # The filename IS the uniqueSubHash
                filename = file_path.name
                
                # Validate that the stored uniqueSubHash matches filename
                if data.get('uniqueSubHash') != filename:
                    self.logger.warning(
                        f"Filename {filename} doesn't match stored uniqueSubHash {data.get('uniqueSubHash')}"
                    )
                
                shortlink = LocalShortlink(
                    file_path=file_path,
                    unique_subhash=filename,
                    full_hash=data['fullHash'],
                    config=data['config'],
                    prefix=data['prefix'],
                    creation_time=creation_time
                )
                    
                shortlinks.append(shortlink)
                
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                self.stats.errors += 1
                
        # Sort by creation time to maintain order
        shortlinks.sort(key=lambda x: x.creation_time)
        self.stats.total_files = len(shortlinks)
        
        return shortlinks
    
    def get_existing_subhashes_for_prefix(self, prefix: str) -> Dict[str, str]:
        """Query DynamoDB for all existing subhashes with given prefix."""
        if self.dry_run:
            return {}
            
        existing = {}
        try:
            response = self.table.query(
                KeyConditionExpression=Key('prefix').eq(prefix),
                ProjectionExpression='unique_subhash, full_hash'
            )
            
            for item in response.get('Items', []):
                existing[item['unique_subhash']] = item['full_hash']
                
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.table.query(
                    KeyConditionExpression=Key('prefix').eq(prefix),
                    ProjectionExpression='unique_subhash, full_hash',
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                for item in response.get('Items', []):
                    existing[item['unique_subhash']] = item['full_hash']
                    
        except Exception as e:
            self.logger.error(f"Error querying DynamoDB for prefix {prefix}: {e}")
            
        return existing
    
    def find_unique_subhash_for_s3(self, full_hash: str, existing_subhashes: Dict[str, str], 
                                   original_subhash: str) -> str:
        """
        For migration, we must preserve the original subhash to maintain URL compatibility.
        We only check if it already exists with the same content (deduplication).
        """
        # Check if the original subhash already exists
        if original_subhash in existing_subhashes:
            if existing_subhashes[original_subhash] == full_hash:
                # Same content already exists - this is fine
                return original_subhash
            else:
                # Different content with same subhash - this should never happen
                # as it would mean a hash collision in the original system
                raise ValueError(f"Hash collision detected for {original_subhash}!")
        
        # Original subhash is unique, use it as-is
        return original_subhash
    
    def migrate_shortlink(self, shortlink: LocalShortlink, 
                         prefix_cache: Dict[str, Dict[str, str]]) -> bool:
        """Migrate a single shortlink to AWS."""
        try:
            # S3 storage uses first 6 characters of full hash as prefix
            s3_prefix = shortlink.full_hash[:PREFIX_LENGTH]
            
            # Get cached subhashes for this prefix, or query if not cached
            if s3_prefix not in prefix_cache:
                prefix_cache[s3_prefix] = self.get_existing_subhashes_for_prefix(s3_prefix)
            
            existing_subhashes = prefix_cache[s3_prefix]
            
            # Preserve the original subhash for migration
            new_subhash = self.find_unique_subhash_for_s3(shortlink.full_hash, existing_subhashes, 
                                                          shortlink.unique_subhash)
            
            # Check if already exists with same content
            if new_subhash in existing_subhashes and existing_subhashes[new_subhash] == shortlink.full_hash:
                self.logger.info(f"Already exists: {shortlink.unique_subhash} -> {new_subhash}")
                self.stats.already_exists += 1
                return True
            
            # Check for hash collision (should never happen in practice)
            if new_subhash == shortlink.unique_subhash and new_subhash in existing_subhashes:
                if existing_subhashes[new_subhash] != shortlink.full_hash:
                    self.logger.error(f"Hash collision detected for {new_subhash}")
                    self.stats.errors += 1
                    return False
            
            # Perform migration
            if not self.dry_run:
                # S3 upload
                s3_key = f"{self.s3_prefix}{shortlink.full_hash}"
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=shortlink.config.encode('utf-8'),
                    ContentType='application/json'
                )
                
                # DynamoDB entry
                self.table.put_item(
                    Item={
                        'prefix': s3_prefix,
                        'unique_subhash': new_subhash,
                        'full_hash': shortlink.full_hash,
                        'stats': {'clicks': 0},
                        'creation_ip': 'migrated-from-local',
                        'creation_date': shortlink.creation_time.isoformat()
                    }
                )
                
            # Update cache
            existing_subhashes[new_subhash] = shortlink.full_hash
            
            self.logger.info(f"Migrated: {shortlink.unique_subhash} -> {new_subhash}")
            self.stats.migrated += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error migrating {shortlink.unique_subhash}: {e}")
            self.stats.errors += 1
            return False
    
    def run_migration(self, storage_dir: Path, batch_size: int = 100):
        """Run the full migration process."""
        self.logger.info(f"Starting migration from {storage_dir}")
        
        # Load all shortlinks
        shortlinks = self.load_local_shortlinks(storage_dir)
        self.logger.info(f"Found {len(shortlinks)} shortlinks to migrate")
        
        # Cache of existing subhashes per prefix to minimize DynamoDB queries
        prefix_cache = {}
        
        # Process in batches for logging
        for i in range(0, len(shortlinks), batch_size):
            batch = shortlinks[i:i + batch_size]
            self.logger.info(f"Processing batch {i // batch_size + 1} ({i+1}-{i + len(batch)} of {len(shortlinks)})")
            
            for shortlink in batch:
                self.migrate_shortlink(shortlink, prefix_cache)
        
        # Print summary
        self.print_summary()
    
    def verify_migration(self, storage_dir: Path, sample_size: int = 10):
        """Verify a sample of migrated links."""
        if self.dry_run:
            return
            
        self.logger.info(f"\nVerifying migration (sample size: {sample_size})")
        
        shortlinks = self.load_local_shortlinks(storage_dir)
        import random
        sample = random.sample(shortlinks, min(sample_size, len(shortlinks)))
        
        verified = 0
        for shortlink in sample:
            try:
                # Check S3
                s3_key = f"{self.s3_prefix}{shortlink.full_hash}"
                
                response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
                s3_config = response['Body'].read().decode('utf-8')
                
                if s3_config == shortlink.config:
                    self.logger.info(f"✓ Verified {shortlink.unique_subhash}")
                    verified += 1
                else:
                    self.logger.error(f"✗ Content mismatch for {shortlink.unique_subhash}")
                    
            except Exception as e:
                self.logger.error(f"✗ Failed to verify {shortlink.unique_subhash}: {e}")
        
        self.logger.info(f"Verified {verified}/{len(sample)} samples")
    
    def print_summary(self):
        """Print migration summary."""
        print("\n=== Migration Summary ===")
        print(f"Total files found: {self.stats.total_files}")
        print(f"Successfully migrated: {self.stats.migrated}")
        print(f"Already exists: {self.stats.already_exists}")
        print(f"Collisions resolved: {self.stats.collisions_resolved}")
        print(f"Errors: {self.stats.errors}")
        print(f"Skipped: {self.stats.skipped}")
        print(f"Dry run: {self.dry_run}")

def main():
    parser = argparse.ArgumentParser(description='Migrate Compiler Explorer shortlinks to AWS')
    parser.add_argument('--local-storage-dir', required=True, help='Local storage directory')
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket name')
    parser.add_argument('--s3-prefix', default='', help='S3 key prefix')
    parser.add_argument('--dynamodb-table', required=True, help='DynamoDB table name')
    parser.add_argument('--aws-region', required=True, help='AWS region')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--verify', action='store_true', help='Verify migration with random samples')
    parser.add_argument('--verify-sample-size', type=int, default=10, help='Number of samples to verify')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run without writing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run migration
    migrator = ShortlinkMigrator(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        dynamodb_table=args.dynamodb_table,
        region=args.aws_region,
        dry_run=args.dry_run
    )
    
    storage_dir = Path(args.local_storage_dir)
    migrator.run_migration(storage_dir, batch_size=args.batch_size)
    
    # Optional verification
    if args.verify and not args.dry_run:
        migrator.verify_migration(storage_dir, args.verify_sample_size)

if __name__ == '__main__':
    main()