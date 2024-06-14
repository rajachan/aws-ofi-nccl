/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_MR_H_
#define NCCL_OFI_MR_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <pthread.h>

/*
 * Initial size of the MR cache. The cache will grow as needed (with a
 * realloc()) in the registration path if more entries need to be held. Using
 * the same default NCCL uses.
 */
#define NCCL_OFI_MR_CACHE_SIZE 128

/**
 * A memory registration cache entry
 */
typedef struct nccl_ofi_reg_entry {
	uintptr_t addr;
	size_t pages;
	int refcnt;
	void *handle;
} nccl_ofi_reg_entry_t;

/**
 * Device-specific memory registration cache.
 */
typedef struct nccl_ofi_mr_cache {
	nccl_ofi_reg_entry_t **slots;
	int size;
	int used;
	pthread_mutex_t lock;
} nccl_ofi_mr_cache_t;


int nccl_ofi_mr_cache_init(nccl_ofi_mr_cache_t *cache, int size);
void nccl_ofi_mr_cache_finalize(nccl_ofi_mr_cache_t *cache);

int nccl_ofi_mr_cache_grow(nccl_ofi_mr_cache_t *cache);

int nccl_ofi_mr_cache_add_entry(nccl_ofi_mr_cache_t *cache,
				void *data,
				int size,
				int type,
				void **handle);
int nccl_ofi_mr_cache_del_entry(nccl_ofi_mr_cache_t *cache, void *handle);

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_MR_H_
