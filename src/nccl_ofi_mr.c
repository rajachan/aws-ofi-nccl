/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <errno.h>
#include "nccl_ofi.h"
#include "nccl_ofi_mr.h"
#include "nccl_ofi_pthread.h"


inline nccl_net_ofi_comm_t* nccl_ofi_mr_get_comm_from_cache(nccl_ofi_mr_cache_t *cache)
{
	nccl_net_ofi_device_t *dev = container_of(cache, nccl_net_ofi_device_t, mr_cache);
	nccl_net_ofi_ep_t *ep = container_of(dev, nccl_net_ofi_ep_t, device);
	nccl_net_ofi_comm_t *comm = container_of(ep, nccl_net_ofi_comm_t, ep);
	return comm;
}

int nccl_ofi_mr_reg_comm(nccl_ofi_mr_cache_t *cache, void *addr, int size, int type, void **mhandle)
{
	nccl_net_ofi_comm_t *comm = nccl_ofi_mr_get_comm_from_cache(cache);
	int ret = 0;

	/* Register with network */
	switch (comm->type) {
	case NCCL_NET_OFI_SEND_COMM:;
		nccl_net_ofi_send_comm_t *send_comm =
			(nccl_net_ofi_send_comm_t *)comm;
		ret = send_comm->regMr(send_comm, addr, size, type, mhandle);
		break;
	case NCCL_NET_OFI_RECV_COMM:;
		nccl_net_ofi_recv_comm_t *recv_comm =
			(nccl_net_ofi_recv_comm_t *)comm;
		ret = recv_comm->regMr(recv_comm, addr, size, type, mhandle);
		break;
	default:
		NCCL_OFI_WARN("Unexpected communicator type. Communicator type: %d",
			      comm->type);
		ret = -EINVAL;
		goto out;
	}

out:
	return ret;
}


int nccl_ofi_mr_dereg_comm(nccl_ofi_mr_cache_t *cache, void *mhandle)
{
	nccl_net_ofi_comm_t *comm = nccl_ofi_mr_get_comm_from_cache(cache);
	int ret = 0;

	switch (comm->type) {
	case NCCL_NET_OFI_SEND_COMM:;
		nccl_net_ofi_send_comm_t *send_comm =
			(nccl_net_ofi_send_comm_t *)comm;
		ret = send_comm->deregMr(send_comm, mhandle);
		goto out;
	case NCCL_NET_OFI_RECV_COMM:;
		nccl_net_ofi_recv_comm_t *recv_comm =
			(nccl_net_ofi_recv_comm_t *)comm;
		ret = recv_comm->deregMr(recv_comm, mhandle);
		goto out;
	default:
		NCCL_OFI_WARN("Unexpected communicator type. Communicator type: %d",
			      comm->type);
		ret = -EINVAL;
		goto out;
	}
out:
	return ret;
}

int nccl_ofi_mr_cache_init(nccl_ofi_mr_cache_t *cache, int size)
{
	int ret = 0;

	cache = calloc(1, sizeof(*cache));
	if (!cache) {
		ret = errno;
		goto out;
	}

	ret = nccl_net_ofi_mutex_init(&cache->lock, NULL);
	if (OFI_UNLIKELY(ret)) {
		NCCL_OFI_WARN("Unable to initialize MR cache mutex.");
		goto out;
	}
out:
	return ret;
}

void nccl_ofi_mr_cache_finalize(nccl_ofi_mr_cache_t *cache)
{
	int ret;

	free(cache->slots);
	ret = nccl_net_ofi_mutex_destroy(&cache->lock);
	if (OFI_UNLIKELY(ret)) {
		NCCL_OFI_WARN("Could not destroy MR cache mutex");
	}
	free(cache);
}

int nccl_ofi_mr_cache_grow(nccl_ofi_mr_cache_t *cache)
{
	void *ptr;
	int ret = 0;
	cache->size *= 2;
	NCCL_OFI_TRACE(NCCL_NET, "Growing cache to size %d", cache->size);
	ptr =  realloc(cache->slots, cache->size * sizeof(*cache->slots));
	if (!ptr) {
		NCCL_OFI_WARN("Unable to grow cache");
		ret = errno;
		goto out;
	}
	cache->slots = ptr;

out:
	return ret;
}


int nccl_ofi_mr_cache_lookup(nccl_ofi_mr_cache_t *cache, void *handle)
{
	for (int i = 0; i < cache->used; i++) {
		if (handle == cache->slots[i])
			return i;
	}
	return -1;
}

int nccl_ofi_mr_cache_add_entry(nccl_ofi_mr_cache_t *cache,
						  void *data,
						  int size,
						  int type,
						  void **handle)
{
	nccl_ofi_reg_entry_t *entry;
	static __thread uintptr_t page_size;
	uintptr_t addr;
	size_t pages;
	int ret = 0;

	page_size = (uintptr_t) system_page_size;
	addr = (uintptr_t)data & -page_size; /* start of page of data */
	pages = ((uintptr_t)data + size - addr + page_size-1)/page_size; /* Number of pages in buffer */

	nccl_net_ofi_mutex_lock(&cache->lock);
	for (int slot = 0;;slot++) {
		if (slot == cache->used || addr < cache->slots[slot]->addr) {
			/* cache missed */

			/* grow the cache if needed */
			if (cache->used == cache->size) {
				nccl_ofi_mr_cache_grow(cache);
			}

			assert(cache->slots);
			memmove(cache->slots+slot+1, cache->slots+slot, (cache->size - slot) * sizeof(nccl_ofi_reg_entry_t*));
			cache->slots[slot] = calloc(1, sizeof(nccl_ofi_reg_entry_t));

			entry = cache->slots[slot];
			ret = nccl_ofi_mr_reg_comm(cache, data, size, type, &entry->handle);
			if (ret < 0) {
				goto out;
			}
			entry->addr = addr;
			entry->pages = pages;
			entry->refcnt = 1;
			cache->used++;
			*handle = entry->handle;
		} else if ((addr >=  cache->slots[slot]->addr) &&
                           ((addr - cache->slots[slot]->addr)/system_page_size + pages) <=  cache->slots[slot]->pages) {
			/* cache hit */
			NCCL_OFI_TRACE(NCCL_NET, "Found MR handle for %p in cache slot %d", data, slot);
			cache->slots[slot]->refcnt++;
			*handle = cache->slots[slot]->handle;
			goto out;
		}

	}

out:
	nccl_net_ofi_mutex_unlock(&cache->lock);
	return ret;
}

int nccl_ofi_mr_cache_del_entry(nccl_ofi_mr_cache_t *cache, void *handle)
{
	int slot = -1;
	int ret = 0;

	nccl_net_ofi_mutex_lock(&cache->lock);
	slot = nccl_ofi_mr_cache_lookup(cache, handle);
	if (slot < 0) {
		NCCL_OFI_WARN("Did not find entry to delete");
		ret = -ENOENT;
		goto out;
	}

	/* Keep entry alive for other users */
	if (--cache->slots[slot]->refcnt) {
		goto out;
	}

	/* No more users, dereg with network */
	nccl_ofi_mr_dereg_comm(cache, cache->slots[slot]->handle);

	/* Free this entry and defrag cache */
	free(cache->slots[slot]);
	memmove(cache->slots+slot, cache->slots+slot+1, (cache->size-slot-1)*sizeof(nccl_ofi_mr_cache_t*));

	/* Final registration using the cache, free the cache */
	if (--cache->used == 0) {
		nccl_ofi_mr_cache_finalize(cache);
	}

out:
	nccl_net_ofi_mutex_unlock(&cache->lock);
	return ret;
}
