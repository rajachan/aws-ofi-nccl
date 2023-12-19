#ifndef NCCL_OFI_TUNER_H_
#define NCCL_OFI_TUNER_H_

#include <linux/limits.h>
#include <float.h>
#include "nccl-headers/tuner.h"

/*
 * TODO: This should not be statically defined. The plugin interface lets us
 * tune the number of channels as well, but that can come later (once a
 * proto+algo combination is chosen, we can compute the cost with different
 * channel count and optimize for it.
 */
#define NCCL_OFI_TUNER_NUM_CHANNELS	(8)

/* Latency in µsecs and bandwidths in Bytes/µsec */
#define NET_LATENCY		(20)
#define INTRANODE_BW		(12.5 * 1024 * 1024 * 1024 * 1e-6) /* per rail */
#define INTERNODE_BW		(31.5 * 1024 * 1024 * 1024 * 1e-6) /* PCI gen4 x16 baseline */
#define NET_NUM_RAILS		(4)    /* Available to each GPU */

/*
 * With EFA, we expect a ~2µsec cost in the device and ~1µsec cost to write that
 * completion up to the host stack.
 */
#define NET_COMP_OVERHEAD	(3)

/*
 * NCCL's algo-specific latencies for intra-node cases: with and without NVLink.
 * The struct is directly taken from NCCL v2.19.4, and the network coefficients
 * are dropped (in favor of the ones we use from nccl_ofi_tuner_model_params.
 * TODO: This is all messy, need to manage the coefficients and net params better.
 */
#define NCCL_HW_NVLINK		(0)
#define NCCL_HW_PCI		(1)

/* From hwLat[] in NCCL. Values in µsecs. */
static const float nccl_hw_lat[2][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
	{ /* NVLink */
		{ .6, 1.25,  28 }, /* Tree (LL, LL128, Simple) */
		{ .6,  1.9, 3.4 }, /* Ring (LL, LL128, Simple) */
		{  0,    0, 3.7 }, /* Collnet Direct - Unused */
		{  0,    0, 2.8 }, /* Collnet Chain - Unused */
		{  0,    0,  23 }, /* NVLS (Simple only) */
		{  0,    0,  23 }  /* NVLS Tree (Simple only)*/
	},
	{ /* PCIE */
		{ 1.0, 1.9,  28 }, /* Tree (LL, LL128, Simple) */
		{ 1.0, 2.5, 5.7 }, /* Ring (LL, LL128, Simple) */
		{   0,   0, 3.7 }, /* Collnet Direct - Unused */
		{   0,   0, 2.8 }, /* Collnet Chain - Unused */
		{   0,   0,   0 }, /* NVLS (Simple only) */
		{   0,   0,   0 }  /* NVLS Tree (Simple only) */
	}
};

/* From baseLat[] in NCCL. Values in µsecs. */
static const float nccl_base_lat[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
	{  6.8, 14.0,    0 }, /* Tree */
	{  6.6, 14.0,  8.4 }, /* Ring */
	{    0,    0,    0 }, /* Collnet Direct */
	{    0,    0,    0 }, /* Collnet Chain */
	{    0,    0,    0 }, /* NVLS */
	{    0,    0,    0 }  /* NVLS Tree */
};

struct nccl_ofi_tuner_model_params {
	float net_lat;
	float internode_bw;
	float intranode_bw;
	int rails;
};

struct nccl_ofi_tuner_context {
	/* communicator size */
	int num_ranks;
	int num_nodes;

	struct nccl_ofi_tuner_model_params model_params;

	float base_costs[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
};

/* Global context, allocated at _init() */
struct nccl_ofi_tuner_context *ctx;

/* Modeling functions */
void nccl_ofi_tuner_model_costs();
float nccl_ofi_tuner_compute_cost(ncclFunc_t func, int algo, int proto, int pipe_ops, size_t size);

#endif /* NCCL_OFI_TUNER_H_ */
