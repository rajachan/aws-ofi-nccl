#include <stdlib.h>
#include <math.h>
#include "nccl-headers/tuner.h"
#include "nccl_ofi_tuner.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"

float nccl_ofi_tuner_compute_base_cost(ncclFunc_t func, int algo, int proto)
{
	/*
	 * Just passing up the NCCL base latencies for now. These costs could be
	 * computed too, but that can come as a follow up.
	 */
	return nccl_base_lat[algo][proto];
}

float nccl_ofi_tuner_compute_cost(ncclFunc_t func, int algo, int proto, int pipe_ops, size_t size)
{
	struct nccl_ofi_tuner_model_params *params = &ctx->model_params;
	float cost = -1;
	float latency = 0;
	float bw = 0;
	float p2p_lat = 0;
	float net_lat = 0;
	int num_steps = 0;
	int num_internode_steps = 0;

	/*
	 * Intranode P2P transfers go over nvlink for NVLS
	 * algorithms and over PCI for standard trees
	 */
	p2p_lat = (algo == NCCL_ALGO_NVLS_TREE || algo == NCCL_ALGO_NVLS)
		   ? nccl_hw_lat[NCCL_HW_NVLINK][algo][proto]
		   : nccl_hw_lat[NCCL_HW_PCI][algo][proto];

	/*
	 * TODO: There is more involved than the NET_COMP_OVERHEAD itself for
	 * the simple protocol, including overheads from libfabric and NCCL's
	 * proxy thread itself in processing a completion handed to the host by
	 * the device. Costs associated with out-of-order completions that could
	 * stall the pipeline should be captured here as well.
	 */
	net_lat = (proto == NCCL_PROTO_SIMPLE)
		    ? params->net_lat + NET_COMP_OVERHEAD
		    : params->net_lat;

	switch(func) {
	case ncclFuncAllReduce:
		switch(algo) {
		case NCCL_ALGO_RING:
			num_steps = 2 * (ctx->num_ranks - 1);
			num_internode_steps = 2 * ctx->num_nodes;
			latency = (num_internode_steps * net_lat)
				  + (num_steps - num_internode_steps) * p2p_lat;
			bw = params->internode_bw * params->rails * NCCL_OFI_TUNER_NUM_CHANNELS;
			break;
		case NCCL_ALGO_NVLS_TREE:
			latency = p2p_lat + (2 * log2(ctx->num_nodes) * net_lat);
			bw = NCCL_OFI_MIN(params->intranode_bw, (params->internode_bw * params->rails) / 2)
			     * NCCL_OFI_TUNER_NUM_CHANNELS;
			break;
		case NCCL_ALGO_TREE:
			/* No correction factor like with NCCL (which it applies for 68B-256MiB messages */
			latency = ((2 * ((ctx->num_ranks / ctx->num_nodes) - 1) * p2p_lat)
				   + (2 * log2(ctx->num_nodes) * net_lat));
			bw = (params->internode_bw * params->rails * NCCL_OFI_TUNER_NUM_CHANNELS) / 2;
			break;
		default:
			NCCL_OFI_WARN("Algorithm %d for collective %d  without a model.", algo, func);
		}
		break;
	default:
		NCCL_OFI_WARN("Unsupported collective %d, fallback to NCCL's selection.", func);
	}

	/* Penalize the low-latency protocol bandwidths for their overhead */
	if (proto == NCCL_PROTO_LL)
		/* 8B total with 4B data and 4B flags, so take a 50% hit */
		bw *= 0.5;
	else if (proto == NCCL_PROTO_LL128)
		/* 120B data and 8B flags */
		bw *= 0.9375;

	/* Simplest hockney based: t = (⍺ + βm) */
	cost = latency + size / bw;

	return cost;
}


/*
 * Compute the base costs for each of the algorithms at plugin initialization
 * time using only the comm size. Depending on the analytical model used, we
 * might have to update the cost at operation time based on the message size.
 */
void nccl_ofi_tuner_model_costs()
{
	ncclFunc_t func;
	int algo, proto = 0;
	for (func = 0; func < NCCL_NUM_FUNCTIONS; func++) {
		for (algo = 0; algo < NCCL_NUM_ALGORITHMS; algo++) {
			for(proto = 0; proto < NCCL_NUM_PROTOCOLS; proto++) {
				ctx->base_costs[func][algo][proto] = 
					nccl_ofi_tuner_compute_base_cost(func, algo, proto);
			}
		}
	}
}
