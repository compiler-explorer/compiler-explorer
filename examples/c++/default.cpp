#include "dpu/mfc.h"
#include "mfcp.h"
__device__ void P00_K1_AggregateMfcp(bool dummy_param)
{
	uint16_t metacolumn_0c990c5b0 = collect<100, uint16_t>();
	int64_t a3932e600 = collect<101, int64_t>();
	uint64_t control = collect<102, uint64_t>();
	uint64_t start_eid_raw = collect<103, uint64_t>();
	// aggregation_0, QxSumAggregateExpression
	mfc::atomic::Add<int64_t, mfc::PNZ>(reinterpret_cast<int64_t *>(extractEidFromInputControl(control) + 0), ((((((metacolumn_0c990c5b0 & 1) == 1) != 0) & (((metacolumn_0c990c5b0 & 2) == 2) != 0))) ? (a3932e600) : (0)), ((((true != 0) & (((((metacolumn_0c990c5b0 & 1) == 1) != 0) & (((metacolumn_0c990c5b0 & 2) == 2) != 0)) != 0))) ? (isNotNull(control, 1)) : (false)));
	// aggregation_1, QxSumAggregateExpression
	mfc::atomic::Add<int64_t, mfc::PNZ>(reinterpret_cast<int64_t *>(extractEidFromInputControl(control) + 8), ((((((metacolumn_0c990c5b0 & 4) == 4) != 0) & (((metacolumn_0c990c5b0 & 2) == 2) != 0))) ? (a3932e600) : (0)), ((((true != 0) & (((((metacolumn_0c990c5b0 & 4) == 4) != 0) & (((metacolumn_0c990c5b0 & 2) == 2) != 0)) != 0))) ? (isNotNull(control, 1)) : (false)));
	// aggregation_2, QxCountAggregateExpression
	mfc::atomic::Add<int64_t, mfc::PNZ>(reinterpret_cast<int64_t *>(extractEidFromInputControl(control) + 16), 1, ((((((((metacolumn_0c990c5b0 & 16) == 16) != 0) & (((metacolumn_0c990c5b0 & 8) == 8) != 0)) != 0) & (true != 0))) ? (isNotNull(control, 2)) : (false)));
	// aggregation_3, QxUpdateAggregatorsINL
	mfc::atomic::Or<int32_t, mfc::PANY>(reinterpret_cast<int32_t *>(extractEidFromInputControl(start_eid_raw) - -4), (((true << 2) + (((((((((metacolumn_0c990c5b0 & 2) == 2) != 0) & (((metacolumn_0c990c5b0 & 4) == 4) != 0))) ? (isNotNull(control, 1)) : (false)) << 1) + ((((((metacolumn_0c990c5b0 & 2) == 2) != 0) & (((metacolumn_0c990c5b0 & 1) == 1) != 0))) ? (isNotNull(control, 1)) : (false)))))), true);
	uint64_t controlOutput = calculateEmptyControlOutput();
	deliver<104>(controlOutput);
}