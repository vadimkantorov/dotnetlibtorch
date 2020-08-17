#include <iostream>

#include <torch/torch.h>
#include <ATen/DLConvertor.h>

extern "C"
{

	#include "dlpack.h"

	DLManagedTensor process_dlpack_with_libtorch(DLManagedTensor dl_managed_tensor_in)
	{
		torch::Tensor tensor = at::fromDLPack(&dl_managed_tensor_in);

		auto res = tensor + 1;

		return *at::toDLPack(res);
	}

}
