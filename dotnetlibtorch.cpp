#include <iostream>
#include <ctype>

#include <torch/torch.h>
#include <torch/script.h>

#include <ATen/dlpack.h>
#include <ATen/DLConvertor.h>

struct LibTorchInferenceSession
{
	torch::jit::script::Module model;
	c10::Device device;
};

extern "C" LibTorchInferenceSession* load_model(const char* jit_scripted_serialized_model_path, c10::DeviceType device_type = at::kCPU, int16_t device_id = 0)
{
	torch::jit::script::Module model = torch::jit::load(jit_scripted_serialized_model_path);
	c10::Device device(device_type, device_id);
	model = model.to(device);
	return new LibTorchInferenceSession { model, device };
}

extern "C" LibTorchInferenceSession* destroy_model(LibTorchInferenceSession* inference_session)
{
	delete inference_session;
}

extern "C"  DLManagedTensor run_model(LibTorchInferenceSession* inference_session, DLManagedTensor dl_managed_tensor_in)
{
	torch::Tensor tensor = at::fromDLPack(&dl_managed_tensor_in);
	tensor = tensor.to(inference_session->device);
	autor res = inference_session->model(tensor);
	res = res.to(at::kCPU);
	return *at::toDLPack(res);
}

extern "C" DLManagedTensor process_dlpack_with_libtorch(DLManagedTensor dl_managed_tensor_in)
{
	torch::Tensor tensor = at::fromDLPack(&dl_managed_tensor_in);
	
	auto res = tensor + 1;

	return *at::toDLPack(res);
}
