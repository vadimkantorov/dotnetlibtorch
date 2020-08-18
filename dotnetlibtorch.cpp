#include <torch/torch.h>
#include <torch/script.h>

#include <ATen/dlpack.h>
#include <ATen/DLConvertor.h>

struct LibTorchInferenceSession
{
	torch::jit::script::Module model;
	c10::Device device;
};

extern "C" LibTorchInferenceSession* load_model(const char* jit_scripted_serialized_model_path, c10::DeviceType device_type = c10::DeviceType::CPU, c10::DeviceIndex device_id = -1)
{
	c10::Device device(device_type, device_id);
	torch::jit::script::Module model = torch::jit::load(jit_scripted_serialized_model_path, device);
	model.eval();
	return new LibTorchInferenceSession { model, device };
}

extern "C" void destroy_model(LibTorchInferenceSession* inference_session)
{
	delete inference_session;
}

extern "C"  DLManagedTensor run_model(LibTorchInferenceSession* inference_session, DLManagedTensor dl_managed_tensor_in)
{
	torch::Tensor tensor = at::fromDLPack(&dl_managed_tensor_in);
	
	torch::NoGradGuard no_grad;
	auto res = inference_session->model.forward({tensor.to(inference_session->device)}).toTensor().to(c10::DeviceType::CPU);
	
	return *at::toDLPack(res);
}

extern "C" DLManagedTensor process_dlpack_with_libtorch(DLManagedTensor dl_managed_tensor_in)
{
	torch::Tensor tensor = at::fromDLPack(&dl_managed_tensor_in);
	
	torch::NoGradGuard no_grad;
	auto res = tensor + 1;

	return *at::toDLPack(res);
}
