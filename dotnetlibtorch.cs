using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Diagnostics;
using DLPack;

namespace DotNetLibTorch
{
	static class dotnetlibtorch
	{
		public enum DeviceType : Int16
		{
			CPU = 0,
  			CUDA = 1
		}
		
		[DllImport(nameof(dotnetlibtorch))]
		public static extern DLManagedTensor process_dlpack_with_libtorch(DLManagedTensor dl_managed_tensor);
		
		[DllImport(nameof(dotnetlibtorch), CharSet = CharSet.Ansi)]
		public static extern IntPtr load_model([MarshalAs(UnmanagedType.LPStr)]string jit_scripted_serialized_model_path, DeviceType device_type = DeviceType.CPU, Int16 device_id = -1);

		[DllImport(nameof(dotnetlibtorch))]
		public static extern void destroy_model(IntPtr inference_session);
		
		[DllImport(nameof(dotnetlibtorch))]
		public static extern DLManagedTensor run_model(IntPtr inference_session, DLManagedTensor dl_managed_tensor_in);
	}
	
	class Test
	{
		public static void PrintMatrix<T>(in DLTensor dl_tensor) where T : unmanaged
		{
			Console.WriteLine(dl_tensor);
			Debug.Assert(dl_tensor.ndim == 2 && dl_tensor.CheckType<T, T>());
			var shape = dl_tensor.ShapeSpan();
			for(var r = 0; r < shape[0]; r++)
				for(var c = 0; c < shape[1]; c++)
					Console.WriteLine("({0}, {1}) = {2}", r, c, dl_tensor.Read<T>(r, c));
		}
		
		public static unsafe void Main(string[] args)
		{
			var data = new Single[2, 3] {
				{0.0f, 1.0f, 2.0f},
				{3.0f, 4.0f, 5.0f}
			};
			var device_type = dotnetlibtorch.DeviceType.CPU;
			Int16 device_index = -1;
			
			fixed(Single* ptr_data = data)
			fixed(Int64* ptr_shape = DLTensor.ShapeFromArray(data))
			{
				// Passing strides is optional. If no strides are passed, libtorch assumes row-major contiguous strides
				// You can explicitly construct them with Int64[] strides = DLTensor.RowMajorContiguousStrides(shape: DLTensor.ShapeFromArray(data)) and pass the pinned pointer to FromBlob

				Console.WriteLine("Before passing to libtorch");
				DLManagedTensor input = DLManagedTensor.FromBlob(ptr_data, data.Rank, ptr_shape);
				PrintMatrix<Single>(in input.dl_tensor);
				
				Console.WriteLine("After passing to libtorch");
				DLManagedTensor output;
				if(args.Length > 1)
				{
					var inference_session = dotnetlibtorch.load_model(args[1]);
					output = dotnetlibtorch.run_model(inference_session, input, device_type, device_index);
					dotnetlibtorch.destroy_model(inference_session);
				}
				else
					output = dotnetlibtorch.process_dlpack_with_libtorch(input);
				
				PrintMatrix<Single>(in output.dl_tensor);
				
				// Calling the DLPack deleter. If your library function returns a Tensor managed by PyTorch, this would free the memory.
				output.CallDeleter();
				Console.WriteLine("Called deleter");
			}
		}
	}
}
