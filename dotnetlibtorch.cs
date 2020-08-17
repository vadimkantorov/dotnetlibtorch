using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Diagnostics;
using DLPack;

namespace DLPackTest
{
	class Test
	{
		[DllImport("dotnetlibtorch")]
		public static extern DLManagedTensor process_dlpack_with_libtorch(DLManagedTensor dl_managed_tensor);
		
		public static void PrintMatrix<T>(in DLTensor dl_tensor) where T : unmanaged
		{
			Debug.Assert(dl_tensor.CheckType<T, T>());
			var shape = dl_tensor.ShapeSpan();
			for(var r = 0; r < shape[0]; r++)
				for(var c = 0; c < shape[1]; c++)
					Console.WriteLine("({0}, {1}) = {2}", r, c, dl_tensor.Read<T>(r, c));
		}
		
		public static unsafe void Main(string[] args)
		{
			var data = new Int32[2, 3] {
				{0, 1, 2},
				{3, 4, 5}
			};
			var shape = DLTensor.ShapeFromArray(data);
			var strides = DLTensor.RowMajorContiguousTensorStrides(shape);

			fixed(int* ptr_data = data)
			fixed(Int64* ptr_shape = shape)
			fixed(Int64* ptr_strides = strides)
			{
				// Passing strides is optional. If no strides are passed, libtorch assumes row-major contiguous strides
				Console.WriteLine("Before passing to libtorch");
				DLManagedTensor input = DLManagedTensor.FromBlob(ptr_data, data.Rank, ptr_shape, ptr_strides);
				Console.WriteLine(input.dl_tensor);
				PrintMatrix<Int32>(in input.dl_tensor);
				
				Console.WriteLine("After passing to libtorch");
				DLManagedTensor output = process_dlpack_with_libtorch(input);
				Console.WriteLine(output.dl_tensor);
				PrintMatrix<Int32>(in output.dl_tensor);
				
				// Calling the DLPack deleter. If your library function returns a Tensor managed by PyTorch, this would free the memory.
				output.CallDeleter();
			}
		}
	}
}
