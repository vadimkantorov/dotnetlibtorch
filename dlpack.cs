// Source: https://github.com/vadimkantorov/dotnetdlpack/blob/master/dlpack.cs

using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices; 

namespace DLPack
{
	public enum DLDeviceType : Int32
	{
		kDLCPU = 1,
		kDLGPU = 2,
		kDLCPUPinned = 3,
		kDLOpenCL = 4,
		kDLVulkan = 7,
		kDLMetal = 8,
		kDLVPI = 9,
		kDLROCM = 10,
		kDLExtDev = 12,
	}

	public enum DLDataTypeCode : Byte
	{
		kDLInt = 0,
		kDLUInt = 1,
		kDLFloat = 2,
		kDLBfloat = 4,
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct DLDataType
	{
		public DLDataTypeCode type_code;
		public Byte bits;
		public UInt16 lanes;

		public static DLDataType From<T>()
		{
			var dtype = new DLDataType();
			dtype.lanes = 1;
			dtype.bits = (Byte) (Unsafe.SizeOf<T>() * 8);
				
			var t = typeof(T);

			if(t == typeof(SByte) || t == typeof(Int16) || t == typeof(Int32) || t == typeof(Int64))
				dtype.type_code = DLDataTypeCode.kDLInt;
			else if(t == typeof(Byte) || t == typeof(UInt16) || t == typeof(UInt32) || t == typeof(UInt64))
				dtype.type_code = DLDataTypeCode.kDLUInt;
			else if(t == typeof(Single) || t == typeof(Double))
				dtype.type_code = DLDataTypeCode.kDLFloat;
			else
				throw new Exception($"Type [{typeof(T)}] is not supported");
			return dtype;
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct DLContext
	{
		public DLDeviceType device_type;
		public Int32 device_id;
	}

	[StructLayout(LayoutKind.Sequential)]
	public struct DLTensor
	{
		public IntPtr data;
		public DLContext ctx;
		public Int32 ndim;
		public DLDataType dtype;
		public IntPtr shape;
		public IntPtr strides;
		public UInt64 byte_offset;
	
		public bool CheckType<T, TT>() where T: unmanaged where TT: unmanaged
		{
			var T_dtype = DLDataType.From<T>();
			var TT_dtype = DLDataType.From<TT>();
			return dtype.type_code == T_dtype.type_code && dtype.bits == TT_dtype.bits && dtype.lanes == TT_dtype.lanes;
		}
		
		public unsafe ReadOnlySpan<Int64> ShapeSpan()
		{
			return shape != IntPtr.Zero ? new ReadOnlySpan<Int64>(shape.ToPointer(), ndim) : ReadOnlySpan<Int64>.Empty; 
		}

		public unsafe Int64 Numel()
		{
			Int64 numel = 1;
			var shape = ShapeSpan();
			for(Int32 i = 0; i < ndim; i++)
				numel *= shape[i];
			return numel;
		}
		
		public unsafe ReadOnlySpan<Int64> StridesSpan(bool assumeRowMajorContiguousStrides = false)
		{
			return strides != IntPtr.Zero ? new ReadOnlySpan<Int64>(strides.ToPointer(), ndim) : assumeRowMajorContiguousStrides ? RowMajorContiguousTensorStrides(ShapeSpan()) : ReadOnlySpan<Int64>.Empty; 
		}

		public unsafe ReadOnlySpan<T> DataSpanLessThan2Gb<T>() where T : unmanaged
		{
			var bits = Numel() * dtype.bits * dtype.lanes;
			Int32 length = (Int32)(bits / (Unsafe.SizeOf<T>() * 8));
			return new ReadOnlySpan<T>(data.ToPointer(), length);
		}
		
		public unsafe T Read<T>(params Int64[] coords) where T : unmanaged
		{
			var strides = StridesSpan(assumeRowMajorContiguousStrides : true);
			Int64 offset = 0;
			for(Int32 i = 0; i < ndim; i++)
				offset += strides[i] * coords[i];
			T* ptr = (T*)data.ToPointer();
			return ptr[offset];
		}

		public override string ToString()
		{
			var s_h_a_p_e = string.Join(",", ShapeSpan().ToArray());
			var s_t_r_i_d_e_s = string.Join(",", StridesSpan().ToArray());
			return $"type_code={dtype.type_code}, bits={dtype.bits}, lanes={dtype.lanes}, ndim={ndim}, shape=[{s_h_a_p_e}], strides=[{s_t_r_i_d_e_s}]"; 
		}

		public static Int64[] RowMajorContiguousTensorStrides(ReadOnlySpan<Int64> shape)
		{
			var strides = new Int64[shape.Length];
			for(Int32 i = 0; i < strides.Length; i++)
				strides[i] = i == shape.Length - 1 ? 1 : shape[i + 1];
			return strides;
		}

		public static Int64[] ShapeFromArray(Array array)
		{
			Int64[] shape = new Int64[array.Rank];
			for(Int32 i = 0; i < array.Rank; i++)
				shape[i] = array.GetLongLength(i);
			return shape;
		}
	}
	
	public delegate void DLDeleterFunc(ref DLManagedTensor self);

	[StructLayout(LayoutKind.Sequential)]
	public struct DLManagedTensor
	{
		public DLTensor dl_tensor;
		public IntPtr manager_ctx;
		public DLDeleterFunc deleter;

		public void CallDeleter()
		{
			if(deleter != null)
				deleter(ref this);
		}

		public static void EmptyDeleter(ref DLManagedTensor self)
		{
		}

		public unsafe static DLManagedTensor FromBlob<T>(T* data, Int32 ndim, Int64* shape, Int64* strides = null) where T : unmanaged
		{
			var dl_managed_tensor = new DLManagedTensor();
			dl_managed_tensor.dl_tensor.data = (IntPtr)data;
			dl_managed_tensor.dl_tensor.ctx.device_type = DLDeviceType.kDLCPU;
			dl_managed_tensor.dl_tensor.ndim = ndim;
			dl_managed_tensor.dl_tensor.dtype = DLDataType.From<T>();
			dl_managed_tensor.dl_tensor.shape = (IntPtr)shape;
			dl_managed_tensor.dl_tensor.strides = (IntPtr)strides;
			dl_managed_tensor.dl_tensor.byte_offset = 0;
			dl_managed_tensor.deleter = EmptyDeleter;
			return dl_managed_tensor;
		}
	}
}
