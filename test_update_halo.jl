# RUN AS FOLLOWS: julia test_update_halo.jl 

# NOTE: All tests of this file can be run with any number of processes.
# Nearly all of the functionality can however be verified with one single process
# (thanks to the usage of periodic boundaries in most of the full halo update tests).

push!(LOAD_PATH, "../src")
using Test
import MPI, LoopVectorization
using CUDA, AMDGPU
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import ImplicitGlobalGrid: @require, longnameof

test_cuda = CUDA.functional()
test_amdgpu = AMDGPU.functional()

array_types          = ["CPU"]
gpu_array_types      = []
device_types         = ["auto"]
gpu_device_types     = []
allocators           = Function[zeros]
gpu_allocators       = []
ArrayConstructors    = [Array]
GPUArrayConstructors = []
CPUArray             = Array
if test_cuda
    cuzeros = CUDA.zeros
    push!(array_types, "CUDA")
    push!(gpu_array_types, "CUDA")
    push!(device_types, "CUDA")
    push!(gpu_device_types, "CUDA")
    push!(allocators, cuzeros)
    push!(gpu_allocators, cuzeros)
    push!(ArrayConstructors, CuArray)
    push!(GPUArrayConstructors, CuArray)
end
if test_amdgpu
    roczeros = AMDGPU.zeros
    push!(array_types, "AMDGPU")
    push!(gpu_array_types, "AMDGPU")
    push!(device_types, "AMDGPU")
    push!(gpu_device_types, "AMDGPU")
    push!(allocators, roczeros)
    push!(gpu_allocators, roczeros)
    push!(ArrayConstructors, ROCArray)
    push!(GPUArrayConstructors, ROCArray)
end

## Test setup
MPI.Init();
nprocs = MPI.Comm_size(MPI.COMM_WORLD); # NOTE: these tests can run with any number of processes.
ndims_mpi = GG.NDIMS_MPI;
nneighbors_per_dim = GG.NNEIGHBORS_PER_DIM; # Should be 2 (one left and one right neighbor).
nx = 7;
ny = 5;
nz = 6;
dx = 1.0
dy = 1.0
dz = 1.0

@testset "$(basename(@__FILE__)) (processes: $nprocs)" begin

    # NOTE: I have removed here many tests in order not to make this example too long.
    
    @testset "3. data transfer components" begin
        @testset "iwrite_sendbufs! / iread_recvbufs!" begin
            @testset "sendranges / recvranges ($array_type arrays)" for (array_type, device_type, zeros) in zip(array_types, device_types, allocators)
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(2,2,3), quiet=true, init_MPI=false, device_type=device_type);
                P   = zeros(nx,  ny,  nz  );
                A   = zeros(nx-1,ny+2,nz+1);
                P, A = GG.wrap_field.((P, A));
                @test GG.sendranges(1, 1, P) == [                    2:2,             1:size(P,2),             1:size(P,3)]
                @test GG.sendranges(2, 1, P) == [size(P,1)-1:size(P,1)-1,             1:size(P,2),             1:size(P,3)]
                @test GG.sendranges(1, 2, P) == [            1:size(P,1),                     2:2,             1:size(P,3)]
                @test GG.sendranges(2, 2, P) == [            1:size(P,1), size(P,2)-1:size(P,2)-1,             1:size(P,3)]
                @test GG.sendranges(1, 3, P) == [            1:size(P,1),             1:size(P,2),                     3:3]
                @test GG.sendranges(2, 3, P) == [            1:size(P,1),             1:size(P,2), size(P,3)-2:size(P,3)-2]
                @test GG.recvranges(1, 1, P) == [                    1:1,             1:size(P,2),             1:size(P,3)]
                @test GG.recvranges(2, 1, P) == [    size(P,1):size(P,1),             1:size(P,2),             1:size(P,3)]
                @test GG.recvranges(1, 2, P) == [            1:size(P,1),                     1:1,             1:size(P,3)]
                @test GG.recvranges(2, 2, P) == [            1:size(P,1),     size(P,2):size(P,2),             1:size(P,3)]
                @test GG.recvranges(1, 3, P) == [            1:size(P,1),             1:size(P,2),                     1:1]
                @test GG.recvranges(2, 3, P) == [            1:size(P,1),             1:size(P,2),     size(P,3):size(P,3)]
                @test_throws ErrorException  GG.sendranges(1, 1, A)
                @test_throws ErrorException  GG.sendranges(2, 1, A)
                @test GG.sendranges(1, 2, A) == [            1:size(A,1),                     4:4,             1:size(A,3)]
                @test GG.sendranges(2, 2, A) == [            1:size(A,1), size(A,2)-3:size(A,2)-3,             1:size(A,3)]
                @test GG.sendranges(1, 3, A) == [            1:size(A,1),             1:size(A,2),                     4:4]
                @test GG.sendranges(2, 3, A) == [            1:size(A,1),             1:size(A,2), size(A,3)-3:size(A,3)-3]
                @test_throws ErrorException  GG.recvranges(1, 1, A)
                @test_throws ErrorException  GG.recvranges(2, 1, A)
                @test GG.recvranges(1, 2, A) == [            1:size(A,1),                     1:1,             1:size(A,3)]
                @test GG.recvranges(2, 2, A) == [            1:size(A,1),     size(A,2):size(A,2),             1:size(A,3)]
                @test GG.recvranges(1, 3, A) == [            1:size(A,1),             1:size(A,2),                     1:1]
                @test GG.recvranges(2, 3, A) == [            1:size(A,1),             1:size(A,2),     size(A,3):size(A,3)]
                finalize_global_grid(finalize_MPI=false);
            end;

            # NOTE: I have removed here many tests in order not to make this example too long.


            @testset "write_h2h! / read_h2h!" begin
                init_global_grid(nx, ny, nz; quiet=true, init_MPI=false);
                P  = zeros(nx,  ny,  nz  );
                P .= [iz*1e2 + iy*1e1 + ix for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
                P2 = zeros(size(P));
                halowidths = (1,1,1)
                # (dim=1)
                buf = zeros(halowidths[1], size(P,2), size(P,3));
                ranges = [2:2, 1:size(P,2), 1:size(P,3)];
                GG.write_h2h!(buf, P, ranges, 1);
                @test all(buf[:] .== P[ranges[1],ranges[2],ranges[3]][:])
                GG.read_h2h!(buf, P2, ranges, 1);
                @test all(buf[:] .== P2[ranges[1],ranges[2],ranges[3]][:])
                # (dim=2)
                buf = zeros(size(P,1), halowidths[2], size(P,3));
                ranges = [1:size(P,1), 3:3, 1:size(P,3)];
                GG.write_h2h!(buf, P, ranges, 2);
                @test all(buf[:] .== P[ranges[1],ranges[2],ranges[3]][:])
                GG.read_h2h!(buf, P2, ranges, 2);
                @test all(buf[:] .== P2[ranges[1],ranges[2],ranges[3]][:])
                # (dim=3)
                buf = zeros(size(P,1), size(P,2), halowidths[3]);
                ranges = [1:size(P,1), 1:size(P,2), 4:4];
                GG.write_h2h!(buf, P, ranges, 3);
                @test all(buf[:] .== P[ranges[1],ranges[2],ranges[3]][:])
                GG.read_h2h!(buf, P2, ranges, 3);
                @test all(buf[:] .== P2[ranges[1],ranges[2],ranges[3]][:])
                finalize_global_grid(finalize_MPI=false);
            end;

            # NOTE: I have removed here many tests in order not to make this example too long.

            @static if test_cuda || test_amdgpu
                @testset "write_d2x! / write_d2h_async! / read_x2d! / read_h2d_async! ($array_type arrays)" for (array_type, device_type, gpuzeros, GPUArray) in zip(gpu_array_types, gpu_device_types, gpu_allocators, GPUArrayConstructors)
                    init_global_grid(nx, ny, nz; quiet=true, init_MPI=false, device_type=device_type);
                    P  = zeros(nx,  ny,  nz  );
                    P .= [iz*1e2 + iy*1e1 + ix for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
                    P  = GPUArray(P);
                    halowidths = (1,3,1)
                    if array_type == "CUDA"
                        # (dim=1)
                        dim = 1;
                        P2  = gpuzeros(eltype(P),size(P));
                        buf = zeros(halowidths[dim], size(P,2), size(P,3));
                        buf_d, buf_h = GG.register(CuArray,buf);
                        ranges = [2:2, 1:size(P,2), 1:size(P,3)];
                        nthreads = (1, 1, 1);
                        halosize = [r[end] - r[1] + 1 for r in ranges];
                        nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                        @cuda blocks=nblocks threads=nthreads GG.write_d2x!(buf_d, P, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        @cuda blocks=nblocks threads=nthreads GG.read_x2d!(buf_d, P2, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        buf .= 0.0;
                        P2  .= 0.0;
                        custream = stream();
                        GG.write_d2h_async!(buf, P, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        GG.read_h2d_async!(buf, P2, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        CUDA.Mem.unregister(buf_h);
                        # (dim=2)
                        dim = 2;
                        P2  = gpuzeros(eltype(P),size(P));
                        buf = zeros(size(P,1), halowidths[dim], size(P,3));
                        buf_d, buf_h = GG.register(CuArray,buf);
                        ranges = [1:size(P,1), 2:4, 1:size(P,3)];
                        nthreads = (1, 1, 1);
                        halosize = [r[end] - r[1] + 1 for r in ranges];
                        nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                        @cuda blocks=nblocks threads=nthreads GG.write_d2x!(buf_d, P, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        @cuda blocks=nblocks threads=nthreads GG.read_x2d!(buf_d, P2, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        buf .= 0.0;
                        P2  .= 0.0;
                        custream = stream();
                        GG.write_d2h_async!(buf, P, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        GG.read_h2d_async!(buf, P2, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        CUDA.Mem.unregister(buf_h);
                        # (dim=3)
                        dim = 3
                        P2  = gpuzeros(eltype(P),size(P));
                        buf = zeros(size(P,1), size(P,2), halowidths[dim]);
                        buf_d, buf_h = GG.register(CuArray,buf);
                        ranges = [1:size(P,1), 1:size(P,2), 4:4];
                        nthreads = (1, 1, 1);
                        halosize = [r[end] - r[1] + 1 for r in ranges];
                        nblocks  = Tuple(ceil.(Int, halosize./nthreads));
                        @cuda blocks=nblocks threads=nthreads GG.write_d2x!(buf_d, P, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        @cuda blocks=nblocks threads=nthreads GG.read_x2d!(buf_d, P2, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        buf .= 0.0;
                        P2  .= 0.0;
                        custream = stream();
                        GG.write_d2h_async!(buf, P, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P[ranges[1],ranges[2],ranges[3]][:]))
                        GG.read_h2d_async!(buf, P2, ranges, custream); CUDA.synchronize();
                        @test all(buf[:] .== Array(P2[ranges[1],ranges[2],ranges[3]][:]))
                        CUDA.Mem.unregister(buf_h);
                    elseif array_type == "AMDGPU"
                        # NOTE: removed for this example
                    end
                    finalize_global_grid(finalize_MPI=false);
                end;
            end

            # NOTE: I have removed here many tests in order not to make this example too long.

            @testset "iwrite_sendbufs! ($array_type arrays)" for (array_type, device_type, zeros, Array) in zip(array_types, device_types, allocators, ArrayConstructors)
                init_global_grid(nx, ny, nz; periodx=1, periody=1, periodz=1, overlaps=(4,2,3), halowidths=(2,1,1), quiet=true, init_MPI=false, device_type=device_type);
                P = zeros(nx,  ny,  nz  );
                A = zeros(nx-1,ny+2,nz+1);
                P .= Array([iz*1e2 + iy*1e1 + ix for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)]);
                A .= Array([iz*1e2 + iy*1e1 + ix for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]);
                P, A = GG.wrap_field.((P, A));
                GG.allocate_bufs(P, A);
                if     (array_type == "CUDA")   GG.allocate_custreams(P, A);
                elseif (array_type == "AMDGPU") GG.allocate_rocstreams(P, A);
                else                            GG.allocate_tasks(P, A);
                end
                dim = 1
                n = 1
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[3:4,:,:][:]))) # DEBUG: here and later, CPUArray is needed to avoid error in AMDGPU because of mapreduce
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== 0.0))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[3:4,:,:][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== 0.0)
                end
                n = 2
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[end-3:end-2,:,:][:])))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== 0.0))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[end-3:end-2,:,:][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== 0.0)
                end
                dim = 2
                n = 1
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[:,2,:][:])))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== Array(A.A[:,4,:][:])))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[:,2,:][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== CPUArray(A.A[:,4,:][:]))
                end
                n = 2
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[:,end-1,:][:])))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== Array(A.A[:,end-3,:][:])))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[:,end-1,:][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== CPUArray(A.A[:,end-3,:][:]))
                end
                dim = 3
                n = 1
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[:,:,3][:])))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== Array(A.A[:,:,4][:])))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[:,:,3][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== CPUArray(A.A[:,:,4][:]))
                end
                n = 2
                GG.iwrite_sendbufs!(n, dim, P, 1);
                GG.iwrite_sendbufs!(n, dim, A, 2);
                GG.wait_iwrite(n, P, 1);
                GG.wait_iwrite(n, A, 2);
                if (array_type=="CUDA" && GG.cudaaware_MPI(dim)) || (array_type=="AMDGPU" && GG.amdgpuaware_MPI(dim))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,1,P) .== Array(P.A[:,:,end-2][:])))
                    @test all(CPUArray(GG.gpusendbuf_flat(n,dim,2,A) .== Array(A.A[:,:,end-3][:])))
                else
                    @test all(GG.sendbuf_flat(n,dim,1,P) .== CPUArray(P.A[:,:,end-2][:]))
                    @test all(GG.sendbuf_flat(n,dim,2,A) .== CPUArray(A.A[:,:,end-3][:]))
                end
                finalize_global_grid(finalize_MPI=false);
            end;

            # NOTE: I have removed here many tests in order not to make this example too long.

        end;
    end;
end;

## Test tear down
MPI.Finalize()