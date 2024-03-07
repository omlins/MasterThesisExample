push!(LOAD_PATH, "../src")
using Test
using ImplicitGlobalGrid; GG = ImplicitGlobalGrid
import MPI
using CUDA
import ImplicitGlobalGrid: longnameof

test_gpu = true #CUDA.functional() # Currently: Just set this false to test CPU-only functions
if test_gpu
	global cuzeros = CUDA.zeros
	global allocators = [cuzeros]
	global ArrayConstructors = [CuArray]
else
	global cuzeros = nothing  # To enable statements like: if zeros==cuzeros
	global CuArray = nothing  # To enable statements like: if Array==CuArray
	global allocators = [zeros]
	global ArrayConstructors = [Array]
end

## 0. Test setup
MPI.Init();
nprocs = MPI.Comm_size(MPI.COMM_WORLD);
me = MPI.Comm_rank(MPI.COMM_WORLD);
if me==0 println("nprocs=$nprocs") end
ndims_mpi = GG.NDIMS_MPI;
nneighbors_per_dim = GG.NNEIGHBORS_PER_DIM; # Should be 2 (one left and one right neighbor).
nx = 512;
ny = 512;
nz = 512;
dx = 1.0;
dy = 1.0;
dz = 1.0;

@testset "$(basename(@__FILE__)) (processes: $nprocs)" begin

	# NOTE: I have removed here many tests in order not to make this example too long.

	@testset "3. data transfer components" begin
		if me==0 println("3. data transfer components") end
		@testset "iwrite_sendbufs! / iread_recvbufs!" begin
			@testset "write_h2h! / read_h2h!" begin
				if me==0 println("write_h2h! / read_h2h!") end
				init_global_grid(nx, ny, nz, quiet=true, init_MPI=false);
				P  = zeros(nx,  ny,  nz  );
				P .= [iz*1e2 + iy*1e1 + ix for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
				P2 = zeros(size(P));
				halowidths = (1,1,1)
				nt = 10*512
				warmup = 10
				ntransfers = 1
				# (dim=1)
				GBs_ref = 5.5
				GBs_var = 4.5 # Mostly it produces ~1.1 GB/s but sometimes 5.5 GB/s (no values inbetween). No systematics were found (the good result can happen in REPL or with optimised or non-optimsed run or when allocating only zeros - all was seen).
				buf = zeros(halowidths[1], size(P,2), size(P,3));
				ranges = [2:2, 1:size(P,2), 1:size(P,3)];
				for it = 1:nt+warmup
					if (it == warmup+1) global t0 = time() end
					GG.write_h2h!(buf, P, ranges, 1);
				end
				time_s = time() - t0
				GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
				if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
				@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
				GBs_ref = 2.8
				GBs_var = 0.5
				for it = 1:nt+warmup
					if (it == warmup+1) global t0 = time() end
					GG.read_h2h!(buf, P2, ranges, 1);
				end
				time_s = time() - t0
				GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
				if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
				@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
				# (dim=2)
				GBs_ref = 24
				GBs_var = 7
				buf = zeros(size(P,1), halowidths[2], size(P,3));
				ranges = [1:size(P,1), 3:3, 1:size(P,3)];
				for it = 1:nt+warmup
					if (it == warmup+1) global t0 = time() end
					GG.write_h2h!(buf, P, ranges, 2);
				end
				time_s = time() - t0
				GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
				if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
				@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
				for it = 1:nt+warmup
					if (it == warmup+1) global t0 = time() end
					GG.read_h2h!(buf, P2, ranges, 2);
				end
				time_s = time() - t0
				GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
				if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
				@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
				# (dim=3)
				GBs_ref = 57
				GBs_var = 4
				buf = zeros(size(P,1), size(P,2), halowidths[3]);
				ranges = [1:size(P,1), 1:size(P,2), 4:4];
				for it = 1:nt+warmup
					if (it == warmup+1) global t0 = time() end
					GG.write_h2h!(buf, P, ranges, 3);
				end
				time_s = time() - t0
				GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
				if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
				@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
				for it = 1:nt+warmup
					if (it == warmup+1) global t0 = time() end
					GG.read_h2h!(buf, P2, ranges, 3);
				end
				time_s = time() - t0
				GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
				if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
				@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
				finalize_global_grid(finalize_MPI=false);
			end;

			# NOTE: I have removed here many tests in order not to make this example too long.
			
			@static if test_gpu
				@testset "write_d2x! / write_d2h_async! / read_x2d! / read_h2d_async!" begin
					if me==0 println("write_d2x! / write_d2h_async! / read_x2d! / read_h2d_async!") end
					init_global_grid(nx, ny, nz, quiet=true, init_MPI=false);
					P  = zeros(nx,  ny,  nz  );
					P .= [iz*1e2 + iy*1e1 + ix for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
					P  = CuArray(P);
					halowidths = (1,1,1) # In unit tests: (1,3,1)
					nt = 10*512
					warmup = 10
					ntransfers = 1
					#TODO: to see how to define (based on bandwidthTest from NVIDIA?)
					GBs_ref = 10
					GBs_var = 3
					# (dim=1)
					dim = 1;
					P2  = cuzeros(eltype(P),size(P));
					buf = zeros(halowidths[1], size(P,2), size(P,3));
					buf_d, buf_h = GG.register(CuArray,buf);
					ranges = [2:2, 1:size(P,2), 1:size(P,3)];
					nthreads = (1, 32, 1);
	                halosize = [r[end] - r[1] + 1 for r in ranges];
					nblocks  = Tuple(ceil.(Int, halosize./nthreads));
					for it = 1:nt+warmup
						if (it == warmup+1) global t0 = time() end
	                	@cuda blocks=nblocks threads=nthreads GG.write_d2x!(buf_d, P, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
					end
					time_s = time() - t0
					GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
					if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
					@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
					for it = 1:nt+warmup
						if (it == warmup+1) global t0 = time() end
						@cuda blocks=nblocks threads=nthreads GG.read_x2d!(buf_d, P2, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
					end
					time_s = time() - t0
					GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
					if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
					@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
					# (async)
					#TODO: for dim == 1 _async is slow (1.7 GB/s -> that's why kernel launch is currently done for that case)
					GBs_ref = 1
					buf .= 0.0;
					P2  .= 0.0;
					custream = stream();
					for it = 1:nt+warmup
						if (it == warmup+1) global t0 = time() end
						GG.write_d2h_async!(buf, P, ranges, custream); CUDA.synchronize();
					end
					time_s = time() - t0
					GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
					if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
					@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
					for it = 1:nt+warmup
						if (it == warmup+1) global t0 = time() end
						GG.read_h2d_async!(buf, P2, ranges, custream); CUDA.synchronize();
					end
					time_s = time() - t0
					GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
					if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
					@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
					CUDA.Mem.unregister(buf_h);
					#TODO: for dim == 1 _async is slow (1.0 GB/s -> that's why kernel launch is currently done for that case)
					GBs_ref = 10
					# (dim=2)
					GBs_var = 5 # NOTE: for dim>1 this is not actually used, but the async version.
					dim = 2;
					P2  = cuzeros(eltype(P),size(P));
					buf = zeros(size(P,1), halowidths[2], size(P,3));
					buf_d, buf_h = GG.register(CuArray,buf);
					ranges = [1:size(P,1), 3:3, 1:size(P,3)];
					nthreads = (32, 1, 1);
					halosize = [r[end] - r[1] + 1 for r in ranges];
					nblocks  = Tuple(ceil.(Int, halosize./nthreads));
					for it = 1:nt+warmup
						if (it == warmup+1) global t0 = time() end
						@cuda blocks=nblocks threads=nthreads GG.write_d2x!(buf_d, P, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
					end
					time_s = time() - t0
					GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
					if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
					@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
					for it = 1:nt+warmup
						if (it == warmup+1) global t0 = time() end
						@cuda blocks=nblocks threads=nthreads GG.read_x2d!(buf_d, P2, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
					end
					time_s = time() - t0
					GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
					if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
					@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
					# (async)
					buf .= 0.0;
					P2  .= 0.0;
					custream = stream();
					for it = 1:nt+warmup
						if (it == warmup+1) global t0 = time() end
						GG.write_d2h_async!(buf, P, ranges, custream); CUDA.synchronize();
					end
					time_s = time() - t0
					GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
					if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
					@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
					for it = 1:nt+warmup
						if (it == warmup+1) global t0 = time() end
						GG.read_h2d_async!(buf, P2, ranges, custream); CUDA.synchronize();
					end
					time_s = time() - t0
					GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
					if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
					@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
					CUDA.Mem.unregister(buf_h);
					# (dim=3)
					dim = 3
					P2  = cuzeros(eltype(P),size(P));
					buf = zeros(size(P,1), size(P,2), halowidths[3]);
					buf_d, buf_h = GG.register(CuArray,buf);
					ranges = [1:size(P,1), 1:size(P,2), 4:4];
					nthreads = (32, 1, 1);
					halosize = [r[end] - r[1] + 1 for r in ranges];
					nblocks  = Tuple(ceil.(Int, halosize./nthreads));
					for it = 1:nt+warmup
						if (it == warmup+1) global t0 = time() end
						@cuda blocks=nblocks threads=nthreads GG.write_d2x!(buf_d, P, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
					end
					time_s = time() - t0
					GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
					if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
					@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
					for it = 1:nt+warmup
						if (it == warmup+1) global t0 = time() end
						@cuda blocks=nblocks threads=nthreads GG.read_x2d!(buf_d, P2, ranges[1], ranges[2], ranges[3], dim); CUDA.synchronize();
					end
					time_s = time() - t0
					GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
					if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
					@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
					# (async)
					buf .= 0.0;
					P2  .= 0.0;
					custream = stream();
					for it = 1:nt+warmup
						if (it == warmup+1) global t0 = time() end
						GG.write_d2h_async!(buf, P, ranges, custream); CUDA.synchronize();
					end
					time_s = time() - t0
					GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
					if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
					@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
					for it = 1:nt+warmup
						if (it == warmup+1) global t0 = time() end
						GG.read_h2d_async!(buf, P2, ranges, custream); CUDA.synchronize();
					end
					time_s = time() - t0
					GBs = 1.0/1024^3*nt*ntransfers*sizeof(buf)/time_s
					if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
					@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
					CUDA.Mem.unregister(buf_h);
					P  = nothing;
					P2 = nothing;
					finalize_global_grid(finalize_MPI=false);
				end;
			end
			GC.gc();
		end;

		# NOTE: I have removed here many tests in order not to make this example too long.

		@static if test_gpu
			@testset "iwrite_sendbufs! (allocator: $(longnameof(zeros)))" for zeros in allocators
				if me==0 println("iwrite_sendbufs!") end
				init_global_grid(nx, ny, nz, periodx=1, periody=1, periodz=1, overlaps=(2,2,3), quiet=true, init_MPI=false);
				P = zeros(nx,  ny,  nz  );
				A = zeros(nx-1,ny+2,nz+1);
				#TODO: shorten
				if zeros == cuzeros
					P = CuArray([iz*1e2 + iy*1e1 + ix for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)]);
					A = CuArray([iz*1e2 + iy*1e1 + ix for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)]);
				else
					P .= [iz*1e2 + iy*1e1 + ix for ix=1:size(P,1), iy=1:size(P,2), iz=1:size(P,3)];
					A .= [iz*1e2 + iy*1e1 + ix for ix=1:size(A,1), iy=1:size(A,2), iz=1:size(A,3)];
				end
				P, A = GG.wrap_field.((P, A));
				GG.allocate_bufs(P, A);
				if (zeros == cuzeros)
					GG.allocate_custreams(P, A);
				else
					GG.allocate_tasks(P, A);
				end
				nt = 10*512
				warmup = 10
				ntransfers = 2
				T = eltype(P);
				#TODO: to see how to define (based on bandwidthTest from NVIDIA?) (note that for dim=1, I got 18 GBs - dual copy of Tesla seems to work here)
				if zeros == cuzeros
					GBs_ref = 18
					GBs_var = 2.5
				else
					GBs_ref = 10  #TODO: to be revised when pipelining is available.
					GBs_var = 3
				end
				dim = 1
				n = 1
				for it = 1:nt+warmup
					if (it == warmup+1) global t0 = time() end
					# TODO: the perf is twice as high with 2 arrays, i.e. two concurrent kernels: 18 GB/s; however, the **theoretical** max is ~12.75 GB/s. So, something is wrongly measured - maybe measures until goes into buffer, but not until arrives at destination?
					GG.iwrite_sendbufs!(n, dim, P, 1);
					GG.iwrite_sendbufs!(n, dim, A, 2);
					GG.wait_iwrite(n, P, 1);
					GG.wait_iwrite(n, A, 2);
				end
				time_s = time() - t0
				GBs = 1.0/1024^3*nt*ntransfers*sizeof(T)*ny*nz/time_s
				#TODO: quick fix: see if all should be 18 GBs (or is something wrong here?) Rather this should not be 18; the timing must probably not include the full arrival of all data.
				if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
				@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
				n = 2
				for it = 1:nt+warmup
					if (it == warmup+1) global t0 = time() end
					# TODO: the perf is twice as high with 2 arrays, i.e. two concurrent kernels: 18 GB/s; however, the **theoretical** max is ~12.75 GB/s. So, something is wrongly measured - maybe measures until goes into buffer, but not until arrives at destination?
					GG.iwrite_sendbufs!(n, dim, P, 1);
					GG.iwrite_sendbufs!(n, dim, A, 2);
					GG.wait_iwrite(n, P, 1);
					GG.wait_iwrite(n, A, 2);
				end
				time_s = time() - t0
				GBs = 1.0/1024^3*nt*ntransfers*sizeof(T)*ny*nz/time_s
				#TODO: quick fix: see if all should be 18 GBs (or is something wrong here?)
				if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
				@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
				dim = 2
				if zeros == cuzeros
					GBs_ref = 12
					GBs_var = 2.5
				else
					GBs_ref = 20  #TODO: to be revised when pipelining is available.
					GBs_var = 2
				end
				n = 1
				for it = 1:nt+warmup
					if (it == warmup+1) global t0 = time() end
					GG.iwrite_sendbufs!(n, dim, P, 1);
					GG.iwrite_sendbufs!(n, dim, A, 2);
					GG.wait_iwrite(n, P, 1);
					GG.wait_iwrite(n, A, 2);
				end
				time_s = time() - t0
				GBs = 1.0/1024^3*nt*ntransfers*sizeof(T)*nx*nz/time_s
				if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
				@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
				n = 2
				for it = 1:nt+warmup
					if (it == warmup+1) global t0 = time() end
					GG.iwrite_sendbufs!(n, dim, P, 1);
					GG.iwrite_sendbufs!(n, dim, A, 2);
					GG.wait_iwrite(n, P, 1);
					GG.wait_iwrite(n, A, 2);
				end
				time_s = time() - t0
				GBs = 1.0/1024^3*nt*ntransfers*sizeof(T)*nx*nz/time_s
				if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
				@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
				dim = 3
				n = 1
				#TODO: for some reason this is currently a special case.
				if zeros != cuzeros
					GBs_ref = 33 #TODO: to be revised when pipelining is available.
					GBs_var = 3
				end
				for it = 1:nt+warmup
					if (it == warmup+1) global t0 = time() end
					GG.iwrite_sendbufs!(n, dim, P, 1);
					GG.iwrite_sendbufs!(n, dim, A, 2);
					GG.wait_iwrite(n, P, 1);
					GG.wait_iwrite(n, A, 2);
				end
				time_s = time() - t0
				GBs = 1.0/1024^3*nt*ntransfers*sizeof(T)*nx*ny/time_s
				if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
				@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
				n = 2
				for it = 1:nt+warmup
					if (it == warmup+1) global t0 = time() end
					GG.iwrite_sendbufs!(n, dim, P, 1);
					GG.iwrite_sendbufs!(n, dim, A, 2);
					GG.wait_iwrite(n, P, 1);
					GG.wait_iwrite(n, A, 2);
				end
				time_s = time() - t0
				GBs = 1.0/1024^3*nt*ntransfers*sizeof(T)*nx*ny/time_s
				if me==0 println("GBs: $GBs (GBs_ref=$GBs_ref, GBs_var=$GBs_var)") end
				@test (GBs > GBs_ref - GBs_var) && (GBs < GBs_ref + GBs_var)
				P = nothing;
				A = nothing;
				finalize_global_grid(finalize_MPI=false);
			end;
		end;

		# NOTE: I have removed here many tests in order not to make this example too long.
	end;
end;

## 5. Test tear down
MPI.Finalize()
